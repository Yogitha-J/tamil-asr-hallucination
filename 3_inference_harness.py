"""
Inference harness used for all three models (Whisper, IndicConformer, Qwen2.5-Omni).

run_benchmark() is resumable: it checks which sample_ids already have a
result in out_path and skips them, so an interrupted run (Kaggle/Colab
session timeout, disconnect, etc.) can be restarted without losing progress
or reprocessing completed samples. It also checkpoints in batches (default
1000) with an explicit flush + fsync at the end of each batch.
"""

import os
import json
import time
import torch
from datetime import datetime, timezone


def run_benchmark(samples, transcribe_fn, model_name, model_version, prompt_version,
                   out_path, batch_size=1000, empty_cache_every=50):
    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["sample_id"])
                except json.JSONDecodeError:
                    continue
        print(f"Resuming: {len(done_ids)} already done.")

    remaining = [s for s in samples if s["sample_id"] not in done_ids]
    print(f"Remaining to process: {len(remaining)}")

    with open(out_path, "a") as fout:
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start: batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            print(f"\n--- Starting batch {batch_num} ({len(batch)} samples) ---")

            for i, sample in enumerate(batch):
                try:
                    t0 = time.time()
                    raw = transcribe_fn(sample["audio_path"])
                    elapsed = time.time() - t0
                    result = {
                        "sample_id": sample["sample_id"],
                        "source_corpus": sample["source_corpus"],
                        "speaker_id": sample["speaker_id"],
                        "model_name": model_name,
                        "model_version": model_version,
                        "prompt_version": prompt_version,
                        "raw_response": raw,
                        "cleaned_response": raw.strip(),
                        "inference_time_sec": round(elapsed, 3),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                except Exception as e:
                    fout.write(json.dumps({
                        "sample_id": sample["sample_id"], "error": str(e),
                        "model_name": model_name, "prompt_version": prompt_version,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }, ensure_ascii=False) + "\n")

                if i % empty_cache_every == 0:
                    torch.cuda.empty_cache()
                if i % 200 == 0:
                    print(f"[{model_name}] batch {batch_num}: {i}/{len(batch)} done")

            fout.flush()
            os.fsync(fout.fileno())
            print(f"--- Batch {batch_num} complete. ---")

    print(f"\nFinished {model_name}: results in {out_path}")


# ============================================================
# Tier 1: Whisper large-v3-turbo
# ============================================================
def load_whisper():
    import librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    whisper_id = "openai/whisper-large-v3-turbo"
    processor = WhisperProcessor.from_pretrained(whisper_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        whisper_id, torch_dtype=torch.float16
    ).to("cuda")
    forced_ids = processor.get_decoder_prompt_ids(language="ta", task="transcribe")

    def transcribe(audio_path):
        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda").half()
        with torch.inference_mode():
            ids = model.generate(inputs, forced_decoder_ids=forced_ids, max_new_tokens=256)
        return processor.batch_decode(ids, skip_special_tokens=True)[0]

    return transcribe, whisper_id


# ============================================================
# Tier 2: AI4Bharat IndicConformer (Tamil mode)
# ============================================================
def load_indicconformer():
    import librosa
    from transformers import AutoModel

    indic_id = "ai4bharat/indic-conformer-600m-multilingual"
    model = AutoModel.from_pretrained(indic_id, trust_remote_code=True).to("cuda")

    def transcribe(audio_path):
        # NOTE: use librosa, not torchaudio, for audio loading here.
        # torchaudio's compiled extension raised a libcudart.so.13 error
        # on Kaggle's CUDA 12.x environment; librosa loads cleanly.
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).unsqueeze(0).to("cuda")
        with torch.inference_mode():
            text = model(wav_tensor, "ta", "ctc")
        return text

    return transcribe, indic_id


# ============================================================
# Tier 3: Qwen2.5-Omni-7B, 4-bit NF4, audio encoder excluded from quantization
# ============================================================
def load_qwen():
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig
    from qwen_omni_utils import process_mm_info

    qwen_id = "Qwen/Qwen2.5-Omni-7B"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        # CRITICAL: without this, the audio encoder gets quantized along with
        # the language backbone, which causes severe output degradation
        # (repetition loops, script-switching, incoherence). See paper Section III-F.
        llm_int8_skip_modules=["audio_tower"],
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(qwen_id)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        qwen_id, quantization_config=bnb_config, device_map="cuda", low_cpu_mem_usage=True
    )
    model.disable_talker()

    SYSTEM_PROMPT = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )

    def make_transcribe(prompt_text):
        def transcribe(audio_path):
            full_prompt = prompt_text + " Output ONLY the transcription text, with no explanation, commentary, or questions."
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": full_prompt},
                ]},
            ]
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            inputs = processor(text=text, audio=audios, images=images, videos=videos,
                                return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                out_ids = model.generate(**inputs, return_audio=False, do_sample=False,
                                          repetition_penalty=1.1, max_new_tokens=128)
            out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            result = out_text.split(full_prompt)[-1].strip()
            if result.lower().startswith("assistant"):
                result = result[len("assistant"):].strip()
            return result
        return transcribe

    PROMPTS = {
        "verbatim_v1": "Transcribe only the speech present in the audio. Preserve the spoken Tamil wording as closely as possible. Do not summarize, translate, infer, complete, or add information that is not spoken.",
        "normalized_v1": "Transcribe the spoken Tamil into standard written Tamil while preserving the meaning and content of the speech. Do not add information.",
    }

    return make_transcribe, PROMPTS, qwen_id
