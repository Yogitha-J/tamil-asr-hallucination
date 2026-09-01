# Hallucination in Speech and Audio-Language Models: A Cross-Paradigm Study on Tamil Transcription

Code and analysis scripts accompanying the paper of the same name, comparing
hallucination behavior across a general-purpose ASR model (Whisper), an
Indic-specialized ASR model (IndicConformer), and a general-purpose
audio-language model (Qwen2.5-Omni) on 7,475 Tamil utterances from the
Kathbath and IndicVoices corpora.

## Authors

Yogitha J, Pawar Akshata Mohan
Department of Computer Science, St. Joseph's College of Engineering, Chennai, India

## Repository structure

```
scripts/
  1_manifest_kathbath.py       Build Kathbath manifest (handles train/valid id collisions)
  2_manifest_indicvoices.py    Build IndicVoices manifest
  3_inference_harness.py       Resumable inference loop + model loading (Whisper, IndicConformer, Qwen)
  4_score_results.py           WER/CER + lexical-proxy classification, repetition-collapse detection
  5_rescore_labse.py           Replace lexical proxy with real LaBSE semantic similarity
  6_threshold_sensitivity_and_leakage_check.py
                                Threshold sensitivity analysis + Kathbath train/valid contamination check
  7_statistical_analysis.py    Speaker-level bootstrap CIs, correlation tests, paired significance tests
  8_generate_figures.py        Produces all four paper figures
  9_merge_human_validation.py  Merges human annotator labels against the automatic classifier

data_samples/
  kathbath_scored_results_labse.csv          Full scored results, Kathbath (720 utterances x 4 model/prompt conditions)
  indicvoices_scored_results_COMPLETE_labse.csv  Full scored results, IndicVoices (6,755 utterances x 4 conditions)
  bootstrap_ci_summary.csv                   Speaker-level bootstrap CI values underlying Table I
  human_validation_merged.csv                Merged automatic + human labels for all 200 validation samples
  human_validation_master_key_batches_ABC.csv  Master key linking review row IDs to true sample identity (Batches A-C)
  human_validation_master_key_batchD.csv     Master key for Batch D (Qwen normalized)

figures/
  (output directory for generated figures)
```

## Pipeline order

Run the scripts in numeric order. Each stage's output feeds the next:

1. Build manifests for both corpora (scripts 1-2).
2. Run inference for all three models, both Qwen prompt conditions (script 3).
   This step needs a GPU (we used free-tier Kaggle/Colab T4 instances) and
   takes considerably longer for IndicVoices (6,755 utterances) than
   Kathbath (720).
3. Score raw model output against references (script 4). This produces a
   lexical-proxy hallucination classification, which is a fast first pass
   but should not be treated as final.
4. Rescore with real LaBSE semantic similarity (script 5). This needs
   internet access to huggingface.co and replaces the lexical proxy with
   the numbers actually reported in the paper.
5. Run the threshold sensitivity check and the Kathbath contamination check
   (script 6) to confirm the reported model ordering is not an artifact of
   the specific classification thresholds chosen, and that IndicConformer's
   near-zero hallucination rate is not simply memorization of Kathbath's
   training split.
6. Run the full statistical analysis (script 7): speaker-level bootstrap
   confidence intervals, the repetition-collapse correlation analysis, and
   the paired verbatim-vs-normalized significance tests.
7. Generate figures (script 8).
8. If you are redoing the human validation step, use script 9 to merge
   annotator responses (from the Excel review batches) against the
   automatic classifier and compute agreement.

## A note on data

We do not redistribute the Kathbath or IndicVoices audio here. Both corpora
are publicly available from AI4Bharat under their own licenses:

- Kathbath / IndicSUPERB: https://github.com/AI4Bharat/IndicSUPERB
- IndicVoices: https://github.com/AI4Bharat/IndicVoices

Once you have downloaded the audio and `meta.jsonl` files from these
sources, point scripts 1 and 2 at their local paths.

## Known implementation pitfalls (see also paper Section III-F)

- **Qwen2.5-Omni quantization**: quantizing the full model to 4-bit,
  including the audio encoder, causes severe output degradation (repetition
  loops, script-switching, incoherence). The audio encoder (`audio_tower`)
  must be excluded from quantization via `llm_int8_skip_modules`. See
  `3_inference_harness.py`.
- **IndicConformer audio loading**: use `librosa`, not `torchaudio`, to load
  audio before passing it to the model. We hit a `libcudart.so.13` error
  from `torchaudio`'s compiled extension on Kaggle's CUDA environment;
  `librosa` loads cleanly in the same environment.
- **Kathbath identifier collisions**: 100 utterance ids appear in both the
  train and valid splits, pointing to different recordings. `4_score_results.py`
  resolves this via positional disambiguation, verified against actual
  transcript content; see the module docstring for the exact reasoning.

## Requirements

See `requirements.txt`. We did not pin every dependency version
consistently across every session in which results were generated (see
paper Limitations); we recommend pinning versions from the start if
extending this work.

## Citation

If you use this code, please cite the paper (see the main repository or
contact the authors for the current citation format).

## License

Code in this repository is released under the MIT License (see `LICENSE`).
This does not apply to the Kathbath or IndicVoices corpora themselves,
which retain their own licenses from AI4Bharat.

## Reproducing the Analysis

Run commands from the repository root.

```bash
pip install -r requirements.txt

python scripts/6_threshold_sensitivity_and_leakage_check.py
python scripts/7_statistical_analysis.py
python scripts/8_generate_figures.py
```

The published analysis inputs and merged human-validation results are provided
under `data_samples/`. Raw audio, private annotation master keys, and internal
evaluation workbooks are intentionally excluded from the public repository.

Script 9 (`9_merge_human_validation.py`) is retained as a research workflow
reference, but requires the private annotation files used during manual review
and therefore is not directly reproducible from the public repository.

## Public Data and Licensing

Before redistributing any corpus-derived reference/prediction text, users should
verify the redistribution terms of the underlying datasets. This repository
does not include raw audio files.
