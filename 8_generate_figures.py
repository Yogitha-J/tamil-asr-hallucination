from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_samples"

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
})

# Grayscale-safe, colorblind-friendly palette (distinct in both color and print)
COLOR_KATHBATH = "#4C72B0"
COLOR_INDICVOICES = "#DD8452"
COLOR_FAITHFUL = "#55A868"
COLOR_RECOGERR = "#DD8452"
COLOR_HALLU = "#C44E52"
COLOR_EMPTY = "#8172B2"

HATCH_KATHBATH = ""
HATCH_INDICVOICES = "//"

kb = pd.read_csv(DATA / "kathbath_scored_results_labse.csv")
iv = pd.read_csv(DATA / "indicvoices_scored_results_COMPLETE_labse.csv")

def extract_speaker(sample_id, corpus):
    if corpus == "kathbath":
        return sample_id.rsplit("_", 1)[0]
    m = re.match(r"(S\d+)_", sample_id)
    return m.group(1) if m else sample_id

kb["speaker_id"] = kb["sample_id"].apply(lambda x: extract_speaker(x, "kathbath"))
iv["speaker_id"] = iv["sample_id"].apply(lambda x: extract_speaker(x, "indicvoices"))

common_cols = ["sample_id","speaker_id","source_corpus","model_name","prompt_version","reference",
               "prediction","wer","cer","semantic_similarity","label_semantic","repetition_flag","duration",
               "has_unintelligible_tag"]
combined = pd.concat([kb[common_cols], iv[common_cols]], ignore_index=True)

MODEL_ORDER = [
    ("IndicConformer-600M-multilingual", "dedicated_asr", "IndicConformer"),
    ("Whisper-large-v3-turbo", "dedicated_asr", "Whisper"),
    ("Qwen2.5-Omni-7B-bnb4bit", "verbatim_v1", "Qwen\n(verbatim)"),
    ("Qwen2.5-Omni-7B-bnb4bit", "normalized_v1", "Qwen\n(normalized)"),
]

# ============================================================
# FIGURE 1: Main results — WER and Hallucination rate by model, grouped by corpus
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

x = np.arange(len(MODEL_ORDER))
width = 0.35

for ax, metric, title, ylabel in [
    (axes[0], "wer", "(a) Word Error Rate", "Mean WER"),
    (axes[1], "hallu", "(b) Hallucination Rate", "Hallucination Rate"),
]:
    kb_vals, iv_vals = [], []
    for model, prompt, _ in MODEL_ORDER:
        for corpus, vals in [("kathbath", kb_vals), ("indicvoices", iv_vals)]:
            sub = combined[(combined["model_name"]==model) & (combined["prompt_version"]==prompt) & (combined["source_corpus"]==corpus)]
            if metric == "wer":
                vals.append(sub["wer"].mean())
            else:
                vals.append((sub["label_semantic"]=="HALLUCINATION_CANDIDATE").mean())

    ax.bar(x - width/2, kb_vals, width, label="Kathbath", color=COLOR_KATHBATH, hatch=HATCH_KATHBATH, edgecolor="black", linewidth=0.6)
    ax.bar(x + width/2, iv_vals, width, label="IndicVoices", color=COLOR_INDICVOICES, hatch=HATCH_INDICVOICES, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([m[2] for m in MODEL_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if metric == "hallu":
        ax.set_ylim(0, 1.05)

axes[0].legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig("fig1_main_results.png", bbox_inches="tight")
plt.savefig("fig1_main_results.pdf", bbox_inches="tight")
plt.close()
print("Saved fig1_main_results.{png,pdf}")

# ============================================================
# FIGURE 2: Stacked bar — error category distribution per model x corpus
# ============================================================
fig, ax = plt.subplots(figsize=(7.16, 3.8))

labels_order = ["FAITHFUL", "PARAPHRASE_OR_RECOGNITION_ERROR", "HALLUCINATION_CANDIDATE", "EMPTY_OUTPUT"]
label_colors = {"FAITHFUL": COLOR_FAITHFUL, "PARAPHRASE_OR_RECOGNITION_ERROR": COLOR_RECOGERR,
                "HALLUCINATION_CANDIDATE": COLOR_HALLU, "EMPTY_OUTPUT": COLOR_EMPTY}
label_display = {"FAITHFUL": "Faithful", "PARAPHRASE_OR_RECOGNITION_ERROR": "Recognition error",
                  "HALLUCINATION_CANDIDATE": "Hallucination", "EMPTY_OUTPUT": "Empty output"}

bar_positions = []
bar_labels = []
pos = 0
data_by_bar = []
for model, prompt, short_name in MODEL_ORDER:
    for corpus in ["kathbath", "indicvoices"]:
        sub = combined[(combined["model_name"]==model) & (combined["prompt_version"]==prompt) & (combined["source_corpus"]==corpus)]
        pcts = sub["label_semantic"].value_counts(normalize=True) * 100
        data_by_bar.append({lbl: pcts.get(lbl, 0) for lbl in labels_order})
        corpus_short = "KB" if corpus == "kathbath" else "IV"
        clean_name = short_name.replace(chr(10), " ")
        bar_labels.append(f"{clean_name} ({corpus_short})")
        bar_positions.append(pos)
        pos += 1

bottoms = np.zeros(len(data_by_bar))
for lbl in labels_order:
    vals = [d[lbl] for d in data_by_bar]
    ax.bar(bar_positions, vals, bottom=bottoms, label=label_display[lbl], color=label_colors[lbl], edgecolor="black", linewidth=0.4, width=0.7)
    bottoms += np.array(vals)

ax.set_xticks(bar_positions)
ax.set_xticklabels(bar_labels, fontsize=7, rotation=30, ha="right")
ax.set_ylabel("Percentage of samples")
ax.set_ylim(0, 100)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.subplots_adjust(bottom=0.28, top=0.90)
plt.savefig("fig2_error_distribution.png", bbox_inches="tight")
plt.savefig("fig2_error_distribution.pdf", bbox_inches="tight")
plt.close()
print("Saved fig2_error_distribution.{png,pdf}")

# ============================================================
# FIGURE 3: Repetition-collapse rate by duration bin (Whisper, IndicVoices)
# ============================================================
fig, ax = plt.subplots(figsize=(3.4, 3.0))

whisper_iv = combined[(combined["model_name"]=="Whisper-large-v3-turbo") & (combined["source_corpus"]=="indicvoices")].copy()
bins = [0, 5, 10, 15, 20, 30]
bin_labels = ["0-5s", "5-10s", "10-15s", "15-20s", "20-30s"]
whisper_iv["duration_bin"] = pd.cut(whisper_iv["duration"], bins=bins, labels=bin_labels)

rep_rate_by_bin = whisper_iv.groupby("duration_bin", observed=True)["repetition_flag"].mean() * 100
counts_by_bin = whisper_iv.groupby("duration_bin", observed=True).size()

bars = ax.bar(range(len(rep_rate_by_bin)), rep_rate_by_bin.values, color=COLOR_HALLU, edgecolor="black", linewidth=0.6)
ax.set_xticks(range(len(rep_rate_by_bin)))
ax.set_xticklabels(rep_rate_by_bin.index, rotation=0)
ax.set_ylabel("Repetition-collapse rate (%)")
ax.set_xlabel("Utterance duration")
ax.set_title("Whisper repetition-collapse\nvs. duration (IndicVoices)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for i, (bar, n) in enumerate(zip(bars, counts_by_bin.values)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"n={n}", ha="center", fontsize=6.5)

plt.tight_layout()
plt.savefig("fig3_repetition_vs_duration.png", bbox_inches="tight")
plt.savefig("fig3_repetition_vs_duration.pdf", bbox_inches="tight")
plt.close()
print("Saved fig3_repetition_vs_duration.{png,pdf}")

# ============================================================
# FIGURE 4: Qwen verbatim vs normalized — paired comparison
# ============================================================
fig, ax = plt.subplots(figsize=(3.6, 3.2))

x = np.arange(2)
width = 0.35
verb_vals, norm_vals = [], []
for corpus in ["kathbath", "indicvoices"]:
    v = combined[(combined["model_name"]=="Qwen2.5-Omni-7B-bnb4bit") & (combined["prompt_version"]=="verbatim_v1") & (combined["source_corpus"]==corpus)]
    n = combined[(combined["model_name"]=="Qwen2.5-Omni-7B-bnb4bit") & (combined["prompt_version"]=="normalized_v1") & (combined["source_corpus"]==corpus)]
    verb_vals.append((v["label_semantic"]=="HALLUCINATION_CANDIDATE").mean() * 100)
    norm_vals.append((n["label_semantic"]=="HALLUCINATION_CANDIDATE").mean() * 100)

bars1 = ax.bar(x - width/2, verb_vals, width, label="Verbatim prompt", color=COLOR_HALLU, edgecolor="black", linewidth=0.6)
bars2 = ax.bar(x + width/2, norm_vals, width, label="Normalized prompt", color="#F4A582", edgecolor="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(["Kathbath", "IndicVoices"])
ax.set_ylabel("Hallucination rate (%)")
ax.set_ylim(0, 112)
ax.set_title("Qwen2.5-Omni: prompt effect\non hallucination rate", pad=8)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}", ha="center", fontsize=6.5)

# significance markers, well clear of the bars and value labels
for i, p in enumerate([0.037, 0.005]):
    marker = f"p={p:.3f} *" if p < 0.05 else "n.s."
    y = max(verb_vals[i], norm_vals[i]) + 8
    ax.text(i, y, marker, ha="center", fontsize=7)

plt.tight_layout()
plt.savefig("fig4_prompt_comparison.png", bbox_inches="tight")
plt.savefig("fig4_prompt_comparison.pdf", bbox_inches="tight")
plt.close()
print("Saved fig4_prompt_comparison.{png,pdf}")

print("\nAll figures generated successfully.")
