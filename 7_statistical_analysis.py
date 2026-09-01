from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_samples"

import pandas as pd
import numpy as np
import re

np.random.seed(42)

kb = pd.read_csv(DATA / "kathbath_scored_results_labse.csv")
iv = pd.read_csv(DATA / "indicvoices_scored_results_COMPLETE_labse.csv")

def extract_speaker(sample_id, corpus):
    if corpus == "kathbath":
        return sample_id.rsplit("_", 1)[0]
    else:
        m = re.match(r"(S\d+)_", sample_id)
        return m.group(1) if m else sample_id

kb["speaker_id"] = kb["sample_id"].apply(lambda x: extract_speaker(x, "kathbath"))
iv["speaker_id"] = iv["sample_id"].apply(lambda x: extract_speaker(x, "indicvoices"))

common_cols = ["sample_id","speaker_id","source_corpus","model_name","prompt_version","reference",
               "prediction","wer","cer","semantic_similarity","label_semantic","repetition_flag",
               "duration","has_unintelligible_tag","split"]
kb["split"] = kb["split"]  # already correctly resolved per-row from the earlier disambiguation
iv["split"] = None
combined = pd.concat([kb[common_cols], iv[common_cols]], ignore_index=True)

print(f"Total rows: {len(combined)}")
print(f"Unique speakers - Kathbath: {kb['speaker_id'].nunique()}, IndicVoices: {iv['speaker_id'].nunique()}")
print()

# ============================================================
# PART 1: Speaker-level bootstrap CIs for headline metrics
# ============================================================

def speaker_bootstrap(df, metric_col, n_boot=2000, is_binary=False):
    """Bootstrap by resampling speakers (with replacement), then all their utterances."""
    speakers = df["speaker_id"].unique()
    n_speakers = len(speakers)
    speaker_groups = {s: df[df["speaker_id"] == s][metric_col].values for s in speakers}

    boot_means = []
    for _ in range(n_boot):
        sampled_speakers = np.random.choice(speakers, size=n_speakers, replace=True)
        vals = np.concatenate([speaker_groups[s] for s in sampled_speakers])
        boot_means.append(np.nanmean(vals))
    boot_means = np.array(boot_means)
    point_estimate = np.nanmean(df[metric_col].values)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return point_estimate, ci_low, ci_high, n_speakers

print("="*100)
print("SPEAKER-LEVEL BOOTSTRAP 95% CONFIDENCE INTERVALS")
print("="*100)

results_summary = []
for corpus in ["kathbath", "indicvoices"]:
    for (model, prompt), grp in combined[combined["source_corpus"]==corpus].groupby(["model_name","prompt_version"]):
        grp = grp.copy()
        grp["is_hallucination"] = (grp["label_semantic"] == "HALLUCINATION_CANDIDATE").astype(float)

        wer_mean, wer_lo, wer_hi, n_spk = speaker_bootstrap(grp, "wer")
        hallu_mean, hallu_lo, hallu_hi, _ = speaker_bootstrap(grp, "is_hallucination")

        print(f"\n{corpus} | {model} | {prompt}  (n_utt={len(grp)}, n_speakers={n_spk})")
        print(f"  WER:  {wer_mean:.3f}  [95% CI: {wer_lo:.3f} - {wer_hi:.3f}]")
        print(f"  Hallucination rate: {hallu_mean:.3f}  [95% CI: {hallu_lo:.3f} - {hallu_hi:.3f}]")

        results_summary.append({
            "corpus": corpus, "model": model, "prompt": prompt, "n_utterances": len(grp), "n_speakers": n_spk,
            "wer_mean": wer_mean, "wer_ci_low": wer_lo, "wer_ci_high": wer_hi,
            "hallucination_rate": hallu_mean, "hallu_ci_low": hallu_lo, "hallu_ci_high": hallu_hi,
        })

pd.DataFrame(results_summary).to_csv("bootstrap_ci_summary.csv", index=False)

# ============================================================
# PART 2: Repetition-collapse correlation with duration / unintelligible tag
# ============================================================
print("\n" + "="*100)
print("REPETITION-COLLAPSE CORRELATION ANALYSIS (Whisper)")
print("="*100)

whisper = combined[(combined["model_name"]=="Whisper-large-v3-turbo")]

for corpus in ["kathbath", "indicvoices"]:
    sub = whisper[whisper["source_corpus"]==corpus]
    rep_true = sub[sub["repetition_flag"]==True]
    rep_false = sub[sub["repetition_flag"]==False]

    print(f"\n{corpus} (n={len(sub)}, repetition cases={len(rep_true)})")
    if len(rep_true) > 0:
        print(f"  Mean duration | repetition=True:  {rep_true['duration'].mean():.2f}s")
        print(f"  Mean duration | repetition=False: {rep_false['duration'].mean():.2f}s")

        unintell_rate_rep = rep_true["has_unintelligible_tag"].mean()
        unintell_rate_norep = rep_false["has_unintelligible_tag"].mean()
        print(f"  <unintelligible> tag rate | repetition=True:  {unintell_rate_rep:.3f}")
        print(f"  <unintelligible> tag rate | repetition=False: {unintell_rate_norep:.3f}")

        # Point-biserial correlation (repetition binary vs duration continuous)
        from scipy import stats
        corr, pval = stats.pointbiserialr(sub["repetition_flag"].astype(int), sub["duration"])
        print(f"  Point-biserial correlation (repetition vs duration): r={corr:.3f}, p={pval:.4f}")

        # Chi-square: repetition vs unintelligible tag
        ct = pd.crosstab(sub["repetition_flag"], sub["has_unintelligible_tag"])
        chi2, chi_p, dof, exp = stats.chi2_contingency(ct)
        print(f"  Chi-square (repetition vs unintelligible tag): chi2={chi2:.2f}, p={chi_p:.4f}")
    else:
        print("  No repetition cases in this corpus for this model.")

# ============================================================
# PART 3: Verbatim vs Normalized prompt comparison for Qwen (paired)
# ============================================================
print("\n" + "="*100)
print("QWEN VERBATIM vs NORMALIZED — PAIRED COMPARISON")
print("="*100)

from scipy import stats as sstats

for corpus in ["kathbath", "indicvoices"]:
    verb = combined[(combined["model_name"]=="Qwen2.5-Omni-7B-bnb4bit") &
                     (combined["prompt_version"]=="verbatim_v1") &
                     (combined["source_corpus"]==corpus)].copy()
    norm = combined[(combined["model_name"]=="Qwen2.5-Omni-7B-bnb4bit") &
                     (combined["prompt_version"]=="normalized_v1") &
                     (combined["source_corpus"]==corpus)].copy()

    if corpus == "kathbath":
        # sample_id collides across train/valid splits (100 known collisions).
        # Each row already carries its OWN correctly-resolved split (from earlier disambiguation) -
        # use that directly rather than re-deriving a sample_id->split mapping, which would collide again.
        verb["join_key"] = verb["sample_id"] + "__" + verb["split"].astype(str)
        norm["join_key"] = norm["sample_id"] + "__" + norm["split"].astype(str)
    else:
        verb["join_key"] = verb["sample_id"]
        norm["join_key"] = norm["sample_id"]

    verb = verb.drop_duplicates(subset="join_key").set_index("join_key")
    norm = norm.drop_duplicates(subset="join_key").set_index("join_key")

    common_ids = verb.index.intersection(norm.index)
    verb_paired = verb.loc[common_ids]
    norm_paired = norm.loc[common_ids]

    print(f"\n{corpus}: {len(common_ids)} paired samples")

    verb_hallu = (verb_paired["label_semantic"]=="HALLUCINATION_CANDIDATE").astype(int)
    norm_hallu = (norm_paired["label_semantic"]=="HALLUCINATION_CANDIDATE").astype(int)

    print(f"  Verbatim hallucination rate:   {verb_hallu.mean():.3f}")
    print(f"  Normalized hallucination rate: {norm_hallu.mean():.3f}")

    # McNemar's test for paired binary outcomes
    table = pd.crosstab(verb_hallu, norm_hallu)
    print("  Contingency table (rows=verbatim, cols=normalized):")
    print(table)
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        result = mcnemar(table.values, exact=(table.values.sum() < 25))
        print(f"  McNemar's test: statistic={result.statistic:.3f}, p={result.pvalue:.4f}")
    except ImportError:
        print("  (statsmodels not available for McNemar's test)")

    # Wilcoxon signed-rank on WER (paired, continuous) - drop rows where either side has missing WER (empty outputs)
    wer_pairs = pd.DataFrame({"verb": verb_paired["wer"].values, "norm": norm_paired["wer"].values}).dropna()
    n_dropped = len(verb_paired) - len(wer_pairs)
    wstat, wp = sstats.wilcoxon(wer_pairs["verb"], wer_pairs["norm"])
    print(f"  Mean WER (verbatim): {verb_paired['wer'].mean():.3f}, Mean WER (normalized): {norm_paired['wer'].mean():.3f}")
    print(f"  (dropped {n_dropped} pairs with missing WER due to empty output on at least one side)")
    print(f"  Wilcoxon signed-rank test on WER: statistic={wstat:.1f}, p={wp:.4f}")
