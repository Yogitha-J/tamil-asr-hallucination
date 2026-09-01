"""
Replaces the lexical-overlap proxy from 4_score_results.py with real
semantic similarity from LaBSE embeddings, and reclassifies accordingly.

Run this on a machine with internet access to huggingface.co (e.g. Kaggle
or Colab) since it needs to download the LaBSE model on first use.
"""

from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, util


def rescore_with_labse(csv_path, out_csv, wer_threshold=0.5, sim_threshold=0.55):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    model = SentenceTransformer("sentence-transformers/LaBSE")

    refs = df["reference"].fillna("").tolist()
    preds = df["prediction"].fillna("").tolist()

    ref_emb = model.encode(refs, convert_to_tensor=True, batch_size=64, show_progress_bar=True)
    pred_emb = model.encode(preds, convert_to_tensor=True, batch_size=64, show_progress_bar=True)
    df["semantic_similarity"] = util.pairwise_cos_sim(ref_emb, pred_emb).cpu().numpy()

    def classify_real(row):
        if pd.isna(row["wer"]):
            return "EMPTY_OUTPUT"
        if row["wer"] <= wer_threshold:
            return "FAITHFUL"
        elif row["semantic_similarity"] >= sim_threshold:
            return "PARAPHRASE_OR_RECOGNITION_ERROR"
        else:
            return "HALLUCINATION_CANDIDATE"

    df["label_semantic"] = df.apply(classify_real, axis=1)
    df.to_csv(out_csv, index=False)

    summary = df.groupby(["model_name", "prompt_version"]).agg(
        n=("sample_id", "count"),
        mean_wer=("wer", "mean"),
        mean_cer=("cer", "mean"),
        mean_semantic_sim=("semantic_similarity", "mean"),
    ).reset_index()
    print(summary)

    label_pcts = df.groupby(["model_name", "prompt_version", "label_semantic"]).size().unstack(fill_value=0)
    label_pcts = label_pcts.div(label_pcts.sum(axis=1), axis=0) * 100
    print(label_pcts)

    return df


if __name__ == "__main__":
    rescore_with_labse("kathbath_scored_results.csv", "kathbath_scored_results_labse.csv")
    rescore_with_labse("indicvoices_scored_results.csv", "indicvoices_scored_results_labse.csv")
