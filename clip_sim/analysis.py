import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import faiss
from scipy.stats import spearmanr, linregress
from utils.get_embeddings import cache_clip_embeddings
from clip_sim.plot_utils import plot_similarity_vs_frequency

def save_similarity_and_count_stats(
    concept_df,
    sim_col,
    count_col,
    save_dir,
    prefix=None
):
    """Save raw data, similarity histogram, and count histogram."""
    os.makedirs(save_dir, exist_ok=True)
    prefix = prefix or sim_col
    stats_path_prefix = os.path.join(save_dir, f"{prefix}_stats")

    # Save raw dataframe
    # concept_df.to_csv(f"{stats_path_prefix}_raw.csv", index=False)
    concept_df.to_json(f"{stats_path_prefix}_raw.json", orient="records", indent=2)

    # Similarity histogram
    if sim_col in concept_df.columns:
        sim_values = concept_df[sim_col].dropna()
        if len(sim_values) > 0:
            plt.figure(figsize=(6, 4))
            sns.histplot(sim_values, bins=40, kde=True)
            plt.title(f"Distribution of {sim_col}")
            plt.xlabel("Average Similarity")
            plt.tight_layout()
            plt.savefig(f"{stats_path_prefix}_similarity_hist.png")
            print(f"📈 Saved similarity histogram to {stats_path_prefix}_similarity_hist.png")
            plt.close()

    # Count histogram
    if count_col in concept_df.columns:
        count_values = concept_df[count_col]
        if len(count_values) > 0:
            plt.figure(figsize=(6, 4))
            sns.histplot(count_values, bins=40)
            plt.title("Distribution of Matched Caption Counts")
            plt.xlabel("Matched Captions")
            plt.tight_layout()
            plt.savefig(f"{stats_path_prefix}_matchcount_hist.png")
            print(f"📉 Saved match count histogram to {stats_path_prefix}_matchcount_hist.png")
            plt.close()


def summarize_descriptor_alignment_stats(
    df: pd.DataFrame,
    sim_col: str,
    freq_col: str,
    descriptor_name: str,
    log_transform_freq: bool = True,
    freq_bin_quantiles: tuple = (0.2, 0.8),
    verbose: bool = True,
):
    df = df.dropna(subset=[sim_col, freq_col]).copy()
    df["log_freq"] = np.log1p(df[freq_col]) if log_transform_freq else df[freq_col]

    # === Core stats
    sim_mean = df[sim_col].mean()
    sim_std = df[sim_col].std()
    match_rate = (df[freq_col] > 0).mean()

    spearman_corr, spearman_p = spearmanr(df[freq_col], df[sim_col])
    slope, intercept, r_value, p_value, stderr = linregress(df["log_freq"], df[sim_col])
    r2 = r_value ** 2

    # === Frequency bin-based variance analysis
    low_thresh = df[freq_col].quantile(freq_bin_quantiles[0])
    high_thresh = df[freq_col].quantile(freq_bin_quantiles[1])

    low_var = df.loc[df[freq_col] <= low_thresh, sim_col].var()
    high_var = df.loc[df[freq_col] >= high_thresh, sim_col].var()

    summary_dict = {
        "name": descriptor_name,
        "sim_mean": sim_mean,
        "sim_std": sim_std,
        "match_rate": match_rate,
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p,
        "regression_slope": slope,
        "regression_intercept": intercept,
        "regression_r2": r2,
        "low_freq_variance": low_var,
        "high_freq_variance": high_var,
    }

    if verbose:
        print(f"\n📊 Descriptor Set: {descriptor_name}")
        print(f"- Mean similarity:       {sim_mean:.4f}")
        print(f"- Std similarity:        {sim_std:.4f}")
        print(f"- Spearman corr:         {spearman_corr:.4f} (p={spearman_p:.1e})")
        print(f"- Regression: sim = {slope:.4f} * log(freq+1) + {intercept:.4f}")
        print(f"- R²:                    {r2:.4f}")
        print(f"- Match rate:            {match_rate*100:.1f}% (nonzero freq concepts)")
        print(f"- Variance (low freq):   {low_var:.4f}")
        print(f"- Variance (high freq):  {high_var:.4f}")
        print(json.dumps(summary_dict, indent=4))

    return summary_dict

def run_clip_faiss_matching(
    df, 
    descriptors, 
    save_dir, 
    sim_col_name="avg_sim_clip", 
    sim_threshold=0.3, 
    top_k=50, 
    force=True,
    caption_embs=None,
    model=None,
    tokenizer=None,
    device="cuda"
):
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, f"{sim_col_name}_results.json")

    if os.path.exists(result_path) and not force:
        print(f"🔁 Loading cached results from {result_path}")
        return pd.read_json(result_path)

    print("🔍 Running full CLIP + FAISS pipeline...")

    # Embed all caption texts and descriptors
    clip_texts = df['text'].tolist()
    clip_sims = df['clip_l14_similarity_score'].tolist()
    if len(clip_texts) == 0 or len(descriptors) == 0:
        print(len(df), len(df['text']), len(df['clip_l14_similarity_score']))   
        import pdb; pdb.set_trace()
    
    if caption_embs is None:
        caption_embs = cache_clip_embeddings(clip_texts, model, tokenizer, device=device, save_dir="caption")
    descriptor_embs = cache_clip_embeddings(descriptors, model, tokenizer, device=device, save_dir="descriptor").astype(np.float32)

    # Build FAISS index
    caption_embs = caption_embs.astype(np.float32)
    faiss.normalize_L2(caption_embs)
    index = faiss.IndexFlatIP(caption_embs.shape[1])
    index.add(caption_embs)

    D, I = index.search(descriptor_embs, top_k)

    avg_scores, match_counts = [], []
    for sims, idxs in zip(D, I):
        valid_sims = [clip_sims[j] for j, sim in zip(idxs, sims) if sim >= sim_threshold and pd.notna(clip_sims[j])]
        match_counts.append(len(valid_sims))
        avg_scores.append(np.mean(valid_sims) if valid_sims else np.nan)
    print(f"Found {len(avg_scores)} valid scores")

    # Save results
    concept_freq_df = pd.DataFrame({
        'concept': descriptors,
        'freq_in_captions': match_counts,
        sim_col_name: avg_scores
    })

    print(f"✅ Saved results to {result_path}")
    concept_freq_df.to_json(result_path, orient="records", indent=2)

    # Plotting
    plot_path = os.path.join(save_dir, f"concept_frequency_vs_{sim_col_name}.png")
    plot_similarity_vs_frequency(concept_freq_df, sim_col_name, plot_path)

    save_similarity_and_count_stats(
        concept_freq_df,
        sim_col=sim_col_name,
        count_col='freq_in_captions',
        save_dir=save_dir,
        prefix=sim_col_name
    )

    return concept_freq_df

