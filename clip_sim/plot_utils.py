import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.stats import spearmanr

def _make_stats_summary_text(values, name=""):
    return (
        f"{name} Stats:\n"
        f"Mean: {values.mean():.3f}\n"
        f"Std: {values.std():.3f}\n"
        f"Skew: {skew(values):.2f}\n"
        f"Kurtosis: {kurtosis(values):.2f}\n"
        f"Min: {values.min():.2f}, Max: {values.max():.2f}\n"
    )

def plot_similarity_vs_frequency(concept_freq_df, sim_col, save_path=None):
    valid_df = concept_freq_df.dropna(subset=[sim_col])
    corr = spearmanr(valid_df['freq_in_captions'], valid_df[sim_col])
    print(f"📊 Spearman correlation: {corr.correlation:.4f}, p = {corr.pvalue:.4e}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=valid_df, x='freq_in_captions', y=sim_col)
    plt.xscale('log')
    plt.title(f"Concept Frequency vs. {sim_col.replace('_', ' ').title()}")
    plt.xlabel("Log Frequency in Captions")
    plt.ylabel("Average CLIP-L14 Similarity")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Plot saved to {save_path}")
    plt.close()
    
def plot_descriptor_marginal_distributions_with_stats(
    df: pd.DataFrame,
    sim_col: str,
    freq_col: str,
    descriptor_name: str = "",
    save_dir: str = None,
    bins: int = 50,
):
    df = df.dropna(subset=[sim_col, freq_col]).copy()
    sim_values = df[sim_col]
    freq_values = df[freq_col]
    log_freq_values = np.log1p(freq_values)

    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    # === Similarity plot ===
    plt.figure(figsize=(6, 4))
    sns.histplot(sim_values, bins=bins, kde=True)
    plt.title(f"{descriptor_name} - CLIP Similarity Distribution")
    plt.xlabel("CLIP Similarity")
    plt.ylabel("Count")
    plt.tight_layout()
    summary_text = _make_stats_summary_text(sim_values, "Similarity")
    plt.gca().text(0.98, 0.98, summary_text, fontsize=9, va="top", ha="right",
                   transform=plt.gca().transAxes, bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    if save_dir:
        fname = os.path.join(save_dir, f"{descriptor_name}_similarity_dist.png")
        plt.savefig(fname)
        print(f"📈 Saved: {fname}")
    plt.close()

    # === Frequency plot (linear) ===
    plt.figure(figsize=(6, 4))
    sns.histplot(freq_values, bins=bins, kde=False)
    plt.title(f"{descriptor_name} - Concept Frequency (Linear)")
    plt.xlabel("Frequency in Captions")
    plt.ylabel("Count")
    plt.tight_layout()
    summary_text = _make_stats_summary_text(freq_values, "Frequency")
    plt.gca().text(0.98, 0.98, summary_text, fontsize=9, va="top", ha="right",
                   transform=plt.gca().transAxes, bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    if save_dir:
        fname = os.path.join(save_dir, f"{descriptor_name}_freq_dist_linear.png")
        plt.savefig(fname)
        print(f"📉 Saved: {fname}")
    plt.close()

    # === Frequency plot (log) ===
    plt.figure(figsize=(6, 4))
    sns.histplot(log_freq_values, bins=bins, kde=False)
    plt.title(f"{descriptor_name} - Concept Frequency (Log Scale)")
    plt.xlabel("log(1 + Frequency)")
    plt.ylabel("Count")
    plt.tight_layout()
    summary_text = _make_stats_summary_text(log_freq_values, "log(Frequency)")
    plt.gca().text(0.98, 0.98, summary_text, fontsize=9, va="top", ha="right",
                   transform=plt.gca().transAxes, bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    if save_dir:
        fname = os.path.join(save_dir, f"{descriptor_name}_freq_dist_log.png")
        plt.savefig(fname)
        print(f"📉 Saved: {fname}")
    plt.close()