import os
import json
import time
import torch
import pandas as pd
from collections import defaultdict
from utils.models import load_clip_model
from utils.get_embeddings import cache_clip_embeddings
from clip_sim.analysis import run_clip_faiss_matching, summarize_descriptor_alignment_stats
from clip_sim.plot_utils import plot_descriptor_marginal_distributions_with_stats
from config import CACHE_DIR, DEVICE, RESULTS_DIR, METADATA_DIR, DESCRIPTOR_DIR, FORMATED_DESCRIPTOR_DIR
from pathlib import Path
from text_utils import normalize
# === Main Execution ===
if __name__ == "__main__":
    # use arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run concept analysis pipeline.")
    parser.add_argument("--dataset", type=str, default="iteration_formated", help="Dataset name (cifar100, nabirds, cub)")
    parser.add_argument("--num_subset", type=int, default=1000000, help="Number of samples to subset from the dataset")
    parser.add_argument("--sim_threshold", type=float, default=0.8, help="Similarity threshold for matching")     
    parser.add_argument("--top_k", type=float, default=0.001, help="Top K ratio matches to consider")  

    args = parser.parse_args()

    dataset_name = args.dataset  
    num_subset = args.num_subset 

    sim_threshold = args.sim_threshold
    top_k = round(args.top_k*num_subset)
    json_dict = {
        "iteration": [
            DESCRIPTOR_DIR
        ],
        "iteration_formated": [
            FORMATED_DESCRIPTOR_DIR
        ]
    }
    
    if dataset_name == "iteration":
        # iterate over all dir in json_dict["iteration"] got all json files
        json_files = []
        for root, dirs, files in os.walk(json_dict["iteration"][0]):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))
        json_dict["iteration"] = json_files
    if dataset_name == "iteration_formated":
        json_files = []
        for root, dirs, files in os.walk(json_dict["iteration_formated"][0]):
            for file in files:
                if file.endswith(".json"):
                    assert os.path.exists(os.path.join(root, file)), f"File {os.path.join(root, file)} does not exist"
                    json_files.append(os.path.join(root, file))
        json_dict["iteration_formated"] = json_files

    # === Load Parquet File ===
    # === Load Parquet File for CLIP training data===
    # download from website https://huggingface.co/datasets/apf1/datafilteringnetworks_2b
    # https://github.com/mlfoundations/datacomp#downloading-commonpool
    # We only use the small scale (450 GB
    metadata_files = list(Path(METADATA_DIR).glob("*.parquet"))
    print(f"Found {len(metadata_files)} metadata files in {METADATA_DIR}")
    df = pd.concat([pd.read_parquet(Path(METADATA_DIR) / f) for f in metadata_files], ignore_index=True)
    df['text'] = df['text'].astype(str)
    print(f"Loaded {len(df)} rows from metadata files.") # 26

    num_subset_million = int(num_subset / 1e6)
    save_root = f"{RESULTS_DIR}/clip_sim/freq_{dataset_name}{num_subset_million}M_original"
    Path(save_root).mkdir(parents=True, exist_ok=True)

    json_files = json_dict[dataset_name]

    # Sample once with fixed random_state to create a deterministic order
    ordered_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Take top `num_subset` rows — always a prefix of the full permutation
    subset_df = ordered_df.iloc[:min(num_subset, len(ordered_df))]

    clip_texts = subset_df['text'].tolist()
    model, tokenizer = load_clip_model(DEVICE)
    caption_embs = cache_clip_embeddings(clip_texts, model, tokenizer, cache_dir=Path(CACHE_DIR) / f"clip_pretrain_captions", device=DEVICE, save_dir=f"caption{num_subset_million}M")

    for fname in json_files:
        try:
            start_time = time.time()
            fname = Path(fname)
            cls2concepts_all = defaultdict(list)

            # last two directories
            save_dir = Path(save_root) / fname.parent.parent.name / fname.parent.name / fname.stem
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            print(f"Processing {fname}...") 
            print(f"Saving to {save_dir}")

            all_concepts = []
            with open(fname, "r") as f:
                data = json.load(f)

            for concepts in data.values():
                if isinstance(concepts, list):
                    all_concepts.extend(concepts)
                elif isinstance(concepts, str):
                    all_concepts.append(concepts)

                
            all_concepts = [concept for concept in all_concepts if isinstance(concept, str)]
            print(f"Found {len(all_concepts)} unique concepts in {fname}")

            if len(all_concepts) == 0:
                print(f"No valid concepts found in {fname}. ")
                import pdb; pdb.set_trace()  

            all_concepts_norm = list(set(normalize(c) for c in all_concepts))


            descriptor_name = f"{fname.stem}_n{num_subset}_th{sim_threshold}_top{top_k}"

            concept_freq_df = run_clip_faiss_matching(
                subset_df,
                all_concepts_norm,
                save_dir=save_dir,
                sim_col_name=f"avg_sim_clip_l14_{descriptor_name}",
                sim_threshold=sim_threshold,
                top_k=top_k,
                caption_embs=caption_embs,
                model=model,
                tokenizer=tokenizer,    
                device=DEVICE,
            )
            

            summary = summarize_descriptor_alignment_stats(
                concept_freq_df,
                sim_col=f"avg_sim_clip_l14_{descriptor_name}",
                freq_col='freq_in_captions',
                descriptor_name=descriptor_name,
            )

            plot_descriptor_marginal_distributions_with_stats(
                df=concept_freq_df,
                sim_col=f"avg_sim_clip_l14_{descriptor_name}",
                freq_col='freq_in_captions',
                descriptor_name=descriptor_name,
                save_dir=save_dir,
                bins=20,
            )

            endtime = time.time()
            print(f"⏰ Time taken: {endtime - start_time:.2f} seconds")


            # Optional: use fuzzy or exact matching instead of CLIP+FAISS 
            # print("Running exact matching...")
            # concept_freq_df = run_concept_analysis(subset_df, all_concepts_norm, exact_match, f"avg_sim_exact_n{num_subset}", save_dir, n_jobs=16)
            # print("Running fuzzy matching...")  
            # concept_freq_df = run_concept_analysis(subset_df, all_concepts_norm, fuzzy_match_factory(fuzzy_match_factory), f"avg_sim_fuzzy_n{num_subset}_th{fuzzy_match_factory}", save_dir, n_jobs=16)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue    