import torch
from pathlib import Path
from tqdm import tqdm
from dino_align.eval.cbd_eval import calculate_cbd_accuracy
from utils.io import save_json
from config import CACHE_DIR, RESULTS_DIR

def list_embedding_files(dataset: str, model_name: str) -> list:
    """
    Returns a list of Paths to all embedding files for the given dataset and model.
    Assumes structure: embeddings/<dataset>/<model_name>/<split>/embed.pt
    """
    root = Path(CACHE_DIR) / Path("image_embeddings") / dataset / model_name
    if not root.exists():
        return []
    return list(root.rglob("embed.pt"))


def evaluate_all_descriptors(
    descriptors_path,
    model_name,
    output_path=f"{RESULTS_DIR}/dino_align/accuracy.json",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    descriptors_path = Path(descriptors_path)
    output_path = Path(output_path)
    results = {}
    descriptor_files = list(descriptors_path.rglob("*.json"))
    print(f"Found {len(descriptor_files)} descriptor files.")
    for descriptor_file in tqdm(sorted(descriptor_files), desc="Evaluating descriptors"):
        dataset = descriptor_file.parent.name
        descriptor_name = descriptor_file.stem
        embedding_files = list_embedding_files(dataset, model_name)

        if not embedding_files:
            print(f"⚠️ No embedding files found for dataset '{dataset}', skipping.")
            continue

        results.setdefault(dataset, {})
        results[dataset].setdefault(descriptor_name, {})

        for embed_file in sorted(embedding_files):
            split = embed_file.parent.name  # e.g. "train", "test"
            try:
                acc = calculate_cbd_accuracy(
                    image_embed_path=embed_file,
                    descriptor_path=descriptor_file,
                    model_name=model_name,
                    device=device
                )
                results[dataset][descriptor_name][split] = acc
            except Exception as e:
                print(f"❌ Failed on {dataset}/{descriptor_name}/{split}: {e}")
                results[dataset][descriptor_name][split] = None

                
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, output_path)
    print(f"✅ Saved results to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors_path", type=str, required=True,
                        help="Top-level folder with per-dataset descriptor subfolders")
    parser.add_argument("--model_name", type=str, default="clip-vit-large-patch14",
                        help="Model name used in the embeddings folder structure (e.g., clip-vit-large-patch14)")
    parser.add_argument("--output_path", type=str, default=f"{RESULTS_DIR}/dino_align/no_class_names/accuracy.json")
    args = parser.parse_args()

    evaluate_all_descriptors(
        descriptors_path=args.descriptors_path,
        model_name=args.model_name,
        output_path=args.output_path
    )



