import torch
from pathlib import Path
from tqdm import tqdm
from eval.alignment import compute_alignment
from utils.io import save_json, read_json
from utils.get_embeddings import get_clip_text_embeddings

K_LOOKUP = {"cub": 30, "nabirds": 50, "cifar100": 100}


def list_embedding_files(dataset: str, model_name: str) -> list:
    root = Path("embeddings") / dataset / model_name
    if not root.exists():
        return []
    return list(root.rglob("embed.pt"))


def evaluate_all_alignment_scores(
    descriptors_path,
    model_name,
    output_path="results/alignment_results.json",
    alignment_mode="clip",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    descriptors_path = Path(descriptors_path)
    output_path = Path(output_path)
    results = {}

    descriptor_files = list(descriptors_path.rglob("*.json"))
    print(f"Found {len(descriptor_files)} descriptor files.")

    for descriptor_file in tqdm(sorted(descriptor_files), desc="Evaluating descriptor alignment"):
        dataset = descriptor_file.parent.name
        descriptor_name = descriptor_file.stem
        K = K_LOOKUP.get(dataset, 30)
        print(f"Dataset = {dataset}, Descriptors = {descriptor_name}, K = {K}")
        # Get all embeddings for this dataset/model_name
        embedding_files = list_embedding_files(dataset, model_name)
        if not embedding_files:
            print(f"⚠️ No embedding files found for dataset '{dataset}', skipping.")
            continue
        print("Embedding files: ", embedding_files)
        # Load and encode all descriptors
        descriptor_data = read_json(descriptor_file)
        all_descs = [desc for descs in descriptor_data.values() for desc in descs]
        print("Descriptor Length:", len(all_descs))
        if len(all_descs) < 300:
            print(all_descs)
        desc_embeds = get_clip_text_embeddings(all_descs, model_name=model_name, device=device)

        for embed_file in sorted(embedding_files):
            split = embed_file.parent.name  # from embeddings/<dataset>/<model>/<split>/embed.pt

            results.setdefault(dataset, {})
            results[dataset].setdefault(descriptor_name, {})

            try:
                clip_data = torch.load(embed_file)
                image_embeds = clip_data["image_embeddings"].to(device)

                # Determine target embedding structure
                if alignment_mode == "clip":
                    target_embeds = image_embeds
                elif alignment_mode == "gt":
                    labels = clip_data["class_ids"]
                    num_classes = labels.max().item() + 1
                    target_embeds = torch.nn.functional.one_hot(labels, num_classes=num_classes).float().to(device)
                elif alignment_mode == "dino":
                    dino_path = Path("embeddings") / dataset / "dinov2-base" / split / "embed.pt"
                    if not dino_path.exists():
                        raise FileNotFoundError(f"DINO embeddings not found at: {dino_path}")
                    dino_data = torch.load(dino_path)
                    target_embeds = dino_data["image_embeddings"].to(device)
                else:
                    raise ValueError(f"Unknown alignment_mode: {alignment_mode}")

                score = compute_alignment(
                    image_embeds=image_embeds,
                    desc_embeds=desc_embeds.to(device),
                    target_embeds=target_embeds,
                    k=K,
                    use_batch=False,
                    verbose=True
                )

                results[dataset][descriptor_name][split] = score
            except Exception as e:
                print(f"❌ Failed on {dataset}/{descriptor_name}/{split}: {e}")
                results[dataset][descriptor_name][split] = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, output_path)
    print(f"✅ Saved alignment results to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors_path", type=str, required=True,
                        help="Top-level folder with per-dataset descriptor subfolders")
    parser.add_argument("--model_name", type=str, default="clip-vit-large-patch14",
                        help="Model name used in the embeddings folder structure (e.g., clip-vit-large-patch14)")
    parser.add_argument("--alignment_mode", type=str, choices=["clip", "dino", "gt"], default="dino")
    parser.add_argument("--output_path", type=str, default="results/dino_alignment.json")
    args = parser.parse_args()

    evaluate_all_alignment_scores(
        descriptors_path=args.descriptors_path,
        model_name=args.model_name,
        output_path=args.output_path,
        alignment_mode=args.alignment_mode
    )
