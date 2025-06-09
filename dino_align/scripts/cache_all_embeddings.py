from utils.get_embeddings import get_image_embeddings

if __name__ == "__main__":
    datasets = ["cub", "nabirds", "cifar100"]
    models = ["dino", "clip_vit_large_patch14"]
    splits = ["test"]

    for dataset in datasets:
        for model in models:
            for split in splits:
                get_image_embeddings(
                    dataset_name=dataset,
                    model_name=model,
                    split=split,
                    save=True,
                    overwrite=True,
                )