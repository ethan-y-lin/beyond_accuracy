import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle
import hashlib
from utils.datasets import get_dataset
from utils.models import get_model, get_processor
from config import CACHE_DIR, DEVICE, DATA_DIR

def get_image_embeddings(dataset_name, model_name, split="test", save=False, overwrite=False):
    save_path = Path(f"{CACHE_DIR}/image_embeddings/{dataset_name}/{model_name}/{split}/embed.pt")

    if save_path.exists() and save and not overwrite:
        print(f"Embeddings already exist at {save_path}. Skipping extraction.")
        data = torch.load(save_path)
        return data["image_embeddings"], data["class_ids"]

    dataset = get_dataset(dataset_name, DATA_DIR, train=(split == "train"))
    model = get_model(model_name, DEVICE)
    processor = get_processor(model_name)

    def collate_fn(batch):
        images, labels = zip(*batch)
        inputs = processor(images=images, return_tensors="pt")
        return inputs, torch.tensor(labels)

    loader = DataLoader(dataset, batch_size=64, num_workers=2, collate_fn=collate_fn)

    all_embeddings = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"Extracting {model_name} embeddings"):
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            if "dino" in model_name:
                outputs = model(**inputs)
                embeds = outputs.last_hidden_state[:, 0]  # CLS token
            elif "clip" in model_name:
                embeds = model.get_image_features(**inputs)  # [B, D]
            else:
                raise ValueError(f"Unsupported model type in model_name: {model_name}")
            all_embeddings.append(embeds.cpu())
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "image_embeddings": all_embeddings,
            "class_ids": all_labels,
        }, save_path)

    return all_embeddings, all_labels

def get_clip_text_embeddings(prompts, model_name, device=DEVICE, batch_size=2048):
    """
    Generate CLIP text embeddings using a model name.

    Args:
        prompts (List[str]): List of text prompts.
        model_name (str): Name of the model to load (e.g., "openai/clip-vit-base-patch32").
        device (str): "cuda" or "cpu".
        batch_size (int): Batch size for inference.
        save (bool): Whether to save the output.
        output_path (str or Path): Where to save the embeddings, if save is True.

    Returns:
        torch.Tensor: [num_prompts, embedding_dim] tensor of text embeddings.
    """
    # Load model and processor
    model = get_model(model_name, device)
    processor = get_processor(model_name)
    res = []

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating text embeddings with {model_name}"):
        batch_prompts = prompts[i : i + batch_size]
        inputs = processor(text=batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            res.append(text_features.cpu())

    all_embeddings = torch.cat(res, dim=0)
    return all_embeddings

def text_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()  # Or use sha256 for even more safety

def cache_clip_embeddings(
    texts, model, tokenizer,
    cache_dir=CACHE_DIR, device=DEVICE,
    batch_size=8192, dtype=np.float16,
    save_dir="clip_pretrain_captions",
    overwrite=False,
):
    cache_dir = Path(cache_dir) / save_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    emb_file = cache_dir / "embeddings.npy"
    idx_file = cache_dir / "text2index.pkl"

    # Load cache if it exists
    if emb_file.exists() and idx_file.exists() and not overwrite:
        all_embs = np.load(emb_file, mmap_mode="r").astype(dtype, copy=False)
        with open(idx_file, "rb") as f:
            text2idx = pickle.load(f)
        cached_keys = set(text2idx.keys())
        print(f"✅ Loaded {len(text2idx)} cached embeddings")
    else:
        all_embs = np.empty((0, 768), dtype=dtype)
        text2idx = {}
        cached_keys = set()

    # Track what we still need to compute
    to_compute = []
    to_compute_indices = []
    final_indices = [None] * len(texts)

    for i, text in enumerate(texts):
        key = text_hash(text)
        if key in cached_keys:
            final_indices[i] = text2idx[key]
        else:
            to_compute.append(text)
            to_compute_indices.append(i)

    # Compute missing embeddings
    new_embs = []
    if to_compute:
        for i in tqdm(range(0, len(to_compute), batch_size), desc="Encoding new texts"):
            batch_text = to_compute[i:i+batch_size]
            inputs = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feats = feats.cpu().to(torch.float16 if dtype == np.float16 else torch.float32)
                new_embs.append(feats)

        new_embs = torch.cat(new_embs).numpy().astype(dtype)
        offset = len(text2idx)
        for i, text in enumerate(to_compute):
            key = text_hash(text)
            text2idx[key] = offset + i

        # Update final indices and cache
        for i_orig, i_new in zip(to_compute_indices, range(len(to_compute))):
            final_indices[i_orig] = offset + i_new

        all_embs = np.concatenate([all_embs, new_embs], axis=0)
        np.save(emb_file, all_embs)
        with open(idx_file, "wb") as f:
            pickle.dump(text2idx, f)
        print(f"💾 Saved {len(text2idx)} embeddings to disk")

    assert None not in final_indices, "Some final indices are not resolved."
    return all_embs[final_indices]
