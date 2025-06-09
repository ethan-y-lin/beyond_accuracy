import json
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path
from utils.get_embeddings import get_clip_text_embeddings  # Reuse your unified embedding function
from utils.io import read_json
from config import RESULTS_DIR

def make_descriptor_sentence(class_name, descriptor):
    if descriptor.startswith(("a", "an")):
        return f"{class_name}, which is {descriptor}"
    elif descriptor.startswith(("has", "often", "typically", "may", "can")):
        return f"{class_name}, which {descriptor}"
    elif descriptor.startswith("used"):
        return f"{class_name}, which is {descriptor}"
    else:
        return f"{class_name}, which has {descriptor}"


def get_text_descriptors(cls2concepts, model_name, device, mod=False, include_class_name=False):
    """
    Converts a class-to-descriptors dictionary into a tensor of descriptor embeddings grouped by class.
    """
    cls2conceptidxs = defaultdict(list)
    concepts = []
    dedup_concepts = set()

    for cls, descs in cls2concepts.items():
        if include_class_name:
            descs = [cls] + descs
        for desc in descs:
            if mod:
                desc = make_descriptor_sentence(cls, desc)
            if desc in dedup_concepts:
                continue
            idx = len(concepts)
            concepts.append(desc)
            dedup_concepts.add(desc)
            cls2conceptidxs[cls].append(idx)

    all_embeds = get_clip_text_embeddings(concepts, model_name=model_name, device=device)

    cls2concepts_embed = {
        cls: all_embeds[idxs] for cls, idxs in cls2conceptidxs.items()
    }

    return cls2concepts_embed


def calculate_scores(image_embeddings, text_embeddings, model_scaling=None, batch_size=2048):
    """
    Computes cosine similarity scores between image and text embeddings.
    Returns a list of tensors (one per class).
    """
    device = image_embeddings.device
    image_embeddings = F.normalize(image_embeddings.float(), dim=-1)
    class_lengths = [t.shape[0] for t in text_embeddings]

    all_text = torch.cat([F.normalize(t.float(), dim=-1) for t in text_embeddings], dim=0).to(device)

    scores = []
    for i in range(0, image_embeddings.shape[0], batch_size):
        batch = image_embeddings[i:i+batch_size].to(device)
        sim = batch @ all_text.T
        if model_scaling:
            logit_scale, logit_bias = model_scaling
            sim = torch.sigmoid(sim * logit_scale.exp() + logit_bias)
        scores.append(sim.cpu())

    scores = torch.cat(scores, dim=0)

    # Split back into per-class
    split_scores = []
    start = 0
    for length in class_lengths:
        split_scores.append(scores[:, start:start+length])
        start += length

    return split_scores


def reduce_to_class_scores_by_mean(unreduced_scores):
    return torch.stack([s.mean(dim=1) for s in unreduced_scores], dim=1)  # (N, C)


def calculate_cbd_accuracy(image_embed_path, descriptor_path, model_name, device, store_activation=None):
    image_data = torch.load(image_embed_path)
    image_embeds = image_data["image_embeddings"].to(device)
    labels = image_data["class_ids"].numpy()

    descriptors = read_json(descriptor_path)

    cls2concepts_embed = get_text_descriptors(descriptors, model_name=model_name, device=device)
    unreduced_scores = calculate_scores(image_embeds, list(cls2concepts_embed.values()))
    class_scores = reduce_to_class_scores_by_mean(unreduced_scores)

    preds = class_scores.argmax(dim=-1).cpu().numpy()
    acc = (preds == labels).mean()
    print("CBD Accuracy:", acc)

    # === Save activations if requested ===
    # === Save targeted descriptor activations if requested ===
    if store_activation is not None:
        activation_output_path = f"{RESULTS_DIR}/dino_align/{descriptor_path.stem}_activations.json"
        class_names = list(cls2concepts_embed.keys())
        print(f"💾 Saving descriptor activations to {activation_output_path}")
        activation_json = {}

        for idx in range(image_embeds.shape[0]):
            gt_class = class_names[labels[idx]]
            pred_class = class_names[preds[idx]]

            entry = {}

            for cls in {gt_class, pred_class}:
                descs = descriptors[cls]
                sim_values = unreduced_scores[class_names.index(cls)][idx].tolist()
                entry[cls] = {desc: sim for desc, sim in zip(descs, sim_values)}

            activation_json[str(idx)] = entry

        Path(activation_output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(activation_output_path, "w") as f:
            json.dump(activation_json, f, indent=2)
    return acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--descriptor_path", type=str, required=True)
    parser.add_argument("--image_embed_path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calculate_cbd_accuracy(
        image_embed_path=args.image_embed_path,
        descriptor_path=args.descriptor_path,
        model_name=args.model,
        device=device
    )
