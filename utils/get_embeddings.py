import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
from utils.datasets import get_dataset
from utils.models import get_model, get_processor
from pathlib import Path
# Constants
SHARED_DIR = "/share/j_sun/ethan"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.datasets import get_dataset
from utils.models import get_model, get_processor

# Constants
SHARED_DIR = "/share/j_sun/ethan"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_image_embeddings(dataset_name, model_name, split="test", save=False, overwrite=False):
    save_path = Path(f"embeddings/{dataset_name}/{model_name}/{split}/embed.pt")

    if save_path.exists() and save and not overwrite:
        print(f"Embeddings already exist at {save_path}. Skipping extraction.")
        data = torch.load(save_path)
        return data["image_embeddings"], data["class_ids"]

    dataset = get_dataset(dataset_name, SHARED_DIR, train=(split == "train"))
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