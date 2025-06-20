from transformers import CLIPProcessor, CLIPModel
from transformers import AutoModel, AutoProcessor
from config import DEVICE, CLIP_MODEL
def get_model(model_name, device):
    if "clip" in model_name:
        return CLIPModel.from_pretrained("openai/" + model_name).to(device)
    elif "dino" in model_name:
        return AutoModel.from_pretrained("facebook/" + model_name).to(device)
    else:
        raise ValueError(f"Model {model_name} not found")
    
def get_processor(model_name):
    if "clip" in model_name:
        return CLIPProcessor.from_pretrained("openai/" + model_name)
    elif "dino" in model_name:
        return AutoProcessor.from_pretrained("facebook/" + model_name, use_fast=False)
    else:
        raise ValueError(f"Processor {model_name} not found")


def load_clip_model(device=DEVICE):
    model = get_model(CLIP_MODEL, device)
    tokenizer = get_processor(CLIP_MODEL)
    return model, tokenizer