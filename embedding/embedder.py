import torch
from transformers import CLIPProcessor, CLIPModel


class ImageEmbedder:
    def __init__(self, model_name):
        print("Initializing ImageEmbedder...")
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        self.model = CLIPModel.from_pretrained(model_name).cuda().eval()


    @torch.no_grad()
    def embed(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to("cuda")
        emb = self.model.get_image_features(**inputs)
        return emb[0].cpu().numpy().astype("float32")