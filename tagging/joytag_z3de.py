# tagging/joytag_z3de.py
# JoyTag + Z3DE Tagger 集成模块
# 支持普通标签 + NSFW 标签

import torch
from PIL import Image
from typing import List, Dict
from transformers import AutoProcessor, AutoModelForSequenceClassification
from utils.joytag_pred import predict as joytag_predict
from utils.models.deepdanbooru.deepdanbooru import DeepDanbooruModel

# -----------------------------
# 基础配置
# -----------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
JOYTAG_MODEL = "fancyfeast/joytag"
Z3DE_MODEL = "pearisli/deepdanbooru-pytorch"

# NSFW tag 集合（示例）
NSFW_TAG_SET = set([
    'nude', 'nipples', 'cleavage', 'underwear', 'lingerie', 'panties', 'pussy', 'ass', 'sex', 'erotic', 'fetish', 'bondage'
])

# -----------------------------
# Tagger 类
# -----------------------------
class JoyZ3DETagger:
    def __init__(self, joytag_model=JOYTAG_MODEL, z3de_model=Z3DE_MODEL, device=DEVICE):
        print("Initializing JoyZ3DETagger...")
        self.device = device

        # Z3DE
        self.z3_model = DeepDanbooruModel.from_pretrained("pearisli/deepdanbooru-pytorch").to(self.device)

    @torch.no_grad()
    def tag(self, image: Image.Image) -> Dict:
        """单张图像标注（向后兼容）"""
        results = self.tag_batch([image])
        return results[0]

    @torch.no_grad()
    def tag_batch(self, images: List[Image.Image]) -> List[Dict]:
        """批量图像标注"""
        batch_results = []
        
        # --------------------
        # JoyTag（批量处理）
        # --------------------
        joy_tags_batch = []
        for image in images:
            joy_tag_string, _ = joytag_predict(image)
            joy_tags = [tag.strip() for tag in joy_tag_string.split(',') if tag.strip()]
            joy_tags_batch.append(joy_tags)

        # --------------------
        # Z3DE（批量处理）
        # --------------------
        # 检查 z3_model.tag 是否支持批处理
        # 如果不支持，需要逐个处理
        z3_tags_batch = []
        for image in images:
            z3_tags = self.z3_model.tag(image)
            z3_tags_batch.append(z3_tags[0] if isinstance(z3_tags, (list, tuple)) else z3_tags)

        # --------------------
        # 合并结果
        # --------------------
        for joy_tags, z3_tags in zip(joy_tags_batch, z3_tags_batch):
            all_tags = set(joy_tags) | set(z3_tags)
            nsfw_tags = all_tags & NSFW_TAG_SET
            general_tags = all_tags - nsfw_tags

            batch_results.append({
                "tags_general": sorted(general_tags),
                "tags_nsfw": sorted(nsfw_tags),
                "all_tags": sorted(all_tags)
            })

        return batch_results

    def _extract_tags(self, model_outputs):
        # 这里假设模型输出 logits，每个标签阈值 0.5
        probs = torch.sigmoid(model_outputs.logits[0])
        labels = model_outputs.model.config.id2label
        selected = [labels[i] for i, p in enumerate(probs) if p > 0.5]
        return selected