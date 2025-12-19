import numpy as np
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download
from typing import List, Union

class WD14Tagger:
    def __init__(self, repo="SmilingWolf/wd-v1-4-vit-tagger", threshold=0.35):
        print("Initializing WD14Tagger...")
        self.threshold = threshold

        self.model_path = hf_hub_download(repo, "model.onnx")
        self.tags_path = hf_hub_download(repo, "selected_tags.csv")

        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        self.tags = []
        with open(self.tags_path, "r", encoding="utf-8") as f:
            for line in f:
                self.tags.append(line.strip().split(",")[1])

    def _preprocess(self, image: Image.Image):
        """预处理单张图像"""
        image = image.convert("RGB").resize((448, 448))

        arr = np.asarray(image).astype(np.float32)
        arr = arr[:, :, ::-1]          # RGB -> BGR
        arr = arr / 255.0
        arr = (arr - 0.5) / 0.5

        return arr  # (448, 448, 3)

    def _preprocess_batch(self, images: List[Image.Image]):
        """预处理多张图像"""
        batch = []
        for image in images:
            arr = self._preprocess(image)
            batch.append(arr)
        
        # 堆叠成 batch: (batch_size, 448, 448, 3)
        return np.stack(batch, axis=0)

    def tag(self, image: Image.Image):
        """单张图像标注（向后兼容）"""
        results = self.tag_batch([image])
        return results[0]

    def tag_batch(self, images: List[Image.Image]):
        """批量图像标注"""
        inp = self._preprocess_batch(images)

        outputs = self.session.run(
            None,
            {self.session.get_inputs()[0].name: inp}
        )[0]  # shape: (batch_size, num_tags)
        
        # 为每张图像提取标签
        results = []
        for batch_outputs in outputs:
            tags = [
                tag for tag, score in zip(self.tags, batch_outputs)
                if score > self.threshold
            ]
            results.append(tags)
        
        return results