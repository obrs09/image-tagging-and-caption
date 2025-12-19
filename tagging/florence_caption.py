import torch
from transformers import AutoModelForCausalLM, AutoProcessor

class FlorenceCaptioner:
    def __init__(self, model_name, device="cuda"):
        print("Initializing FlorenceCaptioner...")
        self.device = device

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(device).eval()

    def caption(self, image):
        prompt = "<image>\nDescribe this image in detail."

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        # 🔥 关键修复：image -> FP16，text 保持 int64
        inputs["pixel_values"] = inputs["pixel_values"].to(
            device=self.device,
            dtype=torch.float16
        )
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return text.replace(prompt, "").strip()
