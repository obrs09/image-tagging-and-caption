import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from typing import Union, List

class JoyCaptioner:
    def __init__(self, model_name: str = "fancyfeast/llama-joycaption-beta-one-hf-llava", device: str = "cuda:0"):
        """
        初始化 JoyCaption 模型和处理器
        :param model_name: 模型托管地址
        :param device: 运行设备
        """
        self.device = device
        print(f"Loading model {model_name} to {device}...")
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        
        # 设置 pad_token（如果没有的话）
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        
        # 修复插值模式问题
        if hasattr(self.processor, 'image_processor'):
            if hasattr(self.processor.image_processor, 'resample'):
                self.processor.image_processor.resample = Image.BICUBIC
            if hasattr(self.processor.image_processor, 'interpolation'):
                self.processor.image_processor.interpolation = 'bicubic'
        
        # 加载模型
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map=device
        )
        self.model.eval()
        print("Model loaded successfully.")

    def predict(self, image_input: Union[str, Image.Image], prompt: str = "Write a long descriptive caption for this image in a formal tone.") -> str:
        """
        为单张图像生成描述（保持向后兼容）
        :param image_input: 可以是图像路径(str)或 PIL.Image 对象
        :param prompt: 提示词
        :return: 生成的描述文本
        """
        result = self.predict_batch([image_input], prompt)
        return result[0]

    def predict_batch(self, image_inputs: List[Union[str, Image.Image]], prompt: str = "Write a long descriptive caption for this image in a formal tone.") -> List[str]:
        """
        为多张图像批量生成描述
        :param image_inputs: 图像路径列表或 PIL.Image 对象列表
        :param prompt: 提示词（所有图像使用相同的 prompt）
        :return: 生成的描述文本列表
        """
        # 加载所有图像
        images = []
        for image_input in image_inputs:
            if isinstance(image_input, str):
                image = Image.open(image_input)
            else:
                image = image_input
            
            # 确保图像是 RGB 模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            images.append(image)
        
        # 为每张图像构建对话结构
        convo_strings = []
        for _ in images:
            convo = [
                {"role": "system", "content": "You are a helpful image captioner."},
                {"role": "user", "content": prompt},
            ]
            convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            convo_strings.append(convo_string)
        
        # 准备批量输入数据
        inputs = self.processor(
            text=convo_strings, 
            images=images, 
            return_tensors="pt", 
            padding=True,
            padding_side='left'  # 对于生成任务，左填充通常更好
        ).to(self.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        with torch.no_grad():
            # 批量生成 Token
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                suppress_tokens=None,
                use_cache=True,
                temperature=0.6,
                top_k=None,
                top_p=0.9,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

            # 解码所有生成的文本
            captions = []
            for i, generated_ids in enumerate(generate_ids):
                # 获取原始输入长度（考虑 padding）
                # 找到第一个非 pad token 的位置
                input_ids = inputs['input_ids'][i]
                if self.processor.tokenizer.padding_side == 'left':
                    # 左填充：找到第一个非 pad token
                    non_pad_mask = input_ids != self.processor.tokenizer.pad_token_id
                    first_non_pad = non_pad_mask.nonzero()[0].item() if non_pad_mask.any() else 0
                    input_length = len(input_ids) - first_non_pad
                else:
                    # 右填充
                    input_length = (input_ids != self.processor.tokenizer.pad_token_id).sum().item()
                
                # 裁剪掉 Prompt 部分，仅保留生成的回复
                generated_ids = generated_ids[len(input_ids):]
                
                # 解码
                caption = self.processor.tokenizer.decode(
                    generated_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
                captions.append(caption.strip())
            
            return captions

# --- 使用示例 ---
if __name__ == "__main__":
    # 实例化类
    predictor = JoyCaptioner()

    # 单张图像预测（向后兼容）
    img_path = "image.jpg"
    result = predictor.predict(img_path)
    print("\n--- Single Image Caption ---")
    print(result)
    
    # 批量图像预测
    img_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    results = predictor.predict_batch(img_paths)
    print("\n--- Batch Image Captions ---")
    for i, caption in enumerate(results):
        print(f"Image {i+1}: {caption}")