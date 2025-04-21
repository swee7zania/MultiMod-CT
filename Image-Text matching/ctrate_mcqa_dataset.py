import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from transformers import AutoTokenizer

def center_crop_depth(image, target_depth=128):
    _, D, H, W = image.shape
    if D < target_depth:
        # Padding if too small
        pad = (target_depth - D) // 2
        pad_left = torch.zeros((1, pad, H, W))
        pad_right = torch.zeros((1, target_depth - D - pad, H, W))
        image = torch.cat([pad_left, image, pad_right], dim=1)
    elif D > target_depth:
        # Crop center
        start = (D - target_depth) // 2
        image = image[:, start:start+target_depth, :, :]
    return image

import torch.nn.functional as F

def resize_hw(image, target_hw):
    """
    image: [1, D, H, W]
    returns: resized image [1, D, target_H, target_W]
    """
    _, D, H, W = image.shape
    image = image.permute(1, 0, 2, 3)  # [D, 1, H, W]
    image = F.interpolate(image, size=target_hw, mode='bilinear', align_corners=False)
    image = image.permute(1, 0, 2, 3)  # [1, D, H, W]
    return image


class CTReportMCQADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer_name="bert-base-uncased", max_length=512, image_transform=None):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.image_transform = image_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        image_path = entry["image_path"]
        metadata_prompt = entry["metadata_prompt"]
        options = entry["options"]
    
        # ✅ 图像加载
        img_nii = nib.load(image_path)
        image = img_nii.get_fdata()
        image = torch.tensor(image).unsqueeze(0).float()  # [1, D, H, W]
    
        # ✅ 修改尺寸（更小、提速）
        #image = center_crop_depth(image, target_depth=64)            # 中心裁剪深度
        #image = resize_hw(image, target_hw=(128, 128))               # resize 高宽
        image = center_crop_depth(image, target_depth=32)
        image = resize_hw(image, target_hw=(96, 96))  # 更小空间分辨率
        
        if self.image_transform:
            image = self.image_transform(image)
        
        # 文本处理
        input_ids = []
        attention_masks = []
        label_index = -1

        for i, opt in enumerate(options):
            #不要 prompt 试试
            #full_text = metadata_prompt + " " + opt["text"]
            full_text = opt["text"]
            encoded = self.tokenizer(
                full_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids.append(encoded["input_ids"].squeeze(0))
            attention_masks.append(encoded["attention_mask"].squeeze(0))
            if opt["label"] == 1:
                label_index = i

        input_ids = torch.stack(input_ids)             # [4, N]
        attention_masks = torch.stack(attention_masks) # [4, N]

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "label": torch.tensor(label_index)
        }

# -------------------------------------------
# ✅ 示例主函数：直接运行文件测试数据加载
# -------------------------------------------
if __name__ == "__main__":
    jsonl_path = "ctrate_mcqa_full_report_train.jsonl"  # 请确认路径存在

    print(f"📂 正在加载数据集: {jsonl_path}")
    dataset = CTReportMCQADataset(
        jsonl_path=jsonl_path,
        tokenizer_name="bert-base-uncased",
        max_length=256
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        print("✅ 批数据加载成功!")
        print("图像张量 image.shape:", batch["image"].shape)               # [B, 1, D, H, W]
        print("文本 input_ids.shape:", batch["input_ids"].shape)          # [B, 4, N]
        print("标签 label:", batch["label"])                              # [B]
        
        # 🔍 可视化每个选项的 token 原文
        print("\n📄 每个选项原始文本（去token化）：")
        tokenizer = dataset.tokenizer
        for i in range(4):
            decoded = tokenizer.decode(batch["input_ids"][0][i], skip_special_tokens=True)
            print(f"Option {i}: {decoded}")

        print("\n🎯 正确选项索引:", batch["label"].item())

        break  # 只查看一批样本
