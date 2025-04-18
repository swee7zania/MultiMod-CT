import os
import sys
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import IMAGE_DIR, REPORTS, LABELS


class CTMultimodalDataset(Dataset):
    def __init__(self, split="train", tokenizer=None, max_length=512, transform=None):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

        # 读取 CSV 数据
        self.reports_df = pd.read_csv(REPORTS[split])
        self.labels_df = pd.read_csv(LABELS[split])

        # 构建报告字段
        self.reports_df["report"] = (
            self.reports_df["Findings_EN"].fillna("") + " " +
            self.reports_df["Impressions_EN"].fillna("")
        )

        self.samples = self._merge_metadata()

    def _merge_metadata(self):
        report_dict = dict(zip(self.reports_df["VolumeName"], self.reports_df["report"]))
        label_dict = self.labels_df.set_index("VolumeName").to_dict(orient="index")

        common_ids = list(set(report_dict.keys()) & set(label_dict.keys()))
        print(f"✅ 匹配样本数: {len(common_ids)} / 报告总数: {len(report_dict)} / 标签总数: {len(label_dict)}")

        samples = []
        for vid in common_ids:
            samples.append({
                "id": vid,  # e.g. train_1_a_1.nii.gz
                "report": report_dict[vid],
                "labels": np.array(list(label_dict[vid].values()), dtype=np.float32)
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample["id"]  # e.g. train_1_a_1.nii.gz

        # 文件名去后缀取路径结构
        id_core = sample_id.replace(".nii.gz", "")  # train_1_a_1 → ['train', '1', 'a', '1']
        parts = id_core.split("_")  # ['train', '1', 'a', '1']
        level1 = f"{parts[0]}_{parts[1]}"         # train_1
        level2 = f"{parts[0]}_{parts[1]}_{parts[2]}"  # train_1_a

        # 构建完整路径
        image_path = os.path.join(IMAGE_DIR, level1, level2, sample_id)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ 图像文件不存在: {image_path}")

        image_data = nib.load(image_path).get_fdata().astype(np.float32)
        image_tensor = torch.from_numpy(image_data).unsqueeze(0)  # [1, D, H, W]

        if self.transform:
            image_tensor = self.transform(image_tensor)

        # 文本编码
        if self.tokenizer:
            text_input = self.tokenizer(
                sample["report"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            text_input = {k: v.squeeze(0) for k, v in text_input.items()}
        else:
            text_input = sample["report"]

        return {
            "id": sample_id,
            "image": image_tensor,
            "text": text_input,
            "label": torch.tensor(sample["labels"])
        }


# ✅ 测试入口
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = CTMultimodalDataset(split="train", tokenizer=tokenizer)

    print(f"📦 样本数: {len(dataset)}")
    
    '''
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    batch = next(iter(loader))
    print("✅ ID:", batch["id"])
    print("🖼️ 图像 shape:", batch["image"].shape)
    print("📝 文本 input_ids shape:", batch["text"]["input_ids"].shape)
    print("🏷️ 标签 shape:", batch["label"].shape)
    # 可视化第一张图像的中间切片
    volume = batch["image"][0, 0].numpy()
    z = volume.shape[2] // 2
    plt.imshow(volume[:, :, z], cmap="gray")
    plt.title("中间切片")
    plt.axis("off")
    plt.show()
    '''
    
    # 查找目标样本索引
    target_id = "train_1_a_1.nii.gz"
    idx = next((i for i, s in enumerate(dataset.samples) if s["id"] == target_id), None)

    if idx is None:
        print(f"❌ 未找到样本 {target_id}")
    else:
        sample = dataset[idx]
        print("✅ 样本 ID:", sample["id"])
        print("📝 报告前200字:\n", sample["text"]["input_ids"][:10], "...")  # or sample["report"]

        # 可视化图像中间层
        volume = sample["image"][0].numpy()
        z = volume.shape[2] // 2
        plt.imshow(volume[:, :, z], cmap="gray")
        plt.title(f"{target_id} - 中间切片")
        plt.axis("off")
        plt.show()