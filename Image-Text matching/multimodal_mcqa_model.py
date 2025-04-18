import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class MultimodalMCQAModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder_name="bert-base-uncased", hidden_dim=768):
        super().__init__()
        self.vision_encoder = vision_encoder  # 你可以传入你自己定义的图像编码器
        self.text_encoder = BertModel.from_pretrained(text_encoder_name)

        # 融合后计算相似度得分的 head
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, image, input_ids, attention_mask):
        B, O, N = input_ids.shape  # B=batch size, O=options, N=seq_len

        # 图像编码（得到一个图像 embedding）
        img_feat = self.vision_encoder(image)  # 输出 shape: [B, D]
        img_feat = img_feat.unsqueeze(1).expand(-1, O, -1)  # → [B, O, D]

        # 文本编码（每个选项）
        input_ids = input_ids.view(B * O, N)
        attention_mask = attention_mask.view(B * O, N)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_outputs.pooler_output  # [B*O, D]
        text_feat = text_feat.view(B, O, -1)     # → [B, O, D]

        # 计算匹配分数（可替换为余弦相似度等）
        fusion = img_feat * text_feat  # 简单乘法融合
        logits = self.score_head(fusion).squeeze(-1)  # [B, O]

        return logits

if __name__ == "__main__":
    from ctrate_mcqa_dataset import CTReportMCQADataset
    from torch.utils.data import DataLoader

    # Dummy Vision Encoder（可替换成 ViT 或自定义CNN）
    class DummyVisionEncoder(nn.Module):
        def __init__(self, output_dim=768):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 256 * 256, output_dim)
            )

        def forward(self, x):
            x = x.squeeze(1)  # Remove channel dim [B, 1, D, H, W] → [B, D, H, W]
            return self.fc(x)

    # 加载数据
    dataset = CTReportMCQADataset(
        jsonl_path="ctrate_mcqa_full_report.jsonl",
        tokenizer_name="bert-base-uncased",
        max_length=256
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 模型
    model = MultimodalMCQAModel(vision_encoder=DummyVisionEncoder())
    model.eval()

    # 执行 forward
    for batch in dataloader:
        logits = model(batch["image"], batch["input_ids"], batch["attention_mask"])  # [B, 4]
        print("logits shape:", logits.shape)  # 应为 [B, 4]
        print("预测分数:", logits)
        print("真实标签:", batch["label"])    # [B]
        loss = F.cross_entropy(logits, batch["label"])
        print("loss:", loss.item())
        break
