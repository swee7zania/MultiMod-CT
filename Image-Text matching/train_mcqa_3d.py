import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from ctrate_mcqa_dataset import CTReportMCQADataset
from multimodal_mcqa_model import MultimodalMCQAModel

from torchvision.models.video import r3d_18

# ✅ 使用 ResNet3D 作为 vision encoder
class ResNet3DEncoder(nn.Module):
    def __init__(self, out_dim=768, pretrained=False, debug=False):
        super().__init__()
        self.debug = debug
        self.backbone = r3d_18(pretrained=pretrained)
        self.backbone.fc = nn.Identity()  # [B, 512]
        self.projector = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):  # x: [B, 1, D, H, W]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)  # [B, 3, D, H, W]

        if self.debug:
            print(f"[Input to ResNet3D] {x.shape}")

        x = self.backbone(x)  # [B, 512]
        if self.debug:
            print(f"[Backbone out] {x.shape}")

        x = self.projector(x)  # [B, out_dim]
        if self.debug:
            print(f"[Final encoded] {x.shape}")
        return x

# ✅ 训练参数
BATCH_SIZE = 2
EPOCHS = 3
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 加载数据集
def get_dataloaders():
    train_dataset = CTReportMCQADataset("ctrate_mcqa_full_report_train.jsonl", max_length=MAX_LEN)
    val_dataset = CTReportMCQADataset("ctrate_mcqa_full_report_val.jsonl", max_length=MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

# ✅ 验证逻辑
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(image, input_ids, attention_mask)
            loss = nn.functional.cross_entropy(logits, labels)

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# ✅ 开始训练
def train():
    train_loader, val_loader = get_dataloaders()

    vision_encoder = ResNet3DEncoder(out_dim=768, pretrained=False, debug=False)

    model = MultimodalMCQAModel(vision_encoder, text_encoder_name="bert-base-uncased")
    model = model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    # ✅ 模型保存 & early stopping 设置
    best_val_acc = 0.0
    patience = 2
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}"):
            image = batch["image"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(image, input_ids, attention_mask)
            loss = nn.functional.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        # ✅ 验证
        val_loss, val_acc = evaluate(model, val_loader)

        print(f"✅ Epoch {epoch+1} Summary:")
        print(f"   🔹 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   🔹 Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # ✅ 模型保存（只保存最优）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_mcqa_model.pt")
            print(f"📦 模型已保存！（val acc 提升至 {val_acc:.4f}）")
        else:
            patience_counter += 1
            print(f"⏸️ 没有提升，early stopping计数: {patience_counter}/{patience}")

        # ✅ Early stopping 触发
        if patience_counter >= patience:
            print("🛑 触发 early stopping，训练结束。")
            break

if __name__ == "__main__":
    train()
