import os

# 根目录设为你的本地绝对路径
BASE_DIR = r"D:\aMaster\thesis_dataset\CT-RATE"

# 多标签预测结果路径
LABELS = {
    "train": os.path.join(BASE_DIR, "dataset", "multi_abnormality_labels", "train_predicted_labels.csv"),
    "validation": os.path.join(BASE_DIR, "dataset", "multi_abnormality_labels", "valid_predicted_labels.csv"),
}

# 报告路径
REPORTS = {
    "train": os.path.join(BASE_DIR, "dataset", "radiology_text_reports", "train_reports.csv"),
    "validation": os.path.join(BASE_DIR, "dataset", "radiology_text_reports", "validation_reports.csv"),
}

# 元数据路径
METADATA = {
    "train": os.path.join(BASE_DIR, "dataset", "metadata", "train_metadata.csv"),
    "validation": os.path.join(BASE_DIR, "dataset", "metadata", "validation_metadata.csv"),
}

# 图片路径
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "train")
