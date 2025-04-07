import os
import sys
import nibabel as nib
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import IMAGE_DIR


def load_image_by_folder(folder_name, file_name):
    """
    加载指定子文件夹下的 NIfTI 文件。
    """
    file_path = os.path.join(IMAGE_DIR, folder_name, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件：{file_path}")
    
    img = nib.load(file_path)
    return img.get_fdata()


def show_middle_slice(volume, axis=2):
    slice_idx = volume.shape[axis] // 2
    if axis == 0:
        slice_img = volume[slice_idx, :, :]
    elif axis == 1:
        slice_img = volume[:, slice_idx, :]
    else:
        slice_img = volume[:, :, slice_idx]

    plt.imshow(slice_img.T, cmap="gray", origin="lower")
    plt.title(f"Axis {axis} - Slice {slice_idx}")
    plt.axis("off")
    plt.show()


# ✅ 示例调用
if __name__ == "__main__":
    folder = r"train_1\train_1_a" 
    file = "train_1_a_1.nii.gz"
    
    volume = load_image_by_folder(folder, file)
    print("图像维度:", volume.shape)
    show_middle_slice(volume)
