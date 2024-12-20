import os
from shutil import copy2
from PIL import Image
from torch.utils.data import Dataset

# 数据前期处理：对数据集内图片命名进行修改处理
# 通用路径和更新函数
def clean_filename(filename):
    """
    特殊字符替换函数，用于清理文件名中的非法字符。
    :param filename: 原始文件名
    :return: 清理后的文件名
    """
    illegal_chars = ['*', '?', '<', '>', '|', ':', '"', '\\', '/', '&']
    for char in illegal_chars:
        filename = filename.replace(char, "_")  # 将非法字符替换为 "_"
    return filename

def update_dataset(img_dir, annotations_file, updated_img_dir, updated_annotations_file=None):
    """
    更新数据集文件，包括处理特殊字符并生成新的标签文件。
    :param img_dir: 原始图片路径
    :param annotations_file: 原始标签文件路径
    :param updated_img_dir: 更新后的图片保存路径
    :param updated_annotations_file: 更新后的标签文件保存路径
    """
    # 创建新的文件夹
    os.makedirs(updated_img_dir, exist_ok=True)

    # 处理图片文件
    for file_name in os.listdir(img_dir):
        old_path = os.path.join(img_dir, file_name)
        if os.path.isfile(old_path):
            # 清理文件名中的特殊字符
            new_file_name = clean_filename(file_name)
            new_path = os.path.join(updated_img_dir, new_file_name)
            # 将文件复制到新文件夹
            copy2(old_path, new_path)
            print(f"Copied and renamed: {old_path} -> {new_path}")

    # 更新标签文件（如果提供了标签路径）
    if annotations_file and updated_annotations_file:
        updated_lines = []
        with open(annotations_file, "r") as f:
            for line in f:
                img_name, age = line.strip().split("\t")
                # 更新文件名中的特殊字符
                updated_name = clean_filename(img_name)
                updated_lines.append(f"{updated_name}\t{age}")

        # 将更新后的标签文件写入新路径
        with open(updated_annotations_file, "w") as f:
            f.write("\n".join(updated_lines))

        print(f"Updated annotations saved to: {updated_annotations_file}")

# 数据加载类
class PetDataset(Dataset):
    """
    自定义数据集类，用于加载图片和对应标签。
    """
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = []
        with open(annotations_file, "r") as f:
            for line in f:
                img_name, age = line.strip().split("\t")
                self.img_labels.append((img_name, int(age)))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, age = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, age

# 更新训练集
train_img_dir = r"D:\0Knowledge\Python\deep learning\curriculum_design\dataset3\trainset\trainset"
train_annotations = r"D:\0Knowledge\Python\deep learning\curriculum_design\dataset3\annotations\annotations\train.txt"
updated_train_img_dir = r"D:\0Knowledge\Python\deep learning\curriculum_design\dataset5\updated_trainset"
updated_train_annotations = r"D:\0Knowledge\Python\deep learning\curriculum_design\dataset5\annotations\updated_train.txt"

update_dataset(train_img_dir, train_annotations, updated_train_img_dir, updated_train_annotations)

# 更新验证集
val_img_dir = r"D:\0Knowledge\Python\deep learning\curriculum_design\dataset3\valset\valset"
val_annotations = r"D:\0Knowledge\Python\deep learning\curriculum_design\dataset3\annotations\annotations\val.txt"
updated_val_img_dir = r"D:\0Knowledge\Python\deep learning\curriculum_design\dataset5\updated_valset"
updated_val_annotations = r"D:\0Knowledge\Python\deep learning\curriculum_design\dataset5\annotations\updated_val.txt"

update_dataset(val_img_dir, val_annotations, updated_val_img_dir, updated_val_annotations)

# 更新测试集
test_img_dir = r"D:\0Knowledge\Python\deep learning\curriculum_design\dataset3\updated_testset"
updated_test_img_dir = r"D:\0Knowledge\Python\deep learning\curriculum_design\dataset5\updated_testset"

# 测试集没有标签文件，因此只处理图片
update_dataset(test_img_dir, None, updated_test_img_dir)
