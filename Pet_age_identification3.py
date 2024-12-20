import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from torch.nn import MSELoss
from torch.optim import Adam
from PIL import Image
from tqdm import tqdm

# 数据集路径
# 指定训练集、验证集和测试集的图片路径以及标签文件路径
train_img_dir = r"/kaggle/input/pet-age-raw/dataset5/updated_trainset"  # 训练集图片路径
train_annotations = r"/kaggle/input/pet-age-raw/dataset5/annotations/updated_train.txt"  # 训练集标签文件路径
val_img_dir = r"/kaggle/input/pet-age-raw/dataset5/updated_valset"  # 验证集图片路径
val_annotations = r"/kaggle/input/pet-age-raw/dataset5/annotations/updated_val.txt"  # 验证集标签文件路径
test_img_dir = r"/kaggle/input/pet-age-raw/dataset5/updated_testset"  # 测试集图片路径
output_file = r"/kaggle/working/pred_result.txt"  # 推理结果保存路径

# 数据加载类
class PetDataset(Dataset):
    """
    自定义数据集类，用于加载图片和对应的标签
    """
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        初始化数据集
        :param annotations_file: 标签文件路径
        :param img_dir: 图片所在目录
        :param transform: 数据增强或预处理操作
        """
        self.img_labels = []
        # 读取标签文件，保存图片名和年龄
        with open(annotations_file, "r") as f:
            for line in f:
                img_name, age = line.strip().split("\t")
                self.img_labels.append((img_name, int(age)))  # 将年龄转为整数保存
        self.img_dir = img_dir  # 图片目录
        self.transform = transform  # 数据增强或预处理操作

    def __len__(self):
        """
        返回数据集的大小（图片数量）
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        根据索引加载图片及其对应的标签
        """
        img_name, age = self.img_labels[idx]  # 获取图片名和年龄
        img_path = os.path.join(self.img_dir, img_name)  # 构建图片完整路径
        image = Image.open(img_path).convert("RGB")  # 打开图片并转换为 RGB 格式
        if self.transform:
            image = self.transform(image)  # 进行数据增强或预处理
        return image, age  # 返回图片和标签

# 模型定义
class AgePredictor(torch.nn.Module):
    """
    定义基于 ResNet18 的年龄预测模型
    """
    def __init__(self):
        super(AgePredictor, self).__init__()
        self.model = resnet18(pretrained=True)  # 加载预训练的 ResNet18 模型
        # 替换 ResNet18 的最后一层，全连接层输出一个值（年龄）
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        """
        前向传播
        """
        return self.model(x)

# 训练函数
def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    """
    模型训练函数
    :param model: 训练的模型
    :param train_loader: 训练集数据加载器
    :param val_loader: 验证集数据加载器
    :param device: 运行设备（CPU 或 GPU）
    :param epochs: 训练轮数
    :param lr: 学习率
    """
    criterion = MSELoss()  # 定义损失函数为均方误差（MSE）
    optimizer = Adam(model.parameters(), lr=lr)  # 定义优化器为 Adam

    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        train_loss = 0
        # 显示训练和验证的进度条
        progress_bar = tqdm(total=len(train_loader) + len(val_loader), desc=f"Epoch {epoch+1}/{epochs}")

        # 训练阶段
        for images, ages in train_loader:
            images, ages = images.to(device), ages.to(device, dtype=torch.float32).unsqueeze(1)  # 将数据转移到设备
            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, ages)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            train_loss += loss.item()  # 累加训练损失
            progress_bar.update(1)  # 更新进度条

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}")  # 打印平均训练损失

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        val_loss = 0
        with torch.no_grad():  # 禁用梯度计算，加速验证过程
            for images, ages in val_loader:
                images, ages = images.to(device), ages.to(device, dtype=torch.float32).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, ages)
                val_loss += loss.item()
                progress_bar.update(1)  # 更新进度条

        progress_bar.close()
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss/len(val_loader):.4f}")  # 打印平均验证损失

# 推理函数
def infer_model(model, test_img_dir, output_file, device, transform):
    """
    推理函数，预测测试集图片的年龄并保存结果
    :param model: 训练好的模型
    :param test_img_dir: 测试集图片路径
    :param output_file: 推理结果保存路径
    :param device: 运行设备（CPU 或 GPU）
    :param transform: 数据增强操作
    """
    model.eval()  # 设置模型为评估模式
    results = []  # 用于保存推理结果
    progress_bar = tqdm(total=len(os.listdir(test_img_dir)), desc="Inference Progress")  # 显示推理进度条

    for img_name in os.listdir(test_img_dir):
        img_path = os.path.join(test_img_dir, img_name)
        image = Image.open(img_path).convert("RGB")  # 打开图片并转换为 RGB 格式
        image = transform(image).unsqueeze(0).to(device)  # 数据增强并转为张量
        with torch.no_grad():  # 禁用梯度计算，加速推理过程
            age_prediction = model(image).item()  # 预测年龄
        results.append(f"{img_name}\t{int(round(age_prediction))}")  # 保存图片名和预测年龄
        progress_bar.update(1)  # 更新进度条

    progress_bar.close()
    # 将预测结果保存到文件
    with open(output_file, "w") as f:
        f.write("\n".join(results))
    print(f"Predictions saved to {output_file}")  # 打印保存路径

if __name__ == "__main__":
    # 配置：直接运行训练和推理
    run_training = True  # 是否运行训练
    run_inference = True  # 是否运行推理

    # 指定运行设备（优先使用 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据增强操作
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图片大小
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 初始化模型
    model = AgePredictor().to(device)

    if run_training:
        # 加载训练集和验证集
        train_dataset = PetDataset(train_annotations, train_img_dir, transform)
        val_dataset = PetDataset(val_annotations, val_img_dir, transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练集数据加载器
        val_loader = DataLoader(val_dataset, batch_size=32)  # 验证集数据加载器

        # 开始训练
        train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4)

        # 保存模型
        torch.save(model.state_dict(), "age_predictor.pth")
        print("Model saved as age_predictor.pth")  # 打印模型保存路径

    if run_inference:
        # 加载训练好的模型
        model.load_state_dict(torch.load("age_predictor.pth", map_location=device))

        # 开始推理
        infer_model(model, test_img_dir, output_file, device, transform)
