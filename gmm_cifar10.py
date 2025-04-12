import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# 导入PyTorch版本的GMM
from gmm_torch.gmm import GaussianMixture

# 检查是否有可用的GPU设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_cifar10():
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载CIFAR-10训练集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    
    return trainset

def preprocess_images(dataset):
    # 将图像数据转换为numpy数组并展平
    images = []
    for img, _ in dataset:
        # 将图像移动到GPU
        img = img.to(device)
        # 将图像从[3, 32, 32]转换为[3072]
        img_flat = img.reshape(-1)  # 直接使用PyTorch操作
        images.append(img_flat)
    
    # 直接创建PyTorch张量
    images = torch.stack(images)
    
    # 使用PyTorch进行标准化
    mean = images.mean(dim=0)
    std = images.std(dim=0)
    images_scaled = (images - mean) / (std + 1e-8)
    
    return images_scaled

class GMMCIFAR10Generator:
    def __init__(self, n_components=10):
        self.n_components = n_components
        # 使用PyTorch版本的GMM
        self.gmm = GaussianMixture(
            n_components=n_components,
            n_features=3072,  # CIFAR-10图像展平后的维度
            covariance_type="full",
        ).to(device)
    
    def fit(self, X):
        print("Training GMM model...")
        self.gmm.fit(X)
        print("Training completed!")
    
    def generate_images(self, n_samples=10):
        # 从GMM分布中采样
        X_generated, _ = self.gmm.sample(n_samples)
        
        # 将生成的数据重构为图像格式
        images = X_generated.reshape(-1, 3, 32, 32)
        
        # 生成图像时转回CPU以便显示
        images = images.cpu().numpy()
        
        return images

    def display_generated_images(self, images, n_rows=2, n_cols=5):
        plt.figure(figsize=(15, 6))
        for i in range(min(n_rows * n_cols, len(images))):
            plt.subplot(n_rows, n_cols, i + 1)
            # 转置图像从[C, H, W]到[H, W, C]
            img = images[i].transpose(1, 2, 0)
            # 将像素值裁剪到[0, 1]范围
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

def main():
    # 加载数据集
    print("Loading CIFAR-10 dataset...")
    dataset = load_cifar10()
    
    # 预处理图像
    print("Preprocessing images...")
    X_train = preprocess_images(dataset)
    
    # 创建和训练GMM模型
    generator = GMMCIFAR10Generator(n_components=10)
    generator.fit(X_train)
    
    # 生成新图像
    print("Generating new images...")
    generated_images = generator.generate_images(n_samples=10)
    
    # 显示生成的图像
    print("Displaying generated images...")
    generator.display_generated_images(generated_images)

if __name__ == '__main__':
    main()