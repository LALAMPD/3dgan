import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# 参数设置
noise_dim = 100      # 噪声向量的维度
num_categories = 4  # 类别的数量
cont_dim = 5        # 连续编码的维度
img_channels = 3     # 图像通道数
img_size = 256       # 图像尺寸
point_cloud_dim = 26317*3  # 点云的输出维度
num_samples = 1024

class InfoGANGeneratorWithMixedCodes(nn.Module):
   def __init__(self, noise_dim, num_categories, cont_dim, img_channels,img_size, point_cloud_dim):
        super( InfoGANGeneratorWithMixedCodes, self).__init__()
        self.noise_dim = noise_dim
        self.img_feature_dim = 512  # 设定图像特征维度
        self.num_categories = num_categories
        self.cont_dim = cont_dim
         # 首先，将输入的噪声、类别标签和连续变量联合起来
        self.fc_noise_cat_cont = nn.Linear(noise_dim + num_categories + cont_dim, self.img_feature_dim)
        # 编码器部分
        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),  # 输出尺寸: (128, 128, 64)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 输出尺寸: (64, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 输出尺寸: (32, 32, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 输出尺寸: (16, 16, 512)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),  # 扁平化
            nn.Linear(512 * (img_size // 16) * (img_size // 16), self.img_feature_dim),  # 转为特征向量
            nn.ReLU()
        )
        
    # 合并图像特征和噪声编码的全连接层
        self.fc_combined = nn.Linear(self.img_feature_dim * 2, point_cloud_dim)

   def forward(self, img, noise, c_cat, c_cont):
        # 处理噪声和编码
        combined_noise_cat_cont = torch.cat([noise, c_cat, c_cont], dim=1)
        transformed_noise = self.fc_noise_cat_cont(combined_noise_cat_cont)
        
        # 处理图像
        img_features = self.conv_layers(img)

         # 合并处理后的图像特征和噪声编码
        combined_features = torch.cat([img_features, transformed_noise], dim=1)
        point_cloud = self.fc_combined(combined_features)
        return point_cloud.view(-1, 26317, 3)  # 重整输出为所需的点云形状
    
# 初始化生成器
generator = InfoGANGeneratorWithMixedCodes(noise_dim, num_categories, cont_dim, img_channels, img_size, point_cloud_dim)

# 生成噪声和编码
noise = torch.randn(1, noise_dim)
c_cat = F.one_hot(torch.randint(0, num_categories, (1,)), num_classes=num_categories).float()
c_cont = torch.randn(1, cont_dim)

# 生成随机图像数据作为输入
img = torch.randn(1, img_channels, img_size, img_size)

# 生成点云
point_cloud = generator(img, noise, c_cat, c_cont)
print(point_cloud.shape)  # 应该输出 torch.Size([1, 26317,3])

class FarthestPointSampler(torch.nn.Module):
    def forward(self, point_cloud):
        # 确保点云格式为 (batch_size, num_points, 3)
        if point_cloud.dim() != 3 or point_cloud.size(2) != 3:
            raise ValueError("Input point cloud must have shape [batch_size, num_points, 3]")
        return super().forward(point_cloud)

# 整合模型的正确初始化和使用
class IntegratedGAN(nn.Module):
    def forward(self, img, noise, c_cat, c_cont):
        # 生成点云数据，并确保输出形状为 (batch_size, num_points, 3)
        point_cloud = self.generator(img, noise, c_cat, c_cont).view(-1, 26317, 3)

        # 应用最远点采样优化点云
        sampled_point_cloud = self.sampler(point_cloud)

        # 确保输出形状正确
        if sampled_point_cloud.size(1) != self.sampler.num_samples or sampled_point_cloud.size(2) != 3:
            raise ValueError("Sampled point cloud must have shape [batch_size, num_samples, 3]")

        # 将优化后的点云数据传递给判别器
        real_fake, category, cont_vars = self.discriminator(sampled_point_cloud)
        return real_fake, category, cont_vars

class OptimizedInfoGANDiscriminator(nn.Module):
    def __init__(self, num_categories, cont_dim, num_samples=1024):
        super(OptimizedInfoGANDiscriminator, self).__init__()
        self.num_samples = num_samples  # 下采样后的点数
        self.num_categories = num_categories
        self.cont_dim = cont_dim
        
        # 卷积层用于处理下采样的点云数据
        self.conv_layers = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )

        # 判别器的最终判断层,输出真假评分
        self.fc_real_fake = nn.Linear(512 * num_samples, 1)
        self.sigmoid = nn.Sigmoid()

        # 类别预测
        self.fc_category = nn.Linear(512 * num_samples, num_categories)
        self.softmax = nn.Softmax(dim=1)
        # 连续变量预测
        self.fc_cont = nn.Linear(512 * num_samples, cont_dim)
    
    def forward(self, x):
        x = x.view(-1, 3, self.num_samples)  # 确保输入维度正确
        x = self.conv_layers(x)
        real_fake = self.sigmoid(self.fc_real_fake(x))
        category = self.softmax(self.fc_category(x))
        cont_vars = self.fc_cont(x)
        return real_fake, category, cont_vars



