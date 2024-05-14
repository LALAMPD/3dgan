import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class InfoGANGeneratorWithMixedCodes(nn.Module):
    def __init__(self, noise_dim, num_categories, cont_dim):
        super(  InfoGANGeneratorWithMixedCodes, self).__init__()
        self.noise_dim = noise_dim
        self.num_categories = num_categories
        self.cont_dim = cont_dim
         # 首先，将输入的噪声、类别标签和连续变量联合起来
        self.fc1 = nn.Linear(noise_dim + num_categories + cont_dim, 1024)
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 输出尺寸: (128, 128, 64)
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
            # nn.Flatten(),  # 扁平化
            # nn.Linear(16*16*512, 1000),  # 转为特征向量
            # nn.ReLU()
        )
        
        # 联合编码器和特征
        self.joint = nn.Sequential(
            nn.Linear(512*16*16 + num_categories + cont_dim, 10000),  # 加上编码信息c
            nn.ReLU(),
            nn.Linear(10000, 26317)  # 输出26317个点的点云
        )
    
    def forward(self,x, c_cat, c_cont):
        x = self.encoder(x)
        # 将类别编码和连续变量编码与来自编码器的特征向量合并
        c = torch.cat([c_cat, c_cont], dim=1)
        x_c = torch.cat([x, c], dim=1)
        return self.joint(x_c)  # 返回点云数据
    
# 例子：初始化生成器
generator = InfoGANGeneratorWithMixedCodes(noise_dim=100, num_categories=10, cont_dim=5)

# 生成随机噪声和编码输入
noise = torch.randn(1, 100)  # 噪声
c_cat = torch.zeros(1, 10)  # 类别标签，使用独热编码
c_cat[0, 3] = 1  # 假设第四类被选中
c_cont = torch.randn(1, 5)  # 连续变量

# 生成输出
output = generator(noise, c_cat, c_cont)
print(output.shape)  # 应该输出 torch.Size([1, 26317])


def farthest_point_sample(x, num_samples):
    """ 使用最远点采样(FPS)算法下采样点云 """
    np.random.seed(0)  # 为了可重复性，设置随机种子
    initial_idx = np.random.choice(len(x), 1)
    centroids = [x[initial_idx]]
    distances = np.linalg.norm(x - centroids[0], axis=1)
    for _ in range(1, num_samples):
        new_centroid = x[np.argmax(distances)]
        centroids.append(new_centroid)
        new_distances = np.linalg.norm(x - new_centroid, axis=1)
        distances = np.minimum(distances, new_distances)
    return np.stack(centroids)

class OptimizedInfoGANDiscriminator(nn.Module):
    def __init__(self, num_categories, cont_dim, num_samples=1024):
        super(OptimizedInfoGANDiscriminator, self).__init__()
        self.num_samples = num_samples  # 下采样后的点数
        total_dim = num_categories + cont_dim
        
        # 卷积层用于处理下采样的点云数据
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1),
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

        # 判别器的最终判断层
        self.fc_real_fake = nn.Linear(512 * num_samples, 1)
        self.sigmoid = nn.Sigmoid()

        # 额外编码信息c的回归层
        self.fc_cat = nn.Linear(512 * num_samples, num_categories)
        self.fc_cont = nn.Linear(512 * num_samples, cont_dim)
    
    def forward(self, x):
        # 下采样点云
        x = farthest_point_sample(x, self.num_samples)
        x = torch.tensor(x).unsqueeze(1)  # 增加维度以适应1D卷积
        x = self.conv_layers(x)
        validity = self.sigmoid(self.fc_real_fake(x))
        c_cat_pred = self.fc_cat(x)
        c_cont_pred = self.fc_cont(x)
        return validity, c_cat_pred, c_cont_pred

# 模型实例化和示例张量
num_categories = 10  # 类别标签的数量
cont_dim = 5  # 连续变量的维度
discriminator = OptimizedInfoGANDiscriminator(num_categories, cont_dim)

# 假设input_tensor是一批点云数据
input_tensor = np.random.rand(10, 26317, 3)  # 10个点云样本，每个包含26317个3维点
validity, c_cat_pred, c_cont_pred = discriminator(input_tensor)
print(validity.shape)  # 应输出torch.Size([10, 1])
print(c_cat_pred.shape)  # 应输出torch.Size([10, num_categories])
print(c_cont_pred.shape)  # 应输出torch.Size([10, cont_dim])


