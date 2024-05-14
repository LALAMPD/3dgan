import torch
from torch import nn,ptim 
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageObjDataset  
from model import InfoGANGeneratorWithMixedCodes, OptimizedInfoGANDiscriminator
from torch.utils.tensorboard import SummaryWriter
from postprocessing import process_generated_data


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TensorBoard日志记录器
writer = SummaryWriter('runs/infogan_experiment')

# 损失函数定义
def discriminator_loss(real_output, fake_output):
    real_loss = torch.nn.functional.binary_cross_entropy(real_output, torch.ones_like(real_output))
    fake_loss = torch.nn.functional.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
    return (real_loss + fake_loss) / 2

def generator_loss(fake_output):
    return torch.nn.functional.binary_cross_entropy(fake_output, torch.ones_like(fake_output))

def info_loss(categorical_pred, continuous_pred, categorical_true, continuous_true):
    categorical_loss = torch.nn.functional.cross_entropy(categorical_pred, categorical_true)
    continuous_loss = torch.nn.functional.mse_loss(continuous_pred, continuous_true)
    return categorical_loss + continuous_loss

# 参数设置
noise_dim = 100
num_categories = 10
cont_dim = 5
learning_rate = 0.0002
batch_size = 32
epochs = 100

# 模型初始化
generator = InfoGANGeneratorWithMixedCodes(noise_dim, num_categories, cont_dim)
discriminator = OptimizedInfoGANDiscriminator(num_categories, cont_dim)

# 优化器
optimG = optim.Adam(generator.parameters(), lr=learning_rate)
optimD = optim.Adam(discriminator.parameters(), lr=learning_rate)

# 数据加载与转换操作定义
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = ImageObjDataset(root_dir='C:\Users\s1810\3DINFOGAN_MASTER4\dataset', transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 初始化数据集
dataset = ImageObjDataset(root_dir='C:\Users\s1810\3DINFOGAN_MASTER4\dataset', transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

 # 训练循环
for epoch in range(epochs):
    for i, (real_images, real_objs) in enumerate(data_loader):
        real_images, real_objs = real_images.to(device), real_objs.to(device)
        noise = torch.randn(real_images.size(0), noise_dim).to(device)
        c_cat = torch.randn(real_images.size(0), num_categories).to(device)  # Assuming a random input for categorical
        c_cont = torch.randn(real_images.size(0), cont_dim).to(device)  # Assuming a random input for continuous

         # 判别器更新
        optimD.zero_grad()
        real_output = discriminator(real_objs)
        fake_objs = generator(noise, c_cat, c_cont)
        fake_output = discriminator(fake_objs.detach())
        d_loss = discriminator_loss(real_output, fake_output)
        d_loss.backward()
        optimD.step()

        #  # 生成器更新
        # optimG.zero_grad()
        # fake_output = discriminator(fake_objs)
        # g_loss = generator_loss(fake_output)
        # info_l = info_loss(c_cat, c_cont, c_cat, c_cont)  # 假设简化为 c_cat 和 c_cont 作为真实和预测
        # total_g_loss = g_loss + info_l
        # total_g_loss.backward()
        # optimG.step()

        # 生成器和编码器更新
        optimG.zero_grad()
        # 需要再次通过判别器获取c_cat_pred和c_cont_pred
        fake_output, c_cat_pred, c_cont_pred = discriminator(fake_objs)
        g_loss = generator_loss(fake_output)
        info_l = info_loss(c_cat_pred, c_cont_pred, c_cat, c_cont)
        total_g_loss = g_loss + info_l
        total_g_loss.backward()
        optimG.step()

        # 日志记录
        if i % 100 == 0:  # 每100步记录一次
            writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch * len(data_loader) + i)
            writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(data_loader) + i)
            writer.add_scalar('Loss/Info', info_l.item(), epoch * len(data_loader) + i)

    print(f"Epoch {epoch+1}, Discriminator Loss: {d_loss.item()}, Generator Loss: {total_g_loss.item()}")

processed_mesh = process_generated_data(generator, device, noise_dim, num_categories, cont_dim)
processed_mesh.show()

# 保存模型
torch.save(generator.state_dict(), 'C:\Users\s1810\3DINFOGAN_MASTER4\output\generator')
torch.save(discriminator.state_dict(), 'C:\Users\s1810\3DINFOGAN_MASTER4\output\discrminator')

# 关闭TensorBoard写入器
writer.close()

