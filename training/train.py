import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageObjDataset
from model import InfoGANGeneratorWithMixedCodes, OptimizedInfoGANDiscriminator
from torch.utils.tensorboard import SummaryWriter

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 损失函数定义
def define_loss_functions():
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

    return discriminator_loss, generator_loss, info_loss

# 数据加载器
def get_data_loader(batch_size):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageObjDataset(root_dir='/path/to/your/dataset', transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练函数
def train(generator, discriminator, data_loader, optimG, optimD, epochs, device):
    writer = SummaryWriter('runs/infogan_experiment')
    disc_loss_fn, gen_loss_fn, info_loss_fn = define_loss_functions()
    
    for epoch in range(epochs):
        for i, (real_images, real_objs) in enumerate(data_loader):
            real_images, real_objs = real_images.to(device), real_objs.to(device)
            noise = torch.randn(real_images.size(0), 100, device=device)
            c_cat = torch.randn(real_images.size(0), 10, device=device)
            c_cont = torch.randn(real_images.size(0), 5, device=device)

            # Train Discriminator
            optimD.zero_grad()
            real_output = discriminator(real_objs)
            fake_objs = generator(noise, c_cat, c_cont)
            fake_output = discriminator(fake_objs.detach())
            d_loss = disc_loss_fn(real_output, fake_output)
            d_loss.backward()
            optimD.step()

            # Train Generator
            optimG.zero_grad()
            fake_output, c_cat_pred, c_cont_pred = discriminator(fake_objs)
            g_loss = gen_loss_fn(fake_output)
            info_l = info_loss_fn(c_cat_pred, c_cont_pred, c_cat, c_cont)
            total_g_loss = g_loss + info_l
            total_g_loss.backward()
            optimG.step()

            # Logging
            if i % 100 == 0:
                writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch * len(data_loader) + i)
                writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(data_loader) + i)
                writer.add_scalar('Loss/Info', info_l.item(), epoch * len(data_loader) + i)

        print(f"Epoch {epoch+1}, Discriminator Loss: {d_loss.item()}, Generator Loss: {total_g_loss.item()}")

    writer.close()

# 主执行函数
def main():
    batch_size = 32
    epochs = 100

    generator = InfoGANGeneratorWithMixedCodes(100, 10, 5).to(device)
    discriminator = OptimizedInfoGANDiscriminator(10, 5).to(device)
    data_loader = get_data_loader(batch_size)
    
    optimG = optim.Adam(generator.parameters(), lr=0.0002)
    optimD = optim.Adam(discriminator.parameters(), lr=0.0002)
    
    train(generator, discriminator, data_loader, optimG, optimD, epochs, device)

    # Save models
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

if __name__ == "__main__":
    main()
