import torch
import numpy as np
import trimesh
from sklearn.metrics import mean_squared_error
from model import InfoGANGeneratorWithMixedCodes  # 确保引入正确的生成器类
from dataset import ImageObjDataset
from torch.utils.data import DataLoader
from torchvision import transforms

def load_model(model_path, device, noise_dim, num_categories, cont_dim):
    """
    加载训练好的生成器模型。
    """
    model = InfoGANGeneratorWithMixedCodes(noise_dim, num_categories, cont_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_model(generator, dataset, device):
    """
    评估生成模型的性能。
    """
    generator.eval()
    mse_errors = []

    with torch.no_grad():
        for real_images, real_objs in dataset:
            real_images, real_objs = real_images.to(device), real_objs.to(device)
            noise = torch.randn(real_images.size(0), generator.noise_dim, device=device)
            c_cat = torch.randn(real_images.size(0), generator.num_categories, device=device)
            c_cont = torch.randn(real_images.size(0), generator.cont_dim, device=device)
            fake_objs = generator(noise, c_cat, c_cont)

            # 计算MSE
            mse_error = mean_squared_error(real_objs.cpu().numpy(), fake_objs.cpu().numpy())
            mse_errors.append(mse_error)

    average_mse = np.mean(mse_errors)
    print(f"Avg MSE Error: {average_mse}")
    return average_mse

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'path_to_generator_model.pth'
    noise_dim = 100
    num_categories = 10
    cont_dim = 5

    # 加载模型
    generator = load_model(model_path, device, noise_dim, num_categories, cont_dim)

    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageObjDataset(root_dir='/path/to/your/dataset', transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 执行评估
    evaluate_model(generator, data_loader, device)

if __name__ == "__main__":
    main()
