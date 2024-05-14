import torch
import numpy as np
import trimesh
from model import InfoGANGeneratorWithMixedCodes  # 确保引入正确的生成器类

def tensor_to_mesh(point_cloud_tensor):
    """
    将点云张量转换为trimesh可处理的Mesh对象。
    假设点云是Nx1维的,表示在一维空间中的N个点。
    """
    # 将张量转换为numpy数组
    points = point_cloud_tensor.detach().cpu().numpy().reshape(-1, 3)
    # 创建点云Mesh对象
    cloud = trimesh.points.PointCloud(points)
    # 通过球体重建来创建网格（示例方法，可能需要根据实际需求调整）
    mesh = trimesh.convex.convex_hull(cloud)
    return mesh

def smooth_and_fill_holes(mesh):
    """
    对Mesh对象应用平滑和填充孔洞操作。
    """
    # 检查模型是否有孔洞
    if not mesh.is_watertight:
        # 自动填充孔洞
        mesh.fill_holes()

    # 应用拉普拉斯平滑算法
    mesh = mesh.smoothed()

    return mesh

def generate_and_process(model_path, device, noise_dim, num_categories, cont_dim, num_points=26317):
    """
    生成点云,转换为Mesh,应用后处理,并显示结果。
    """
    # 加载模型
    generator = InfoGANGeneratorWithMixedCodes(noise_dim, num_categories, cont_dim)
    generator.load_state_dict(torch.load(model_path))
    generator.to(device)
    generator.eval()

    # 生成点云
    noise = torch.randn(1, noise_dim, device=device)
    c_cat = torch.randn(1, num_categories, device=device)  # 假设随机类别输入
    c_cont = torch.randn(1, cont_dim, device=device)  # 假设随机连续输入
    point_cloud = generator(noise, c_cat, c_cont)

    # 将点云数据转换为Mesh对象
    mesh = tensor_to_mesh(point_cloud.view(num_points, 1))

    # 应用平滑和填充孔洞
    mesh = smooth_and_fill_holes(mesh)

    # 显示处理后的Mesh
    mesh.show()

# 设置参数
model_path = 'path_to_generator_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_dim = 100
num_categories = 10
cont_dim = 5

# 运行处理流程
generate_and_process(model_path, device, noise_dim, num_categories, cont_dim)
