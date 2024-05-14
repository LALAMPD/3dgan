from PIL import Image,ImageOps
import trimesh
import pyrender
import numpy as np
import open3d as o3d
import pyvista as pv
from torchvision import transforms


def load_image(image_path):
    # 打开图片文件
    image = Image.open(image_path)
    # 将图片转换为RGB格式，确保图片是三通道的
    image = image.convert('RGB')
    # 这里可以添加其他的图像处理步骤
    return image
# 使用示例
image = load_image('C:\Users\s1810\3DINFOGAN_MASTER4\dataset\images')
image.show()



def load_3d_model(model_path):
    # 加载3D模型文件
    mesh = trimesh.load(model_path)
    return mesh

# 使用示例
mesh = load_3d_model('C:\Users\s1810\3DINFOGAN_MASTER4\dataset\objs')
print(mesh)

def render_model_from_viewpoint(mesh_path, camera_position, camera_target, camera_up, fov=60, img_width=640, img_height=480):
    # 加载3D模型
    mesh = trimesh.load(mesh_path)
    
    # 创建场景
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    # 设置相机
    camera = pyrender.PerspectiveCamera(yfov=np.radians(fov))
    cam_pose = look_at(camera_position, camera_target, camera_up)
    scene.add(camera, pose=cam_pose)
    
    # 设置渲染器
    renderer = pyrender.OffscreenRenderer(img_width, img_height)
    
    # 渲染场景
    color, depth = renderer.render(scene)
    renderer.delete()  # 清理资源
    return color

def look_at(eye, center, up):
    # 创建视角矩阵
    z = np.linalg.norm(eye - center)
    x = np.cross(up, z)
    y = np.cross(z, x)
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)
    mat = np.array([[x[0], x[1], x[2], -np.dot(x, eye)],
                    [y[0], y[1], y[2], -np.dot(y, eye)],
                    [z[0], z[1], z[2], -np.dot(z, eye)],
                    [0, 0, 0, 1]], dtype=np.float32)
    return mat

# 使用示例
camera_position = np.array([1, 1, 1])  # 相机位置
camera_target = np.array([0, 0, 0])    # 目标点位置
camera_up = np.array([0, 1, 0])        # 上向量
image = render_model_from_viewpoint('path_to_your_model.obj', camera_position, camera_target, camera_up)

def augment_image(image_path):
    # 加载图片
    image = Image.open(image_path)
    
    # 旋转图片
    rotated_image = image.rotate(45)  # 旋转45度
    
    # 翻转图片
    flipped_image = ImageOps.mirror(image)  # 水平翻转
    
    # 缩放图片
    size = (256, 256)  # 新的尺寸
    resized_image = image.resize(size, Image.ANTIALIAS)
    
    return rotated_image, flipped_image, resized_image

# 示例使用
rotated, flipped, resized = augment_image('path_to_your_image.jpg')
rotated.show()
flipped.show()
resized.show()

def augment_3d_model(model_path):
    # 加载3D模型
    mesh = trimesh.load(model_path)
    
    # 旋转模型
    rotation_matrix = trimesh.transformations.rotation_matrix(
        np.radians(45), [1, 0, 0])  # 绕x轴旋转45度
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(rotation_matrix)
    
    # 添加噪声
    noise = np.random.normal(0, 0.01, mesh.vertices.shape)
    noisy_mesh = mesh.copy()
    noisy_mesh.vertices += noise
    
    return rotated_mesh, noisy_mesh

# 示例使用
rotated_mesh, noisy_mesh = augment_3d_model('path_to_your_model.obj')
rotated_mesh.show()
noisy_mesh.show()

def image_to_tensor(image_path):
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    
    # 定义转换操作：将PIL图片转换为张量并归一化
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 将图片调整为256x256大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    
    # 应用转换
    tensor = transform(image)
    return tensor

# 示例使用
tensor = image_to_tensor('path_to_your_image.jpg')
print(tensor.shape)  # 输出张量的形状

#obj转换为点云tensor
def model_to_point_cloud(model_path):
    # 加载网格模型
    mesh = o3d.io.read_triangle_mesh(model_path)
    mesh.compute_vertex_normals()
    
    # 转换为点云
    pcd = mesh.sample_points_poisson_disk(number_of_points=2048)
    return pcd

# 示例使用
pcd = model_to_point_cloud('path_to_your_model.obj')
o3d.visualization.draw_geometries([pcd])  # 可视化点云

# #obj转换为体素网格
# def model_to_voxel_grid(model_path, density=128):
#     # 加载网格模型
#     mesh = pv.read(model_path)
    
#     # 创建体素网格
#     voxel_grid = mesh.voxelize(density=density)
#     return voxel_grid

# # 示例使用
# voxel_grid = model_to_voxel_grid('path_to_your_model.obj')
# voxel_grid.plot(show_edges=True)
