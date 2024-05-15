import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class ImageObjDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 数据集的目录路径,其中包含images和objs子目录。
            transform (callable, optional): 可选的变换操作，应用于加载的图片。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        self.objs_dir = os.path.join(root_dir, 'objs')
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 图片文件处理
        img_name = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        
        # 对应的OBJ文件处理
        obj_name = os.path.join(self.objs_dir, self.image_files[idx].replace('.jpg', '.npy'))  # 假设您已将OBJ转换为Numpy数组保存
        obj = np.load(obj_name)

        # 转换为Tensor
        obj = torch.from_numpy(obj).float().view(-1, 3)  # 确保形状为 [26317, 3]

        return image, obj

# 使用示例
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageObjDataset(root_dir='C:\\Users\\s1810\\3DINFOGAN_MASTER4\\dataset', transform=transform)
