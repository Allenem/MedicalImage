import os
import PIL.Image as Image
from torch.utils.data import Dataset


def make_dataset(root):
    imgs = []
    n = len(os.listdir(root))//2
    for i in range(n):
        img = os.path.join(root, '%03d.png' % (i+1))
        label = os.path.join(root, '%03d_label.png' % (i+1))
        imgs.append((img, label))
    return imgs


class LoadDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
