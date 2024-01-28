from torch.utils.data import Dataset
from PIL import Image
import os

class HymenopteraDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        path = os.path.join(self.root_dir, self.label_dir)
        self.img_name_list = os.listdir(path)

    def __getitem__(self, index):
        img_name = self.img_name_list[index]
        img_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_name_list)

if __name__ == '__main__':
    root_dir = "hymenoptera_data/train"
    ants_label_dir = "ants"
    bees_label_dir = "bees"
    ants_dataset = HymenopteraDataset(root_dir, ants_label_dir)
    bees_dataset = HymenopteraDataset(root_dir, bees_label_dir)
    train_dataset = ants_dataset + bees_dataset
    print(train_dataset[123])
    print(train_dataset[124])