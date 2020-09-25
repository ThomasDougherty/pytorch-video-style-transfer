import os
from pathlib import Path

import numpy as np
from PIL import Image 
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class Data(Dataset):

    def __init__(self, args):
        super().__init__()
        self.data_dir = args.dataset
        self.image_size = args.image_size
        self.imgs = os.listdir(args.dataset)
        
    def __len__(self):
        return len(self.imgs) 
        
    def __getitem__(self, index):    
        img_name = self.imgs[index]
        img_path = Path(self.data_dir, img_name)
        img = Image.open(img_path)

        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

        img = transform(img)
        sample = {"image": img}       
        return sample
        
def load_data(args):
    get_data_train = Data(args)
    
    train_data_loader = DataLoader(
        get_data_train, 
        batch_size=args.batch_size,  
        shuffle=True, 
        pin_memory=True
    )   
    
    return train_data_loader