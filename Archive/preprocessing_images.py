import torch
import torchvision
import random 
import PIL
import numpy as np

x = np.array([1,2,3,4,5])
print(f"shape:{x.shape}")

class combine(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.tf = torchvision.transforms.ToTensor()
        self.ds = torchvision.datasets.MNIST(root='.',download=True)
        self.ln = len(self.ds)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):

        idx = random.sample(range(self.ln), 4)
        store = []
        label = []
        for i in idx:
            x, y = self.ds[i]
            store.append(x)
            label.append(y)

        img = PIL.Image.new('L', (28*2, 28*2))
        img.paste(store[0], (0, 0))
        img.paste(store[1], (28, 0))
        img.paste(store[2], (0, 28))
        img.paste(store[3], (28, 28))
        return img, label
    
ds = combine()
img, label = ds[0]
print(label)
img.show()

