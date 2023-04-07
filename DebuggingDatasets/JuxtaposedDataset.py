import pickle
import random
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class JuxtaposedDataset(Dataset):
    def __init__(self, transforms) -> None:
        super().__init__()
        self.mnist_dataset = MNIST("./", download=True)
        self.transforms = transforms

        juxtaposition_data_path = Path("./juxtapositions.pkl")
        if not juxtaposition_data_path.exists():
            self.generate_random_juxtapositions()
        self.juxtapositions = self.load_random_juxtapositions()


    def __len__(self):
        return len(self.juxtapositions)

    def __getitem__(self, index):
        p1, p2 = self.juxtapositions[index]
        img1, label1 = self.mnist_dataset[p1]
        img2, label2 = self.mnist_dataset[p2]
        combined_img = self.transforms(np.hstack((img1, img2)))
        return combined_img, label1 * 10 + label2, p1, p2

    def generate_random_juxtapositions(self):
        juxtapositions = []
        indices = [i for i in range(0, len(self.mnist_dataset))]

        while len(indices) >= 1:
            p1 = random.randint(0, len(indices) - 1)
            indices.pop(p1)
            p2 = random.randint(0, len(indices) - 1)
            indices.pop(p2)
            juxtapositions.append((p1, p2))

        with open("./juxtapositions.pkl", "wb") as f:
            pickle.dump(juxtapositions, f)


    def load_random_juxtapositions(self):
        with open("./juxtapositions.pkl", "rb") as f:
            data = pickle.load(f)
        return data
