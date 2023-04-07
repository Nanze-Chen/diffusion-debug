import pickle
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class TranslationDataset(Dataset):
    def __init__(self, transforms) -> None:
        super().__init__()
        self.mnist_dataset = MNIST("./", download=True)
        self.transforms = transforms

        translation_data_path = Path("./translations.pkl")
        if not translation_data_path.exists():
            self.translation_data = self.generate_random_translations()
        self.translations = self.load_random_translations()


    def __len__(self):
        return len(self.mnist_dataset) * 2

    def __getitem__(self, index):
        true_index = index if index < len(self.mnist_dataset) else index - len(self.mnist_dataset)
        image, label = self.mnist_dataset[true_index]

        if index < len(self.mnist_dataset):
            return self.transforms(image), label, (1, 0, 0, 0, 1, 0)
        image = image.transform(image.size, Image.AFFINE, self.translations[true_index])
        image = self.transforms(image)
        return image, label, self.translations[true_index]

    def generate_random_translations(self):
        translations = []
        for _ in range(len(self.mnist_dataset)):
            x, y = random.randint(-14, 14), random.randint(-14, 14)
            t = (1, 0, x, 0, 1, y)
            translations.append(t)

        with open("./translations.pkl", "wb") as f:
            pickle.dump(translations, f)


    def load_random_translations(self):
        with open("./translations.pkl", "rb") as f:
            data = pickle.load(f)
        return data
