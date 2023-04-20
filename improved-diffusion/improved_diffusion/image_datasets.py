import os
import pickle
import random
from pathlib import Path

import blobfile as bf
import numpy as np
import torch
import torchvision.transforms as transforms
from mpi4py import MPI
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, MNIST


def load_data(
    *, data_dir, default_dataset, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir and not default_dataset:
        raise ValueError("Have to specify data_dir or default_dataset")
    elif data_dir and default_dataset:
        raise ValueError("Have to specify either data_dir or default_dataset")
    else:
        if data_dir:
            # data_dir specified
            all_files = _list_image_files_recursively(data_dir)
            classes = None
            if class_cond:
                # Assume classes are the first part of the filename,
                # before an underscore.
                class_names = [bf.basename(path).split("_")[0] for path in all_files]
                sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
                classes = [sorted_classes[x] for x in class_names]
            dataset = ImageDataset(
                image_size,
                all_files,
                classes=classes,
                shard=MPI.COMM_WORLD.Get_rank(),
                num_shards=MPI.COMM_WORLD.Get_size(),
            )
        else:
            # default_dataset specified
            dataset = DefaultDataset(default_dataset, image_size)

        # transform the dataset to dataloader
        if deterministic:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
            )
        else:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
            )
        while True:
            yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class DefaultDataset(Dataset):
    def __init__(self, default_dataset, image_size):
        super().__init__()
        self.original_dataset = self.__get_original_default_dataset__(default_dataset, image_size)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        out_dict = {}
        out_dict["y"] = np.array(y, dtype=np.int64)
        return x, out_dict

    def __get_original_default_dataset__(self, default_dataset, image_size):
        tran_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )
        if default_dataset == "MNIST":
            dataset = MNIST(
                os.path.join(os.getcwd(), "datasets", "mnist"),
                train=True,
                download=True,
                transform=tran_transform
            )
        elif default_dataset == "CIFAR10":
            dataset = CIFAR10(
                os.path.join(os.getcwd(), "datasets", "cifar10"),
                train=True,
                download=True,
                transform=tran_transform,
            )
        elif default_dataset == "Translations":
            dataset = TranslationDataset(tran_transform) 
        elif default_dataset == "Juxtaposition":
            dataset = JuxtaposedDataset(tran_transform)
        else:
            raise ValueError("Invalid default dataset is given")

        return dataset


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


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


class JuxtaposedDataset(Dataset):
    def __init__(self, transforms, prefix=0) -> None:
        super().__init__()
        self.prefix = prefix
        self.mnist_dataset = MNIST("./", download=True)
        self.transforms = transforms

        juxtaposition_data_path = Path(f"./{prefix}_juxtapositions.pkl")
        if not juxtaposition_data_path.exists():
            self.generate_random_juxtapositions()
        self.juxtapositions = self.load_random_juxtapositions()


    def __len__(self):
        return len(self.juxtapositions)

    def __getitem__(self, index):
        p1, p2 = self.juxtapositions[index]
        combined_img = Image.new("L", (p1[0].width + p2[0].width, p1[0].height))
        combined_img.paste(p1[0], (0, 0))
        combined_img.paste(p2[0], (p1[0].width, 0))
        return self.transforms(combined_img), p1[1] * 10 + p2[1]

    def generate_random_juxtapositions(self):
        juxtapositions = []
        prefix_data = []
        for i in self.mnist_dataset:
            if i[1] == self.prefix:
                prefix_data.append(i)
        
        index = 0
        for i in self.mnist_dataset:
            if i[1] == self.prefix:
                continue
            juxtapositions.append((prefix_data[index], i))
            index = (index + 1) % len(prefix_data)

        with open(f"./{self.prefix}_juxtapositions.pkl", "wb") as f:
            pickle.dump(juxtapositions, f)


    def load_random_juxtapositions(self):
        with open(f"./{self.prefix}_juxtapositions.pkl", "rb") as f:
            data = pickle.load(f)
        return data
