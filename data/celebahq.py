import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
from PIL import Image

class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None,
                 uniform_dequantization=False, random_flip=False):
        self.size = size
        self.random_crop = random_crop
        self.uniform_dequantization = uniform_dequantization
        self.random_flip = random_flip
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
        if self.size is not None and self.size > 0:
            pipeline = list()
            pipeline.append(albumentations.SmallestMaxSize(max_size = self.size))
            if not self.random_crop:
                pipeline.append(albumentations.CenterCrop(height=self.size,width=self.size))
            else:
                pipeline.append(albumentations.RandomCrop(height=self.size,width=self.size))
            if self.random_flip:
                pipeline.append(albumentations.HorizontalFlip())
            self.preprocessor = albumentations.Compose(pipeline)
        else:
            self.preprocessor = lambda **kwargs: kwargs
    def __len__(self):
        return self._length
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        if self.uniform_dequantization:
            image = image.astype(np.float32)
            image = image + np.random.uniform()/256.
        image = (image/127.5 - 1.0).astype(np.float32)   # in range -1 ... 1
        return image
    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex
class CelebAHQTrain(FacesBase):
    def __init__(self, config, keys=None):
        super().__init__()
        root = "/export/scratch/compvis/datasets/celeba_hq_full_resolution/celebA-HQ/"
        with open("data/celebahqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=config.data.image_size, random_crop=False,
                               random_flip=config.data.random_flip, uniform_dequantization=config.data.uniform_dequantization)
        self.keys = keys
class CelebAHQValidation(FacesBase):
    def __init__(self, config, keys=None):
        super().__init__()
        root = "/export/scratch/compvis/datasets/celeba_hq_full_resolution/celebA-HQ/"
        with open("data/celebahqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=config.data.image_size,
                               random_crop=False,uniform_dequantization=config.data.uniform_dequantization)
        self.keys = keys