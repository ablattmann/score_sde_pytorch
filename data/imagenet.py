import sys, os, tarfile, pickle, glob, shutil
import yaml
import numpy as np
from tqdm import tqdm, trange
from PIL import Image
import albumentations
from torch.utils.data import Dataset, Subset

from data.celebahq import ImagePaths
# from braket.util import download, retrieve

import data.util as bdu


def give_synsets_from_indices(indices, path_to_yaml="data/imagenet_idx_to_synset.yaml"):
    synsets = []
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    for idx in indices:
        synsets.append(str(di2s[idx]))
    print("Using {} different synsets for construction of Restricted Imagenet.".format(len(synsets)))
    return synsets


def str_to_indices(string):
    """Expects a string in the format '32-123, 256, 280-321'"""
    assert not string.endswith(","), "provided string '{}' ends with a comma, pls remove it".format(string)
    subs = string.split(",")
    indices = []
    for sub in subs:
        subsubs = sub.split("-")
        assert len(subsubs) > 0
        if len(subsubs) == 1:
            indices.append(int(subsubs[0]))
        else:
            rang = [j for j in range(int(subsubs[0]), int(subsubs[1]))]
            indices.extend(rang)
    return sorted(indices)


class ImageNetBase(Dataset):
    def __init__(self, config=None, uniform_dequantization=False):
        self.config = config
        self.uniform_dequantization = uniform_dequantization

        self._prepare()
        self._prepare_synset_to_human()
        self._prepare_idx_to_synset()
        self._prepare_human_to_integer_label()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _prepare(self):
        raise NotImplementedError()

    def _filter_relpaths(self, relpaths):
        ignore = set([
            "n06596364_9591.JPEG",
        ])
        relpaths = [rpath for rpath in relpaths if not rpath.split("/")[-1] in ignore]

        return relpaths

    def _prepare_synset_to_human(self):
        SIZE = 2655750
        URL = "https://heibox.uni-heidelberg.de/f/9f28e956cd304264bb82/?dl=1"
        self.human_dict = os.path.join(self.root, "synset_human.txt")
        # if (not os.path.exists(self.human_dict) or
        #         not os.path.getsize(self.human_dict)==SIZE):
        #     download(URL, self.human_dict)

    def _prepare_idx_to_synset(self):
        URL = "https://heibox.uni-heidelberg.de/f/d835d5b6ceda4d3aa910/?dl=1"
        self.idx2syn = os.path.join(self.root, "index_synset.yaml")
        # if (not os.path.exists(self.idx2syn)):
        #     download(URL, self.idx2syn)

    def _prepare_human_to_integer_label(self):
        URL = "https://heibox.uni-heidelberg.de/f/2362b797d5be43b883f6/?dl=1"
        self.human2integer = os.path.join(self.root, "imagenet1000_clsidx_to_labels.txt")
        # if (not os.path.exists(self.human2integer)):
        #     download(URL, self.human2integer)
        with open(self.human2integer, "r") as f:
            lines = f.read().splitlines()
            assert len(lines) == 1000
            self.human2integer_dict = dict()
            for line in lines:
                value, key = line.split(":")
                self.human2integer_dict[key] = int(value)

    def _load(self):
        with open(self.txt_filelist, "r") as f:
            self.relpaths = f.read().splitlines()
            l1 = len(self.relpaths)
            self.relpaths = self._filter_relpaths(self.relpaths)
            print("Removed {} files from filelist during filtering.".format(l1 - len(self.relpaths)))

        self.synsets = [p.split("/")[0] for p in self.relpaths]
        self.abspaths = [os.path.join(self.datadir, p) for p in self.relpaths]

        unique_synsets = np.unique(self.synsets)
        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        self.class_labels = [class_dict[s] for s in self.synsets]

        with open(self.human_dict, "r") as f:
            human_dict = f.read().splitlines()
            human_dict = dict(line.split(maxsplit=1) for line in human_dict)

        self.human_labels = [human_dict[s] for s in self.synsets]

        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            "human_label": np.array(self.human_labels),
        }
        self.data = ImagePaths(self.abspaths,
                               labels=labels,
                               size=self.config.data.image_size,
                               random_crop=self.config.data.random_crop,
                               uniform_dequantization=self.uniform_dequantization,
                               centered=self.config.data.centered)


class ImageNetTrain(ImageNetBase):
    NAME = "ILSVRC2012_train"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "a306397ccf9c2ead27155983c254227c0fd938e2"
    FILES = [
        "ILSVRC2012_img_train.tar",
    ]
    SIZES = [
        147897477120,
    ]

    def _prepare(self):
        self.random_crop = self.config.data.random_crop
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join(self.root, "filelist.txt")
        self.expected_length = 1281167
        if not bdu.is_prepared(self.root):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
                    import academictorrents as at
                    atpath = at.get(self.AT_HASH, datastore=self.root)
                    assert atpath == path

                print("Extracting {} to {}".format(path, datadir))
                os.makedirs(datadir, exist_ok=True)
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                print("Extracting sub-tars.")
                subpaths = sorted(glob.glob(os.path.join(datadir, "*.tar")))
                for subpath in tqdm(subpaths):
                    subdir = subpath[:-len(".tar")]
                    os.makedirs(subdir, exist_ok=True)
                    with tarfile.open(subpath, "r:") as tar:
                        tar.extractall(path=subdir)

            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            bdu.mark_prepared(self.root)

class ImageNetTrainWrap(ImageNetTrain):
    def __init__(self, **kwargs):
        super().__init__(config=kwargs)

class ImageNetValidation(ImageNetBase):
    NAME = "ILSVRC2012_validation"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5"
    VS_URL = "https://heibox.uni-heidelberg.de/f/3e0f6e9c624e45f2bd73/?dl=1"
    FILES = [
        "ILSVRC2012_img_val.tar",
        "validation_synset.txt",
    ]
    SIZES = [
        6744924160,
        1950000,
    ]

    def _prepare(self):
        self.random_crop = False
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join(self.root, "filelist.txt")
        self.expected_length = 50000
        if not bdu.is_prepared(self.root):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
                    import academictorrents as at
                    atpath = at.get(self.AT_HASH, datastore=self.root)
                    assert atpath == path

                print("Extracting {} to {}".format(path, datadir))
                os.makedirs(datadir, exist_ok=True)
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                vspath = os.path.join(self.root, self.FILES[1])
                if not os.path.exists(vspath) or not os.path.getsize(vspath)==self.SIZES[1]:
                    download(self.VS_URL, vspath)

                with open(vspath, "r") as f:
                    synset_dict = f.read().splitlines()
                    synset_dict = dict(line.split() for line in synset_dict)

                print("Reorganizing into synset folders")
                synsets = np.unique(list(synset_dict.values()))
                for s in synsets:
                    os.makedirs(os.path.join(datadir, s), exist_ok=True)
                for k, v in synset_dict.items():
                    src = os.path.join(datadir, k)
                    dst = os.path.join(datadir, v)
                    shutil.move(src, dst)

            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            bdu.mark_prepared(self.root)

class ImageNetValidationWrap(ImageNetValidation):
    def __init__(self, **kwargs):
        super().__init__(config=kwargs)

class ImageNetHQTrain(ImageNetBase):
    NAME = "ILSVRC2012_train"
    def _prepare(self):
        self.random_crop = True
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join("data/imagenet_hq_train_filelist.txt")  # hacky, but well...


class ImageNetHQValidation(ImageNetBase):
    NAME = "ILSVRC2012_validation"
    def _prepare(self):
        self.random_crop = False
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join("data/imagenet_hq_validation_filelist.txt")  # hacky, but well...


def get_preprocessor(size=None, random_crop=False, additional_targets=None,
                     crop_size=None):
    if size is not None and size > 0:
        transforms = list()
        rescaler = albumentations.SmallestMaxSize(max_size = size)
        transforms.append(rescaler)
        if not random_crop:
            cropper = albumentations.CenterCrop(height=size,width=size)
            transforms.append(cropper)
        else:
            cropper = albumentations.RandomCrop(height=size,width=size)
            transforms.append(cropper)
            flipper = albumentations.HorizontalFlip()
            transforms.append(flipper)
        preprocessor = albumentations.Compose(transforms,
                                              additional_targets=additional_targets)
    elif crop_size is not None and crop_size > 0:
        if not random_crop:
            cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
        else:
            cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
        transforms = [cropper]
        preprocessor = albumentations.Compose(transforms,
                                              additional_targets=additional_targets)
    else:
        preprocessor = lambda **kwargs: kwargs
    return preprocessor


def rgba_to_depth(x):
    assert x.dtype == np.uint8
    assert len(x.shape) == 3 and x.shape[2] == 4
    y = x.copy()
    y.dtype = np.float32
    y = y.reshape(x.shape[:2])
    return np.ascontiguousarray(y)


class BaseWithDepth(Dataset):
    DEFAULT_DEPTH_ROOT="/export/scratch/compvis/datasets/ImageNetDepth"

    def __init__(self, config=None, size=None, random_crop=False,
                 crop_size=None, root=None):
        self.config = config
        self.base_dset = self.get_base_dset()
        self.preprocessor = get_preprocessor(
            size=size,
            crop_size=crop_size,
            random_crop=random_crop,
            additional_targets={"depth": "image"})
        self.crop_size = crop_size
        if self.crop_size is not None:
            self.rescaler = albumentations.Compose(
                [albumentations.SmallestMaxSize(max_size = self.crop_size)],
                additional_targets={"depth": "image"})
        if root is not None:
            self.DEFAULT_DEPTH_ROOT = root

    def __len__(self):
        return len(self.base_dset)

    def preprocess_depth(self, path):
        rgba = np.array(Image.open(path))
        depth = rgba_to_depth(rgba)
        depth = (depth - depth.min())/max(1e-8, depth.max()-depth.min())
        depth = 2.0*depth-1.0
        return depth

    def __getitem__(self, i):
        e = self.base_dset[i]
        e["depth"] = self.preprocess_depth(self.get_depth_path(e))
        # up if necessary
        h,w,c = e["image"].shape
        if self.crop_size and min(h,w) < self.crop_size:
            # have to upscale to be able to crop - this just uses bilinear
            out = self.rescaler(image=e["image"], depth=e["depth"])
            e["image"] = out["image"]
            e["depth"] = out["depth"]
        transformed = self.preprocessor(image=e["image"], depth=e["depth"])
        e["image"] = transformed["image"]
        e["depth"] = transformed["depth"]
        return e


class ImageNetTrainWithDepth(BaseWithDepth):
    # default to random_crop=True
    def __init__(self, random_crop=True, sub_indices=None, **kwargs):
        self.sub_indices = sub_indices
        super().__init__(random_crop=random_crop, **kwargs)

    def get_base_dset(self):
        if self.sub_indices is None:
            return ImageNetTrain()
        else:
            return ImageNetTrain({"sub_indices": self.sub_indices})

    def get_depth_path(self, e):
        fid = os.path.splitext(e["relpath"])[0]+".png"
        fid = os.path.join(self.DEFAULT_DEPTH_ROOT, "train", fid)
        return fid


class ImageNetValidationWithDepth(BaseWithDepth):
    def __init__(self, sub_indices=None, **kwargs):
        self.sub_indices = sub_indices
        super().__init__(**kwargs)

    def get_base_dset(self):
        if self.sub_indices is None:
            return ImageNetValidation()
        else:
            return ImageNetValidation({"sub_indices": self.sub_indices})

    def get_depth_path(self, e):
        fid = os.path.splitext(e["relpath"])[0]+".png"
        fid = os.path.join(self.DEFAULT_DEPTH_ROOT, "val", fid)
        return fid


class RINTrainWithDepth(ImageNetTrainWithDepth):
    def __init__(self, config=None, size=None, random_crop=True, crop_size=None):
        sub_indices = "30-32, 33-37, 151-268, 281-285, 80-100, 365-382, 389-397, 118-121, 300-319"
        super().__init__(config=config, size=size, random_crop=random_crop,
                         sub_indices=sub_indices, crop_size=crop_size)


class RINValidationWithDepth(ImageNetValidationWithDepth):
    def __init__(self, config=None, size=None, random_crop=False, crop_size=None):
        sub_indices = "30-32, 33-37, 151-268, 281-285, 80-100, 365-382, 389-397, 118-121, 300-319"
        super().__init__(config=config, size=size, random_crop=random_crop,
                         sub_indices=sub_indices, crop_size=crop_size)


def imscale(x, factor, keepshapes=False, keepmode="bicubic", is_downfactor=True):
    if factor is None or factor==1:
        return x

    dtype = x.dtype
    assert dtype in [np.float32, np.float64]
    assert x.min() >= -1
    assert x.max() <= 1

    keepmode = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR,
                "bicubic": Image.BICUBIC}[keepmode]

    lr = (x+1.0)*127.5
    lr = lr.clip(0,255).astype(np.uint8)
    lr = Image.fromarray(lr)

    h, w, _ = x.shape
    if is_downfactor:
        nh = h//factor
        nw = w//factor
    else:
        nh = int(round(h*factor))
        nw = int(round(w*factor))
    assert nh > 0 and nw > 0, (nh, nw)

    lr = lr.resize((nw,nh), Image.BICUBIC)
    if keepshapes:
        lr = lr.resize((w,h), keepmode)
    lr = np.array(lr)/127.5-1.0
    lr = lr.astype(dtype)

    return lr


class ImageNetScale(Dataset):
    def __init__(self, size=None, crop_size=None, random_crop=False,
                 up_factor=None, hr_factor=None, keep_mode="bicubic"):
        self.base = self.get_base()

        self.size = size
        self.crop_size = crop_size if crop_size is not None else self.size
        self.random_crop = random_crop
        self.up_factor = up_factor
        self.hr_factor = hr_factor
        self.keep_mode = keep_mode

        transforms = list()

        if self.size is not None and self.size > 0:
            #assert False, ("Not sure if we should still support this, use "
            #               "hr_factor to adjust resolution, then only crop.")
            rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            self.rescaler = rescaler
            transforms.append(rescaler)

        if self.crop_size is not None and self.crop_size > 0:
            if len(transforms) == 0:
                self.rescaler = albumentations.SmallestMaxSize(max_size = self.crop_size)

            if not self.random_crop:
                cropper = albumentations.CenterCrop(height=self.crop_size,width=self.crop_size)
            else:
                cropper = albumentations.RandomCrop(height=self.crop_size,width=self.crop_size)
            transforms.append(cropper)

        if len(transforms) > 0:
            if self.up_factor is not None:
                additional_targets = {"lr": "image"}
            else:
                additional_targets = None
            self.preprocessor = albumentations.Compose(transforms,
                                                       additional_targets=additional_targets)
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        example = self.base[i]
        image = example["image"]
        # adjust resolution
        image = imscale(image, self.hr_factor, keepshapes=False)
        h,w,c = image.shape
        if self.crop_size and min(h,w) < self.crop_size:
            # have to upscale to be able to crop - this just uses bilinear
            image = self.rescaler(image=image)["image"]
        if self.up_factor is None:
            image = self.preprocessor(image=image)["image"]
            example["image"] = image
        else:
            lr = imscale(image, self.up_factor, keepshapes=True,
                         keepmode=self.keep_mode)

            out = self.preprocessor(image=image, lr=lr)
            example["image"] = out["image"]
            example["lr"] = out["lr"]

        return example

class ImageNetScaleTrain(ImageNetScale):
    def __init__(self, random_crop=True, **kwargs):
        super().__init__(random_crop=random_crop, **kwargs)

    def get_base(self):
        return ImageNetTrain()

class ImageNetScaleValidation(ImageNetScale):
    def get_base(self):
        return ImageNetValidation()


from skimage.feature import canny
from skimage.color import rgb2gray


class ImageNetEdges(ImageNetScale):
    def __init__(self, up_factor=1, **kwargs):
        super().__init__(up_factor=1, **kwargs)

    def __getitem__(self, i):
        example = self.base[i]
        image = example["image"]
        h,w,c = image.shape
        if self.crop_size and min(h,w) < self.crop_size:
            # have to upscale to be able to crop - this just uses bilinear
            image = self.rescaler(image=image)["image"]

        lr = canny(rgb2gray(image), sigma=2)
        lr = lr.astype(np.float32)
        lr = lr[:,:,None][:,:,[0,0,0]]

        out = self.preprocessor(image=image, lr=lr)
        example["image"] = out["image"]
        example["lr"] = out["lr"]

        return example


class ImageNetEdgesTrain(ImageNetEdges):
    def __init__(self, random_crop=True, **kwargs):
        super().__init__(random_crop=random_crop, **kwargs)

    def get_base(self):
        return ImageNetTrain()

class ImageNetEdgesValidation(ImageNetEdges):
    def get_base(self):
        return ImageNetValidation()


class ImageNetTwoScale(Dataset):
    def __init__(self, base_config, factor1, factor2):
        self.base = self.get_base()(base_config)
        self.factor1 = factor1
        self.factor2 = factor2

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        example = self.base[i]
        image = example["image"]
        # adjust resolution
        image1 = imscale(image, self.factor1, keepshapes=False, is_downfactor=False)
        image2 = imscale(image, self.factor2, keepshapes=False, is_downfactor=False)
        example["image"] = image1
        example["lr"] = image2
        return example

class ImageNetTwoScaleTrain(ImageNetTwoScale):
    def get_base(self):
        return ImageNetTrain

class ImageNetTwoScaleValidation(ImageNetTwoScale):
    def get_base(self):
        return ImageNetValidation


class BigGANImageNet(Dataset):
    def __init__(self, size, lr_size=None, zoom_pixels=None):
        self.base = self.init_base() # unscaled images
        self.size = size
        self.lr_size = lr_size
        self.zoom_pixels = zoom_pixels  # central crop, works as conditioning and also resized to size
        if zoom_pixels is not None: assert self.zoom_pixels > 0

    def __len__(self):
        return len(self.base)

    def preprocess(self, x):
        dtype = x.dtype
        assert dtype in [np.float32, np.float64]
        assert x.min() >= -1
        assert x.max() <= 1

        x = (x+1.0)*127.5
        x = x.clip(0,255).astype(np.uint8)

        h, w, _ = x.shape
        # largest central crop
        b = min(h, w)
        h0 = (h-b)//2
        w0 = (w-b)//2
        tmp = x[h0:h0+b,w0:w0+b,:]

        zoom = None
        # optional zoom
        if self.zoom_pixels is not None:
            hz, wz, _ = tmp.shape
            bz = self.zoom_pixels // 2
            zoom = tmp[bz:hz-bz, bz:wz-bz, :]
            zoom = Image.fromarray(zoom)
            zoom = zoom.resize((self.size, self.size), Image.BILINEAR)
            zoom = np.array(zoom)/127.5-1.0
            zoom = zoom.astype(dtype)

        x = tmp
        # pillow bilinear instead of tensorflow area resizing
        x = Image.fromarray(x)
        x = x.resize((self.size, self.size), Image.BILINEAR)

        # optional lr image
        if self.lr_size is not None:
            lr = x.resize((self.lr_size, self.lr_size), Image.BICUBIC)
            lr = lr.resize((self.size, self.size), Image.BICUBIC)
            lr = np.array(lr)/127.5-1.0
            lr = lr.astype(dtype)
        else:
            lr = None

        x = np.array(x)/127.5-1.0
        x = x.astype(dtype)

        return x, lr, zoom

    def __getitem__(self, i):
        example = self.base[i]
        x, lr, zoom = self.preprocess(example["image"])
        example["image"] = x
        if zoom is not None:
            example["image_zoom"] = zoom
        if self.lr_size is not None:
            example["lr"] = lr
        return example


class BigGANImageNetTrain(BigGANImageNet):
    def init_base(self):
        return ImageNetTrain()


def BigGANImageNetTrainHR(size, lr_size, zoom_pixels=None):
    # only images with minsize >= 256
    with open("data/imagenet_train_hr_indices.p", "rb") as f:
        indices = pickle.load(f)
    dset = BigGANImageNetTrain(size, lr_size, zoom_pixels=zoom_pixels)
    return Subset(dset, indices)


class BigGANImageNetValidation(BigGANImageNet):
    def init_base(self):
        return ImageNetValidation()


def BigGANImageNetValidationHR(size, lr_size, zoom_pixels=None):
    # only images with minsize >= 256
    with open("data/imagenet_val_hr_indices.p", "rb") as f:
        indices = pickle.load(f)
    dset = BigGANImageNetValidation(size, lr_size, zoom_pixels=zoom_pixels)
    return Subset(dset, indices)


if __name__ == "__main__":
    dset = RINValidationWithDepth(size=290, crop_size=256)
    ex = dset[0]
    print(ex.keys())
    print(ex["image"].shape)
    print("done.")

    def prep_hr():
        dsets = {"train": ImageNetTrain(), "val": ImageNetValidation()}
        for split, dset in dsets.items():
            hr_indices = list()
            for i in trange(len(dset)):
                h,w,c = dset[i]["image"].shape
                if min(h, w) >= 256:
                    hr_indices.append(i)
            print(f"Found {len(hr_indices)}/{len(dset)} images larger than 256.")
            with open(f"data/imagenet_{split}_hr_indices.p", "wb") as f:
                pickle.dump(hr_indices, f)

        for dset in [BigGANImageNetTrainHR(256,64),
                     BigGANImageNetValidationHR(256,64)]:
            print(len(dset))
            ex = dset[0]
            print(ex.keys())
            print(ex["image"].shape)
            print(ex["lr"].shape)

    #prep_hr()
