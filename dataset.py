import os
import cv2
import numbers
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
import pytorch_lightning as pl

from sklearn.utils import shuffle
from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_ubyte
from torch.utils.data import DataLoader, Dataset
from stocaching import SharedCache
from pathlib import Path
from sampler import SamplerFactory

if Path("/data2/mb121/EMBED/images/png/1024x768").exists():
    embed_data_dir = "/data2/mb121/EMBED/images/png/1024x768"
elif Path("/data/EMBED/images/png/1024x768").exists():
    embed_data_dir = "/data/EMBED/images/png/1024x768"
else:
    embed_data_dir = "/vol/biomedic3/data/EMBED/images/png/1024x768"

ANNOTATION_FILE = "labelling_tools/manual_annotations_full_new.csv"


class GammaCorrectionTransform:
    """Apply Gamma Correction to the image"""

    def __init__(self, gamma=0.5):
        self.gamma = self._check_input(gamma, "gammacorrection")

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for gamma correction do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: gamma corrected image.
        """
        gamma_factor = (
            None
            if self.gamma is None
            else float(torch.empty(1).uniform_(self.gamma[0], self.gamma[1]))
        )
        if gamma_factor is not None:
            img = TF.adjust_gamma(img, gamma_factor, gain=1)
        return img


class MammoDataset(Dataset):
    def __init__(
        self,
        data,
        target,
        image_size,
        image_normalization,
        horizontal_flip=False,
        augmentation=False,
        cache_size=0,
    ):
        self.image_size = image_size
        self.image_normalization = image_normalization
        self.do_flip = horizontal_flip
        self.do_augment = augmentation

        # photometric data augmentation
        self.photometric_augment = T.Compose(
            [
                GammaCorrectionTransform(gamma=0.2),
                T.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )

        # geometric data augmentation
        self.geometric_augment = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    transforms=[T.RandomAffine(degrees=10, scale=(0.95, 1.05))], p=0.75
                ),
            ]
        )

        self.img_paths = data.img_path.to_numpy()
        self.study_ids = data.study_id.to_numpy()
        self.image_ids = data.image_id.to_numpy()
        match target:
            case "cancer":
                self.labels = data.is_positive.to_numpy()
            case "density":
                self.labels = data.density_label.to_numpy()
            case "artifact":
                self.labels = data.multilabel_markers.to_numpy()

        # initialize the cache
        self.cache = None
        self.use_cache = cache_size > 0
        if self.use_cache:
            self.cache = SharedCache(
                size_limit_gib=cache_size,
                dataset_len=self.labels.shape[0],
                data_dims=(1, image_size[0], image_size[1]),
                dtype=torch.float32,
            )

        self.mask = target != "artifact"

    def preprocess(self, image, horizontal_flip, mask):

        # resample
        if self.image_size != image.shape:
            image = resize(image, output_shape=self.image_size, preserve_range=True)

        # breast mask
        if mask:
            image_norm = image - np.min(image)
            image_norm = image_norm / np.max(image_norm)
            thresh = cv2.threshold(img_as_ubyte(image_norm), 5, 255, cv2.THRESH_BINARY)[
                1
            ]

            # Connected components with stats.
            nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
                thresh, connectivity=4
            )

            # Find the largest non background component.
            # Note: range() starts from 1 since 0 is the background label.
            max_label, _ = max(
                [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)],
                key=lambda x: x[1],
            )
            mask = output == max_label
            image[mask == 0] = 0

        # flip
        if horizontal_flip:
            left = np.mean(image[:, 0 : int(image.shape[1] / 2)])  # noqa
            right = np.mean(image[:, int(image.shape[1] / 2) : :])  # noqa
            if left < right:
                image = image[:, ::-1].copy()

        return image

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        image = None
        if self.use_cache:
            image = self.cache.get_slot(index)

        if image is None:
            img_path = self.img_paths[index]
            image = imread(img_path).astype(np.float32)
            horizontal_flip = self.do_flip
            image = self.preprocess(image, horizontal_flip, mask=self.mask)
            image = torch.from_numpy(image).unsqueeze(0)

            if self.use_cache:
                self.cache.set_slot(index, image, allow_overwrite=True)

        # normalize intensities to range [0,1]
        image = image / self.image_normalization

        if self.do_augment:
            image = self.photometric_augment(image)
            image = self.geometric_augment(image)

        image = image.repeat(3, 1, 1)

        return {
            "image": image,
            "label": self.labels[index],
            "study_id": self.study_ids[index],
            "image_id": self.img_paths[index],
        }

    def get_labels(self):
        return self.labels


class EMBEDMammoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_file,
        image_size,
        target,
        batch_alpha=0,
        batch_size=32,
        num_workers=6,
        split_dataset=True,
        data_dir=embed_data_dir,
    ):
        super().__init__()
        self.target = target
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_alpha = batch_alpha
        self.batch_size = batch_size
        self.num_workers = num_workers
        if isinstance(csv_file, pd.DataFrame):
            self.data = csv_file
        else:
            self.data = pd.read_csv(csv_file)

        if target != "artifact":
            test_percent = 0.25
            val_percent = 0.1
            # FFDM only
            self.data = self.data[self.data["FinalImageType"] == "2D"]

            # Female only
            self.data = self.data[self.data["GENDER_DESC"] == "Female"]

            # Remove unclear breast density cases
            self.data = self.data[self.data["tissueden"].notna()]
            self.data = self.data[self.data["tissueden"] < 5]

            # MLO and CC only
            self.data = self.data[self.data["ViewPosition"].isin(["MLO", "CC"])]

            # Remove spot compression or magnificiation
            self.data = self.data[self.data["spot_mag"].isna()]
            self.data["laterality"] = self.data["ImageLateralityFinal"]

        else:
            test_percent = 0.25
            # this is 20% of remaining split!
            val_percent = 0.20

        self.test_percent = test_percent
        self.val_percent = val_percent

        self.data["img_path"] = [
            os.path.join(self.data_dir, img_path)
            for img_path in self.data.image_path.values
        ]
        self.data["study_id"] = [
            str(study_id) for study_id in self.data.empi_anon.values
        ]
        self.data["image_id"] = [
            img_path.split("/")[-1] for img_path in self.data.image_path.values
        ]

        if target == "density":
            # Define density categories
            self.data["density_label"] = 0
            self.data.loc[self.data["tissueden"] == 1, "density_label"] = 0
            self.data.loc[self.data["tissueden"] == 2, "density_label"] = 1
            self.data.loc[self.data["tissueden"] == 3, "density_label"] = 2
            self.data.loc[self.data["tissueden"] == 4, "density_label"] = 3

        if split_dataset:
            # Split data into training, validation, and testing
            # Making sure images from the same subject are within the same set
            self.data["split"] = "test"
            unique_study_ids_all = self.data.empi_anon.unique()
            unique_study_ids_all = shuffle(unique_study_ids_all, random_state=33)
            num_test = round(len(unique_study_ids_all) * self.test_percent)

            dev_sub_id = unique_study_ids_all[num_test:]
            self.data.loc[self.data.empi_anon.isin(dev_sub_id), "split"] = "training"

            self.dev_data = self.data[self.data["split"] == "training"]
            self.test_data = self.data[self.data["split"] == "test"]

            unique_study_ids_dev = self.dev_data.empi_anon.unique()

            unique_study_ids_dev = shuffle(unique_study_ids_dev, random_state=33)
            num_train = round(len(unique_study_ids_dev) * (1.0 - self.val_percent))

            valid_sub_id = unique_study_ids_dev[num_train:]
            self.dev_data.loc[self.dev_data.empi_anon.isin(valid_sub_id), "split"] = (
                "validation"
            )

            self.train_data = self.dev_data[self.dev_data["split"] == "training"]
            self.val_data = self.dev_data[self.dev_data["split"] == "validation"]

            self.train_set = MammoDataset(
                data=self.train_data,
                image_size=self.image_size,
                target=target,
                image_normalization=65535.0,
                horizontal_flip=True,
                augmentation=True,
                cache_size=32,
            )
            self.val_set = MammoDataset(
                data=self.val_data,
                image_size=self.image_size,
                target=target,
                image_normalization=65535.0,
                horizontal_flip=True,
                augmentation=False,
                cache_size=8,
            )
            self.test_set = MammoDataset(
                data=self.test_data,
                image_size=self.image_size,
                target=target,
                image_normalization=65535.0,
                horizontal_flip=True,
                augmentation=False,
            )

            train_labels = self.train_set.get_labels()

            val_labels = self.val_set.get_labels()

            test_labels = self.test_set.get_labels()

            if self.batch_alpha > 0:
                assert target != "artifact"
                train_class_idx = [
                    np.where(train_labels == t)[0] for t in np.unique(train_labels)
                ]
                train_batches = len(self.train_set) // self.batch_size

                self.train_sampler = SamplerFactory().get(
                    train_class_idx,
                    self.batch_size,
                    train_batches,
                    alpha=self.batch_alpha,
                    kind="fixed",
                )

            print("samples (train): ", len(self.train_set))
            print("samples (val):   ", len(self.val_set))
            print("samples (test):  ", len(self.test_set))

            if target != "artifact":
                train_class_count = np.array(
                    [
                        len(np.where(train_labels == t)[0])
                        for t in np.unique(train_labels)
                    ]
                )
                val_class_count = np.array(
                    [len(np.where(val_labels == t)[0]) for t in np.unique(val_labels)]
                )
                test_class_count = np.array(
                    [len(np.where(test_labels == t)[0]) for t in np.unique(test_labels)]
                )
                print(train_class_count)
                print(val_class_count)
                print(test_class_count)
                if target == "cancer":
                    print(
                        "pos/neg (train): {}/{}".format(
                            train_class_count[1], train_class_count[0]
                        )
                    )
                    print(
                        "pos/neg (val):   {}/{}".format(
                            val_class_count[1], val_class_count[0]
                        )
                    )
                    print(
                        "pos/neg (test):  {}/{}".format(
                            test_class_count[1], test_class_count[0]
                        )
                    )
                    print(
                        "pos (train):     {:0.2f}%".format(
                            train_class_count[1] / len(train_labels) * 100.0
                        )
                    )
                    print(
                        "pos (val):       {:0.2f}%".format(
                            val_class_count[1] / len(val_labels) * 100.0
                        )
                    )
                    print(
                        "pos (test):      {:0.2f}%".format(
                            test_class_count[1] / len(test_labels) * 100.0
                        )
                    )

        else:
            # to predict for the full dataset
            self.test_set = MammoDataset(
                data=self.data,
                image_size=self.image_size,
                target=target,
                image_normalization=65535.0,
                horizontal_flip=True,
                augmentation=False,
            )

    def train_dataloader(self):
        if self.batch_alpha == 0:
            return DataLoader(
                dataset=self.train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            return DataLoader(
                dataset=self.train_set,
                batch_sampler=self.train_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
            )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

