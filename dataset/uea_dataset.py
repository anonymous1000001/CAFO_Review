import os
import pandas as pd
import numpy as np
import pickle
from typing import Optional
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from utils import bcolors
from dataset.make_images import generate_rp
from dataset.constant_sampler import ConstantRandomSampler

class RecbamDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str], permute_predict_name=None):
        if stage == "fit":
            self.train = UEAMTSDataset("train", self.cfg)
            self.val = UEAMTSDataset("val", self.cfg)
            self.train_sampler = ConstantRandomSampler(self.train)
        elif stage == "test":
            self.test = UEAMTSDataset("test", self.cfg)
            # self.test_sampler = ConstantRandomSampler(self.train)
        elif stage == "predict":
            pass
        else:
            raise ValueError(f"Unknown stage {stage}")

    def train_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.cfg.dataset.batch_size
        else:
            batch_size = batch_size # for linear probing
        return DataLoader(
            self.train,
            batch_size=batch_size,
            num_workers=self.cfg.dataset.num_workers,
            sampler=self.train_sampler,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.cfg.dataset.batch_size
        else:
            batch_size = batch_size # for linear probing
        return DataLoader(
            UEAMTSDataset("test", self.cfg),
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self, permute_predict_name=None, noise_predict_name=None):
        return DataLoader(
            UEAMTSDataset("predict", self.cfg, permute_predict_name=permute_predict_name, noise_predict_name=noise_predict_name),
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            shuffle=False,
            drop_last=False,
        )


class UEAMTSDataset(Dataset):
    """Load Gilon sensor data and meta data using global_id value. index is mapped to global id through label_df"""

    def __init__(self, mode, cfg):
        print(bcolors.OKBLUE + bcolors.BOLD + f"{mode} Mode" + bcolors.ENDC + bcolors.ENDC)
        self.mode = mode
        self.cfg = cfg

        if mode in ("train", "val"):
            self.label_df = pd.read_csv(f"data/{cfg.task.features_save_dir}/train_label.csv")
            data_dict_path = f"data/{cfg.task.features_save_dir}/train_dict.pkl"
            if mode == "train":
                self.label_df = self.label_df[self.label_df["is_train"] == True]
            elif mode == "val":
                self.label_df = self.label_df[self.label_df["is_train"] == False]
        elif mode in ("test"):
            self.label_df = pd.read_csv(f"data/{cfg.task.features_save_dir}/test_label.csv")
            data_dict_path = f"data/{cfg.task.features_save_dir}/test_dict.pkl"
        else:
            raise ValueError(f"Unknown mode {mode}")

        if os.path.isfile(data_dict_path):
            print(f"{data_dict_path} Exists!, loading..")
            with open(data_dict_path, "rb") as f:
                self.sensor_dict = pickle.load(f)
        else:
            raise ValueError(f"{data_dict_path} does not exist!")

        """If you want to perform any ablation on the datasets, please do it here. all the features will be based on the label_df"""

        get_label_statistics(self.label_df)
        self.label_df = self.label_df.reset_index(drop=True)

        if mode == "test":
            self.label_df.to_csv(f"{self.cfg.save_output_path}/test_label.csv", index=False)
    

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, index):
        # TODO: Check if the weighted sampling is working properly
        global_id = self.label_df.iloc[index].global_id
        label = self.label_df[self.label_df["global_id"] == global_id].y_true.values[0]

        original_feature = self.sensor_dict[global_id]
        y_true = torch.tensor(label, dtype=torch.long)

        feature = generate_rp(original_feature, self.cfg, global_id)

        return {
            "feature": torch.tensor(feature, dtype=torch.float32),
            "y_true": y_true,
            "global_id": global_id,
        }


def get_label_statistics(label_df):
    """for sanity check"""
    print("-" * 50)
    print(f"Number of unique global ID: {len(label_df.global_id.unique())}")
    print(f"y_true value counts: {label_df.y_true.value_counts()}")
    print("-" * 50)
