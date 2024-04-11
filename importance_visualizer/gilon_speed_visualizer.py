import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from easydict import EasyDict as edict
from importance_visualizer.base_visualizer import BaseVisualizer
from sklearn.metrics import mean_squared_error

GILON_14_FEATURE_LISTS = [
    "right_accel_x",
    "right_accel_y",
    "right_accel_z",
    "left_accel_x",
    "left_accel_y",
    "left_accel_z",
    "right_fsr1",
    "right_fsr2",
    "right_fsr3",
    "right_fsr4",
    "left_fsr1",
    "left_fsr2",
    "left_fsr3",
    "left_fsr4",
]


class GilonSpeedVisualizer(BaseVisualizer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _load_feature_lists(self):
        if self.cfg.task.in_channels == 14:
            feature_lists = GILON_14_FEATURE_LISTS
        else:
            raise ValueError("Invalid in_channels")
        return feature_lists

    def _define_class_map(self):
        class_map = {
            0: "0km/hr",
            3: "3km/hr",
            # 3.5: "3.5km/hr",
            4: "4km/hr",
            # 4.5: "4.5km/hr",
            5: "5km/hr",
            # 5.5: "5.5km/hr",
            7: "7km/hr",
            # 7.5: "7.5km/hr",
            8: "8km/hr",
        }
        return class_map

    def _load_model_performance(self):
        test_label = pd.read_csv(f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}_test_label.csv")
        class_mse = test_label.groupby("SPEED").apply(lambda x: mean_squared_error(x["y_true"], x["y_pred"]))
        class_mse = class_mse.apply(lambda x: f"{x:.3f}")
        return class_mse

    def plot_global_attention(self):
        # Construct dataframe with feature names and global importance
        df = pd.DataFrame({"Feature": self.feature_lists, "Importance Score": self.global_importance})
        df = df.sort_values(by="Importance Score", ascending=False)

        # Test dataframe with feature names and global importance
        test_df = pd.DataFrame({"Feature": self.feature_lists, "Importance Score": self.test_global_importance})
        test_df = test_df.sort_values(by="Importance Score", ascending=False)
        # plot seaborn barplot for both df and test_df side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        sns.barplot(x="Importance Score", y="Feature", data=df, ax=ax1, palette="Blues_d")
        sns.barplot(x="Importance Score", y="Feature", data=test_df, ax=ax2, palette="Blues_d")
        ax1.set_title("Train Global Importance")
        ax2.set_title("Test Global Importance")
        plt.tight_layout()

        plt.savefig(self.global_attention_path, dpi=300)

    def plot_classwise_attention(self):
        # Plot heatmap of classwise_relative_importance
        class_mse = self._load_model_performance()

        fig, ax1 = plt.subplots(figsize=(10, 6))
        multiplier = self.find_decimal_factor(self.class_wise_relative_importance.max())
        ax2 = ax1.twinx()

        # set title, leave extra vertical space between title and plot
        ax1.set_title(f"{self.cfg.model.model_name} - CV{self.cfg.task.validation_cv_num}", fontsize=20, pad=20)

        cax = inset_axes(
            ax1,
            width="40%",  # width: 40% of parent_bbox width
            height="10%",  # height: 10% of parent_bbox height
            loc="lower left",
            bbox_to_anchor=(0.6, 1.2, 1, 1),
            bbox_transform=ax2.transAxes,
            borderpad=0,
        )
        # convert self.class_wise_relative_importance to dataframe and set index to class name
        df = pd.DataFrame(self.class_wise_relative_importance, index=self.class_map.values())

        sns.heatmap(
            multiplier * df,
            center=0.0,
            annot=True,
            fmt=".3f",
            cmap="bwr",
            linewidths=0.5,
            ax=ax1,
            cbar_ax=cax,
            cbar_kws={"orientation": "horizontal"},
        )
        start, end = ax2.get_ylim()

        ax2.set_yticks(
            np.arange(start, end, end / len(class_mse)) + 0.07, labels=class_mse.iloc[::-1]
        )  # Need to check this label
        ax1.set_xticks([3, 6, 10, 14], labels=["Accel_Right", "Accel_Left", "FSR_Right", "FSR_Left"], rotation=45)
        ax1.axvline(x=3, color="r", ls="--", lw=2, label="accelerometer Right")
        ax1.axvline(x=6, color="b", ls="--", lw=2, label="accelerometer Left")
        ax1.axvline(x=10, color="g", ls="--", lw=2, label="FSR Right")
        ax1.axvline(x=14, color="y", ls="--", lw=2, label="FSR Left")

        # set y-label
        ax1.set_ylabel("Class", fontsize=16, labelpad=15, rotation=90)
        ax2.set_ylabel("MSE", fontsize=16, labelpad=20, rotation=270)
        # set x-label
        ax1.set_xlabel("Feature", fontsize=16, labelpad=8)

        plt.savefig(
            self.classwise_attention_path,
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[cax],
        )


if __name__ == "__main__":

    for exp_num in ["EXP2000", "EXP2001", "EXP2002"]:
        for cv_val in [0, 1, 2, 3]:
            cfg = edict(
                {
                    "model": {"model_name": "Shufflenet"},
                    "exp_num": exp_num,
                    "save_output_path": f"outputs/gilon_speed/{exp_num}",
                    "save_classwise_attention_path": "outputs/gilon_speed/classwise_attention",
                    "save_global_attention_path": "outputs/gilon_speed/global_attention",
                    "task": {"validation_cv_num": cv_val, "in_channels": 14},
                }
            )
            summarizer = GilonSpeedVisualizer(cfg)
            # summarizer.plot_global_attention()
            summarizer.plot_classwise_attention()
            # close all figures
            plt.close("all")
            del summarizer
