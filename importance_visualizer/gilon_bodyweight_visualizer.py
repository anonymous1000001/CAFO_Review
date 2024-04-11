import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from easydict import EasyDict as edict

from importance_visualizer.base_visualizer import BaseVisualizer

# from base_visualizer import BaseVisualizer
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


class GilonBodyWeightVisualizer(BaseVisualizer):
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
            "below50": "Below 50kg",
            "50-60": "50kg to 60kg",
            "60-70": "60kg to 70kg",
            "above70": "Above 70kg",
        }
        return class_map

    def _load_model_performance(self):
        test_label = pd.read_csv(f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}_test_label.csv")
        # perform Weight binning by 40-50, 50-60, 60-70, 70-80,
        test_label["weight_bin"] = pd.cut(test_label["Weight"], bins=[0, 50, 60, 70, 200], labels=["below50", "50-60", "60-70", "above70"])
        # calculate mse between y_pred and y_true after groupby weight_bin
        class_rmse = test_label.groupby("weight_bin").apply(lambda x: mean_squared_error(x["y_true"], x["y_pred"], squared=False))
        class_rmse = class_rmse.apply(lambda x: f"{x:.3f}")
        return class_rmse

    def load_attention_label_csv(self):
        # Load the attention label csv file
        attention_label = pd.read_csv(f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}/attention_label.csv")
        attention_label["y_true_bodyweight"] = attention_label["y_true"].copy()
        attention_label["y_true"] = pd.cut(attention_label["y_true_bodyweight"], bins=[0, 50, 60, 70, 200], labels=["below50", "50-60", "60-70", "above70"])
        return attention_label

    def load_test_attention_scores_and_label_csv(self):
        # Load the test attention label csv file
        test_attention_label = pd.read_csv(f"{self.cfg.save_output_path}/test_{self.cfg.task.validation_cv_num}/channel_test_attention_scores.csv")
        test_attention_label["y_true_bodyweight"] = test_attention_label["y_true"].copy()
        test_attention_label["y_true"] = pd.cut(
            test_attention_label["y_true_bodyweight"], bins=[0, 50, 60, 70, 200], labels=["below50", "50-60", "60-70", "above70"]
        )
        return test_attention_label

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
        plt.savefig(self.global_attention_path)

    def plot_classwise_attention(self):
        # Plot heatmap of classwise_relative_importance
        class_rmse = self._load_model_performance()

        fig, ax1 = plt.subplots(figsize=(10, 6))
        multiplier = self.find_decimal_factor(self.class_wise_relative_importance.max())
        ax2 = ax1.twinx()

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

        ax2.set_yticks(np.arange(start, end, end / len(class_rmse)) + 0.07, labels=class_rmse.iloc[::-1])  # Need to check this label
        ax1.set_xticks([3, 6, 10, 14], labels=["Accel_Right", "Accel_Left", "FSR_Right", "FSR_Left"], rotation=45)
        ax1.axvline(x=3, color="r", ls="--", lw=2, label="accelerometer Right")
        ax1.axvline(x=6, color="b", ls="--", lw=2, label="accelerometer Left")
        ax1.axvline(x=10, color="g", ls="--", lw=2, label="FSR Right")
        ax1.axvline(x=14, color="y", ls="--", lw=2, label="FSR Left")
        plt.savefig(
            self.classwise_attention_path,
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[cax],
        )


if __name__ == "__main__":
    cfg = edict({"save_output_path": "outputs/gilon_bodyweight/EXP6000", "task": {"validation_cv_num": 0, "in_channels": 14}})
    summarizer = GilonBodyWeightVisualizer(cfg)
    summarizer.plot_global_attention()
    summarizer.plot_classwise_attention()
