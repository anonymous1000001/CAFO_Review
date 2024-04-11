import os 
import pandas as pd
import numpy as np
import pickle
from dataset.make_images import generate_rp

def load_rp_dict(cfg, sensor_dict, mode):
    # check if rp_dict exists
    dir_path = f"data/{cfg.task.features_save_dir}/{cfg.task.task_name}_rp_{mode}_dict.pkl"
    if os.path.exists(dir_path):
        with open(dir_path, "rb") as f:
            rp_dict = pickle.load(f)
            print(f"{dir_path} Exists!, loading..")
    else:
        # make 
        rp_dict = {}
        print(f"{dir_path} Does not exist!, making..")
        for key, value in sensor_dict.items():
            rp_dict[key] = generate_rp(value, cfg, key)

        with open(dir_path, "wb") as f:
            pickle.dump(rp_dict, f)
