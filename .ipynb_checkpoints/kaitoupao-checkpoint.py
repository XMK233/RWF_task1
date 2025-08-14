import pandas as pd
import numpy as np
from tqdm import tqdm
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score,roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from time import process_time 
import lightgbm as lgb

import random, os, tqdm, time, json, re, IPython, sys

random.seed(618)
np.random.seed(907)
tqdm.tqdm.pandas() ## 引入这个，就可以在apply的时候用progress_apply了。

sys.path.append("/home/snbmod/WORKSPACE/24100129_xmk")
import utils

new_base_path = os.path.join(
    "/home/snbmod/WORKSPACE/24100129_xmk/data",
    "/".join(
        os.getcwd().split("/")[-1*(len(sys.path[-1].split("/")) - 2):]
    ),
)
print("storage dir:", new_base_path)
print("code dir:", os.getcwd())

## 创建文件夹。
if not os.path.exists(new_base_path):
    os.makedirs(
        new_base_path
    )
if not os.path.exists(os.path.join(new_base_path, "preprocessedData")):
    os.makedirs(
        os.path.join(new_base_path, "preprocessedData")
    )
if not os.path.exists(os.path.join(new_base_path, "originalData")):
    os.makedirs(
        os.path.join(new_base_path, "originalData")
    )
if not os.path.exists(os.path.join(new_base_path, "trained_models")):
    os.makedirs(
        os.path.join(new_base_path, "trained_models")
    )

def see_feaName_hierarchy(fea_list, split_sep = "_", dump_to_file = "dump_file.json"):
    s_l = sorted(fea_list)
    dic = {}
    fea_count = 0
    for i in s_l:
        i = i.strip()
        parts = i.split(split_sep)
        if parts[0] not in dic:
            dic[parts[0]] = {}
        cur_dic = dic[parts[0]]
        for part in parts[1:-1]:
            if part not in cur_dic:
                cur_dic[part] = {}
            cur_dic = cur_dic[part]
        cur_dic[parts[-1]] = i
        fea_count += 1
    # print(dic)
    print("fea num", fea_count)
    with open(dump_to_file, "w") as f:
        json.dump(dic, f, indent=4)

def read_feaList_from_file(fpath, do_lowering = True):
    with open(fpath, "r") as f:
        if do_lowering:
            feas = [i.strip().lower() for i in f.readlines()] #  if i.strip() != ""
        else:
            feas = [i.strip() for i in f.readlines()]
    print(len(feas), fpath)
    return feas

def save_feaList_to_file(feas, fpath, mode = "w"):
    if len(feas) == 0:
        print(f"Finished writing file: {fpath}. Wrote nothing.")
        return
    dir_path = os.path.split(fpath)[0]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(fpath, mode) as f:
        f.write("\n".join(feas) + "\n")
    print(f"Finished writing file: {fpath}")

def create_originalData_path(filename_or_path):
    return os.path.join(new_base_path, "originalData", filename_or_path)
def create_preprocessedData_path(filename_or_path):
    return os.path.join(new_base_path, "preprocessedData", filename_or_path)
def create_trained_models_path(filename_or_path):
    return os.path.join(new_base_path, "trained_models", filename_or_path)

def store_data_to_newbasepath_csv(df, filename, dirname = new_base_path, foldername = "preprocessedData", index=False):
    print("data shape:", df.shape)
    fmt = "csv"
    valid_format = ["csv", "parquet"]
    assert fmt in valid_format, f"invalid format {fmt}, should be {valid_format}"
    file_path = os.path.join(dirname, foldername, filename + f".{fmt}")
    cmd = f'df.to_{fmt}("{file_path}", index={index})'
    print(cmd)
    eval(cmd)
    print("data saved.")
    return file_path

def kill_current_kernel():
    '''杀死当前的kernel释放内存空间。'''
    IPython.Application.instance().kernel.do_shutdown(True)

def load_data_with_head_and_content(
    head_file_path, 
    content_file_path, 
):
    ## 甭管之前有没有分析过特征名称，总之这里重新加载一次head文件。
    raw_data = pd.read_table(
        head_file_path,
        names=['var_names', 'data_type'], 
        usecols=['var_names', 'data_type']
    )
    feats_names = [var.strip() for var in raw_data.var_names.to_list()]
    raw_data = pd.read_table(
        content_file_path,
        names=feats_names, 
        usecols=feats_names
    )
    return raw_data

def load_data_from_preprocessedData_csv(filename, dirname = new_base_path, foldername = "preprocessedData", use_cols = None, encoding = "utf-8"):
    fmt = "csv"
    valid_format = ["csv", "parquet"]
    assert fmt in valid_format, f"invalid format {fmt}, should be {valid_format}"
    file_path = os.path.join(dirname, foldername, filename + f".{fmt}")
    if fmt == "csv":
        if use_cols is None:
            cmd = f'pd.read_{fmt}("{file_path}", quoting=3, encoding="{encoding}", lineterminator="\\n")'
        else:
            cmd = f'pd.read_{fmt}("{file_path}", usecols = {use_cols}, quoting=3, encoding="{encoding}", lineterminator="\\n")'
    else:
        if use_cols is None:
            cmd = f'pd.read_{fmt}("{file_path}", encoding="{encoding}")'
        else:
            cmd = f'pd.read_{fmt}("{file_path}", columns={use_cols}, encoding="{encoding}")'
    print(cmd)
    df = eval(cmd)
    print("data shape:", df.shape)
    return df

from datetime import datetime
class TimerContext:  
    def __enter__(self):  
        self.start_time = str(datetime.now())
        print("start time:", self.start_time)
        return self  
    def __exit__(self, exc_type, exc_val, exc_tb):  
        print("start time:", self.start_time)
        print("end time", str(datetime.now()))

from matplotlib.font_manager import FontProperties
myfont = FontProperties(fname = "/home/snbmod/WORKSPACE/24100129_xmk/SimHei.ttf")

import warnings
warnings.filterwarnings('ignore')
import stepwise

from sklearn.model_selection import GridSearchCV,ParameterGrid,StratifiedKFold,train_test_split,GroupKFold
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, ElasticNet, ElasticNetCV, LogisticRegressionCV

import pickle