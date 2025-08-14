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

import random, os, tqdm, time, json, re, IPython

random.seed(618)
np.random.seed(907)
tqdm.tqdm.pandas() ## 引入这个，就可以在apply的时候用progress_apply了。

import os
proj_path = os.sep.join(os.getcwd().split(os.sep)[:6]) ## 这个在mac下，是6层。别的项目可能不一样。

import utils
new_base_path = os.getcwd().replace(
    os.path.join(proj_path, "code"),
    os.path.join(proj_path, "data").replace("Documents", "Downloads"),
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
def store_data_to_newbasepath(df, filename, dirname = new_base_path, fmt = "parquet", foldername = "preprocessedData", index=False):
    print("data shape:", df.shape)
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

def load_data_from_preprocessedData_csv(filename, dirname = new_base_path, foldername = "preprocessedData", use_cols = None, encoding = "utf-8", sep=","):
    fmt = "csv"
    valid_format = ["csv", "parquet"]
    assert fmt in valid_format, f"invalid format {fmt}, should be {valid_format}"
    file_path = os.path.join(dirname, foldername, filename + f".{fmt}")
    if fmt == "csv":
        if use_cols is None:
            cmd = f'pd.read_{fmt}("{file_path}", encoding="{encoding}", sep="{sep}")' ## quoting=3, lineterminator="\\n"
        else:
            cmd = f'pd.read_{fmt}("{file_path}", usecols = {use_cols}, encoding="{encoding}", sep="{sep}")' ## quoting=3, lineterminator="\\n"
    else:
        if use_cols is None:
            cmd = f'pd.read_{fmt}("{file_path}", encoding="{encoding}")'
        else:
            cmd = f'pd.read_{fmt}("{file_path}", columns={use_cols}, encoding="{encoding}")'
    print(cmd)
    df = eval(cmd)
    print("data shape:", df.shape)
    return df

def load_data_from_preprocessedData(filename, dirname = new_base_path, fmt = "parquet", foldername = "preprocessedData", use_cols = None, encoding = "utf-8"):
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
            cmd = f'pd.read_{fmt}("{file_path}")' ## , encoding="{encoding}"
        else:
            cmd = f'pd.read_{fmt}("{file_path}", columns={use_cols})' ## , encoding="{encoding}"
    print(cmd)
    df = eval(cmd)
    print("data shape:", df.shape)
    return df

def load_data_from_originalData_tab(filename, dirname = new_base_path, foldername = "originalData", use_cols = None, encoding = "utf-8", sep="\t", nrows=None):
    fmt = "csv"
    valid_format = ["csv", "parquet"]
    assert fmt in valid_format, f"invalid format {fmt}, should be {valid_format}"
    file_path = os.path.join(dirname, foldername, filename + f".{fmt}")
    if fmt == "csv":
        if use_cols is None:
            cmd = f'pd.read_{fmt}("{file_path}", encoding="{encoding}", sep="{sep}", nrows={nrows})' ## quoting=3, lineterminator="\\n"
        else:
            cmd = f'pd.read_{fmt}("{file_path}", usecols = {use_cols}, encoding="{encoding}", sep="{sep}", nrows={nrows})' ## quoting=3, lineterminator="\\n"
    else:
        if use_cols is None:
            cmd = f'pd.read_{fmt}("{file_path}", encoding="{encoding}", nrows={nrows})'
        else:
            cmd = f'pd.read_{fmt}("{file_path}", columns={use_cols}, encoding="{encoding}", nrows={nrows})'
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

def millisec2datetime(timestamp):
    time_local = time.localtime(timestamp/1000)
    return time.strftime("%Y-%m-%d %H:%M:%S", time_local)

def run_finish():
    # 假设你的字体文件是 'myfont.ttf' 并且位于当前目录下  
    font = FontProperties(fname="/home/snbmod/WORKSPACE/24100129_xmk/SimHei.ttf", size=24)  
    # 创建一个空白的图形  
    fig, ax = plt.subplots()  
    # 在图形中添加文字  
    ax.text(0.5, 0.5, f"程序于这个点跑完：\n{millisec2datetime(time.time()*1000)}", fontproperties=font, ha="center", va="center")  
    # 设置图形的布局  
    ax.set_xlim(0, 1)  
    ax.set_ylim(0, 1)  
    ax.set_xticks([])  
    ax.set_yticks([])  
    # 显示图形  
    ax.patch.set_color("green")
    plt.show()

def finetune_2_params_lgbm_smallData_1(
    all_trnVld_matrix, 
    model_dir_this_round, 
    model_name, 
    params, ## 基础参数。
    first_param = "max_depth",
    first_range = [3, 5, ], 
    first_type = "int",
    second_param = "min_child_weight",
    second_range = [100, 1000, ],
    second_type = "int",
    record_file_identifier = "",
    early_stopping_rounds = 200, 
    nfold = 5,
    folds = None, 
    categorical_feature = "auto",
    extra_feval = None
):
    '''
    这个函数是固定调两个参数的，一定要调2个参数。多了少了都不行。
    调first_param和second_param俩参数，最终返回修改过后的参数组合以及最佳树数。
    record_file_identifier 这个参数是用来单独标记模型表现记录文件的。
    
    这个仅限于小规模样本的调参。因为我们用了lgb.cv来获得某种参数的表现。
    如果是大规模数据调参，这种方法效率堪忧。
    '''
    
    ## 原版params里面的相应参数，如果不在调参列表里，也要加进去，作为base。
    if not (params[first_param] in first_range):
        first_range = [params[first_param]] + first_range
    if not (params[second_param] in second_range):
        second_range = [params[second_param]] + second_range
    
    
    ## 存模型表现的文件的位置。
    record_file = os.path.join(
        model_dir_this_round, model_name
    ) + f"-finetuning-{first_param}-{second_param}{record_file_identifier}.csv"
    ## 如果这个文件不存在，就硬造一个空的文件。
    if not os.path.exists(record_file):
        pd.DataFrame(
            {
                "param_setting": [],
                "train_auc": [],
                "best_score": [],
                "best_iteration": []
            }

        ).to_csv(
            record_file, index=False
        )
        
    ## 初始化一个新的参数列表。
    params_thisRound = {key:params[key] for key in params}
    # params_thisRound["verbosity"] = 1
    
    ## 调两个参数。
    for v1_ in first_range: 
        for v2_ in second_range: 

            v1 = eval(f"{first_type}({v1_})")
            if second_type == "str":
                v2 = eval(f"{second_type}('{v2_}')")
            else:
                v2 = eval(f"{second_type}({v2_})")
            
            params_thisRound[first_param] = v1
            params_thisRound[second_param] = v2

            records = pd.read_csv(record_file)
            param_setting = f"{first_param}__{v1}-{second_param}__{v2}"
            if param_setting in records.param_setting.to_list():
                display(records[records.param_setting == param_setting])
            else:
                print(f"{param_setting} is now training...".center(100, "="))
                if folds is None:
                    res = lgb.cv(
                        params_thisRound, all_trnVld_matrix, nfold=5,
                        metrics = "auc",
                        early_stopping_rounds = early_stopping_rounds,
                        # verbose_eval = 500,
                        eval_train_metric = True,
                        categorical_feature = categorical_feature,
                        feval = extra_feval
                    )
                elif folds is not None:
                    res = lgb.cv(
                        params_thisRound, all_trnVld_matrix, folds=folds,
                        metrics = "auc",
                        early_stopping_rounds = early_stopping_rounds,
                        # verbose_eval = 500,
                        eval_train_metric = True,
                        categorical_feature = categorical_feature,
                        feval = extra_feval
                    )
                else:
                    raise Exception("fold setting is strange.")
                    
                for idxx, (ta, va) in enumerate(zip(res["train auc-mean"], res["valid auc-mean"])):
                    if ((idxx + 1)%20) == 0:
                        print(f"[{idxx + 1}] train auc-mean: {ta}, valid auc-mean: {va}")
                print("the last perf: ")
                print("[{}] train auc-mean: {}, valid auc-mean: {}".format((idxx + 1), res["train auc-mean"][-1], res["valid auc-mean"][-1]))
                records = pd.concat([records, pd.DataFrame({
                        "param_setting": [param_setting], 
                        "train_auc": [res["train auc-mean"][-1]],
                        "best_score": [res["valid auc-mean"][-1]], # md.best_score["eval"]["auc"], 
                        "best_iteration": [len(res["valid auc-mean"])],
                    })])
                records.to_csv(record_file, index=False)
            print() 
    ## 寻找最佳参数：
    records = pd.read_csv(record_file)
    best_iteration = records.iloc[records.best_score.idxmax(), :]["best_iteration"]
    best_params = records.iloc[records.best_score.idxmax(), :]["param_setting"]
    best_pm_dict = {pt.split("__")[0]: pt.split("__")[-1] for pt in best_params.split("-")}
    params_thisRound[first_param] = eval(f"{first_type}({best_pm_dict[first_param]})")
    params_thisRound[second_param] = eval(f"{second_type}({best_pm_dict[second_param]})")
    return params_thisRound, best_iteration

def train_model_with_different_label_2_variousParam_nbr_lgbm(
    dtrain, dtest, saved_model_name, 
    do_valid = True, save_model_path = "/data/private/public_data/cpu1/minkexiu/DuoTouSubModels/trained_models", 
    params = None, nbr = 100, ## 这个就可以用来设置模型的树的数量
    early_stopping_rounds = 200, 
):
    #############################
    if params is None:
        params = {
            "boosting_type": 'gbdt',
            "objective": "binary",
            "boosting_type": 'gbdt',
            'scale_pos_weight': negNum_posNum_ratio,
            "metric": "auc",
            "device_type": "gpu",
            "learning_rate": 0.1,
            "n_estimators": 100000,

            ## 第一轮调参：
            "max_depth":3,
            "min_child_weight": 100,
            ## 第二轮调参：
            "colsample_bytree": 1,
            "subsample": 1,
            ## 第三轮调参：
            'reg_lambda': 7,
            'reg_alpha': 100,  
            
            'num_leaves': 4,
            "nthread": 25,
            "verbose": -1,
        }
    ############################
    params["verbosity"] = 1
    params["n_estimators"] = nbr
    print(params)
    # callbacks = [lgb.log_evaluation(period = 500), lgb.early_stopping(stopping_rounds = early_stopping_rounds)]
    train_start_time = time.time()
        
    vs = [dtrain, dtest] if do_valid else [dtrain, ]
    vn = ['train', 'eval'] if do_valid else ["train", ]
    # print(vs, vn, nbr)
    booster_maidian = lgb.train(
        params = params,
        train_set = dtrain,
        # num_boost_round = 10, ## 这个参数我设置了之后似乎不生效。
        valid_sets = vs,
        valid_names = vn,
        # callbacks = callbacks,
        verbose_eval=False
    )
        
    train_end_time = time.time()
    
    print(f"train time: {train_end_time - train_start_time}")
    
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    booster_maidian.save_model(os.path.join(save_model_path, saved_model_name))
    
    return booster_maidian

import pickle, joblib

def save_pickle_object(obj, path):
    model_path = path
    with open(model_path, "wb") as f:
        pickle.dump(obj, f, protocol = 2)
    print(model_path)

def load_pickle_object(path):
    return joblib.load(path)

def get_full_importance_noTable_lgbm(
    lgb_model_path,
    feas_cols_v2,
    sorted_by = "gain",
    ignore_no_impc_feas = True, ## 这个参数设置为True之后，会把sorted_by对应的重要性为0的特征从最后返回的iptc_df中去除。
):
    assert sorted_by in {"split", "gain"}, f'sorted_by must be in {{"split", "gain"}}'

    lgb_model = lgb.Booster(model_file = lgb_model_path)
    splits = lgb_model.feature_importance("split")
    gains = lgb_model.feature_importance("gain")

    iptc_df = pd.DataFrame({
        "feature": feas_cols_v2,
        "split": lgb_model.feature_importance("split"),
        "gain": lgb_model.feature_importance("gain"),
    })

    iptc_df = iptc_df.sort_values(by = [sorted_by,],ascending=False).reset_index(drop=True)
    return iptc_df if not ignore_no_impc_feas else iptc_df[iptc_df[sorted_by] > 0]

from IPython.display import display_html
from itertools import chain,cycle
def display_side_by_side(dfs, numSpaceIntervals = 3):
    html_str = ''
    for df in dfs:
        html_str += (df.to_html() + "&nbsp;" * numSpaceIntervals)
    display_html(
        html_str.replace('table','table style="display:inline"'),
        raw=True
    )

def get_intersection(itr1, itr2):
    return set(itr1).intersection(set(itr2))

def get_left_unique(itr1, itr2): ## right unique 便不写了罢。
    return set(itr1) - set(itr2)


########################################################################
import zhdate
def three_num_get_gua(a, b, c):
    '''梅花易数三数起卦，以取本、互、变。'''
    bagua = ["111", "110", "101", "100", "011", "010", "001", "000"]
    guatu = {
        "111": ("☰", "天", "乾金"), 
        "110": ("☱", "泽", "兑金"),
        "101": ("☲", "火", "离火"),
        "100": ("☳" , "雷", "震木"),
        "011": ("☴", "风", "巽木"),
        "010": ("☵", "水", "坎水"),
        "001": ("☶", "山", "艮土"),
        "000": ("☷", "地", "坤土"),
    }
    print(
        "先天八卦数:", ", ".join([f"{i}{guatu[j][-1][0]}"for i, j in zip(range(1,9), bagua)])
    )
    ## https://zhuanlan.zhihu.com/p/457104350
    gua_64 = "天天乾，天风姤，天山遁，天地否，风地观，山地剥，火地晋，火天大有，水水坎，水泽节，水雷屯，水火既济，泽火革，雷火丰，地火明夷，地水师，山山艮，山火贲，山天大畜，山泽损，火泽睽，天泽履，风泽中孚，风山渐，雷雷震，雷地豫，雷水解，雷风恒，地风升，水风井，泽风大过，泽雷随，风风巽，风天小畜，风火家人，风雷益，天雷无妄，火雷噬嗑，山雷顾，山风蛊，火火离，火山旅，火风鼎，火水未济，山水蒙，风水涣，天水松，天火同人，地地坤，地雷复，地泽临，地天泰，雷天大壮，泽天夬，水天需，水地比，泽泽兑，泽水困，泽地萃，泽山咸，水山蹇，地山谦，雷山小过，雷泽归妹"
    gua_64_dict = {x[:2]: x[2:]for x in gua_64.split("，")}
    
    shanggua_idx = 7 if (a % 8 == 0) else (a % 8 - 1)
    xiagua_idx = 7 if (b % 8 == 0) else (b % 8 - 1)
    bianyao_idx = 5 if (c % 6 == 0) else (c % 6 - 1)
    print(f"本卦上：{shanggua_idx+1} 本卦下：{xiagua_idx+1} 变爻：{bianyao_idx+1}", )
    bengua = bagua[xiagua_idx] + bagua[shanggua_idx]
    hugua = bengua[1:-1][:3] + bengua[1:-1][1:]
    biangua = list(bengua)
    biangua[bianyao_idx] = str(1 - int(biangua[bianyao_idx]))
    biangua = "".join(biangua)
    df = pd.DataFrame([[
        guatu[bengua[3:]][0]+guatu[bengua[3:]][2], guatu[hugua[3:]][0]+guatu[hugua[3:]][2], guatu[biangua[3:]][0]+guatu[biangua[3:]][2], 
    ],[
        guatu[bengua[:3]][0]+guatu[bengua[:3]][2], guatu[hugua[:3]][0]+guatu[hugua[:3]][2], guatu[biangua[:3]][0]+guatu[biangua[:3]][2], 
    ]], index=["上卦", "下卦"], columns = [
        guatu[bengua[3:]][1] + guatu[bengua[:3]][1] + gua_64_dict[guatu[bengua[3:]][1] + guatu[bengua[:3]][1]],
        guatu[hugua[3:]][1] + guatu[hugua[:3]][1] + gua_64_dict[guatu[hugua[3:]][1] + guatu[hugua[:3]][1]],
        guatu[biangua[3:]][1] + guatu[biangua[:3]][1] + gua_64_dict[guatu[biangua[3:]][1] + guatu[biangua[:3]][1]],
    ])
    display(df)
    return bengua, hugua, biangua
    
def easy_start_gua():
    """用公历的日、时、分来起卦。"""
    n1, n2, n3 = str(datetime.now())[8:10], str(datetime.now())[11:13], str(datetime.now())[14:16]
    print(n1, n2, n3)
    return three_num_get_gua(int(n1), int(n2), int(n3))
easy_start_gua()

def easy_start_gua_lunar():
    '''用农历的月、日、时辰来起卦。'''
    time_now = datetime.now()
    zh_date_str = str(zhdate.ZhDate.from_datetime(time_now))
    zh_date_str_1 = datetime.strftime(
        datetime(
            *[int(x) for x in re.findall("\d+", zh_date_str)]
        ),
        '%Y-%m-%d'
    )
    zh_hour = (time_now.hour + 1)//2%12+1
    zh_hour_dizhi = "子、丑、寅、卯、辰、巳、午、未、申、酉、戌、亥".split("、")[zh_hour-1]
    
    n1, n2, n3 = zh_date_str_1[5:7], zh_date_str_1[8:10], zh_hour
    print(n1, n2, n3, f"{zh_hour_dizhi}时")
    return three_num_get_gua(int(n1), int(n2), int(n3))
easy_start_gua_lunar()