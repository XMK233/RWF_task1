import os
proj_path = os.sep.join(os.getcwd().split(os.sep)[:6])

import sys
sys.path.append(proj_path)
from kaitoupao2 import *
from matplotlib.font_manager import FontProperties
fnt = FontProperties(fname=os.path.join(proj_path, "SimHei.ttf"))

import pandas as pd
import numpy as np

from collections import defaultdict
tqdm.tqdm.pandas()

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
            
def get_score_diff_2_tmp1(df1, df2, fea_list, epsilon = 1e-8, left_symb = "dz", right_symb = "dq", prefix = "", key_col = "report_sn"): 
    '''
    这个是看两份特征之间的差异的。
    '''
    dz, dq = left_symb, right_symb
    def do_something_1(row):
        '''
        获得一个字典，这个字典的key是特征，value是列表。
        列表里面放着的是啥呢？是有差异的trace以及特征值之类的信息。
        '''
        feas_traces = defaultdict(list)
        try: 
            for fea in target_feas:
                v1, v2 = row[f"{fea}_x"], row[f"{fea}_y"]
                
                if (pd.isna(v1) and not pd.isna(v2)) or (not pd.isna(v1) and pd.isna(v2)):
                    ## 一边为空另一边不为空
                    pass
                else:
                    ## 两边都是空或者两个都不是空的情况的判断。有的情况是相等的，有的情况是不等的。
                    if (v1 is None) and (v2 is None):## 两者为None，那就是都一样的。
                        continue
                    if pd.isna(v1) and pd.isna(v2):## 两者为nan，那就是一样的。
                        continue
                    if v1 == v2:## 值一样，那自然也是一样的。
                        continue
                    if abs(v1 - v2) <= epsilon:## 差距太小，那自然也是一样的。
                        continue
                
                diff_record = {
                    "traceid": row[key_col],
                    f"val_{dz}": v1, 
                    f"val_{dq}": v2,
                }
                feas_traces[fea].append(diff_record)
        except Exception as e:
            print(e)
            print(dict(row))
            print(fea, v1, v2)
            pass
        return feas_traces
    
    dds = pd.merge(
        df1, df2, on=key_col, how="inner"
    )
    target_feas = fea_list
    print(dds.shape)
    # print(target_feas)
    
    # from pandarallel import pandarallel
    # pandarallel.initialize(nb_workers = 8, progress_bar = True)
    df_feas = dds.progress_apply(lambda row: do_something_1(row), axis = 1)
    feas_traces_dds = defaultdict(list)
    for rst in df_feas:
        for key in rst:
            feas_traces_dds[key].extend(rst[key])
    ## 把所有的diff的情况给它打出来。        
    with open(f"{prefix}feas_diffVals.txt", "w") as f:
        json.dump(feas_traces_dds, f, cls=NpEncoder) # indent=4,
    ## 
    traceid_feaList = defaultdict(dict)
    for fea in feas_traces_dds:
        l = feas_traces_dds[fea]
        for item in l:
            tid = item["traceid"]

            traceid_feaList[tid]["traceid"] = item["traceid"]

            if "feas" not in traceid_feaList[tid]:
                traceid_feaList[tid]["feas"] = {}
            traceid_feaList[tid]["feas"][fea] = {
                f"val_{dz}": item[f"val_{dz}"], 
                f"val_{dq}": item[f"val_{dq}"]
            }
    with open(f"{prefix}traceid_feaVal.txt", "w") as f:
        json.dump(traceid_feaList, f, cls=NpEncoder) # indent=4,        
    ## 按照等级对特征进行归类。
    s_l = sorted(list(feas_traces_dds.keys()))
    dic = {}
    for i in s_l:
        parts = i.split("___")
        if parts[0] not in dic:
            dic[parts[0]] = {}
        cur_dic = dic[parts[0]]
        for part in parts[1:-1]:
            if part not in cur_dic:
                cur_dic[part] = {}
            cur_dic = cur_dic[part]
        cur_dic[parts[-1]] = i
    with open(f"{prefix}cascaded_fea_names.txt", "w") as f:
        json.dump(dic, f)

    return feas_traces_dds

def get_intersection(itr1, itr2):
    return set(itr1).intersection(set(itr2))

def get_left_unique(itr1, itr2): ## right unique 便不写了罢。
    return set(itr1) - set(itr2)

def easy_plot(df_tmp, sup_title, target_label, bad_rate_col = "bad_rate"):
    tt = df_tmp.iloc[:, :-1].reset_index()
    # display(tt)
    
    tt_all = df_tmp["All"]
    
    fig = plt.figure(figsize=(12,5))
    fig.suptitle(sup_title, font=fnt)#
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    ax1.set_ylabel("样本数量", font=fnt)#
    ax2.set_ylabel("1样本占比", font=fnt)#
    ax1.bar(
        tt.columns[1:], 
        tt[tt[target_label]==-1][list(tt.columns[1:])].T[0], 
        color = "r", 
        label =  f"-1"
    )
    ax1.bar(
        tt.columns[1:], 
        tt[tt[target_label]==0][list(tt.columns[1:])].T[1], 
        bottom = tt[tt[target_label]==-1][list(tt.columns[1:])].T[0],
        color = "g", 
        label =  f"0"
    )
    ax1.bar(
        tt.columns[1:], 
        tt[tt[target_label]==1][list(tt.columns[1:])].T[2], 
        bottom = tt[tt[target_label]==-1][list(tt.columns[1:])].T[0] + tt[tt[target_label]==0][list(tt.columns[1:])].T[1],
        color = "b", 
        label =  f"1"
    )
    # ax1.bar(tt.columns, tt.loc[0, :], bottom =tt.loc[-1, :], color="g", label=f"0, total: {tt_all[0]}")
    # ax1.bar(tt.columns, tt.loc[1, :], bottom =tt.loc[-1, :]+tt.loc[0, :], color="b", label= f"1, total: {tt_all[1]}")
    # display(
    #     tt[tt[target_label]==bad_rate_col][list(tt.columns[1:])].T
    # )
    ax2.plot(
        tt.columns[1:], 
        tt[tt[target_label]==bad_rate_col][list(tt.columns[1:])].T[4], 
        color="y"
    )
    
    ax1.legend(prop=fnt, loc="upper left")
    ax2.legend(prop=fnt)
    plt.ylim(0, None)
    ax1.tick_params(axis="x", rotation=90)
    # plt.xticks(rotation=90)
    plt.show()

def show_label_dist(df, target_label):
    raw_dt = df
    temp = pd.DataFrame(
        pd.crosstab(raw_dt.apply_month, raw_dt[target_label], margins=True)
    )
    # display(temp)
    ## 如果缺少了某些标签，就用0去补足。
    if -1 not in temp.columns:
        temp[-1] = 0
    if 0 not in temp.columns:
        temp[0] = 0
    if 1 not in temp.columns:
        temp[1] = 0
    temp = temp[[-1, 0, 1, "All"]]
    # display(temp)
    bad_rate_col = "bad_rate (1/(0,1,-1))"
    temp[bad_rate_col] = temp[1]/(temp[0]+temp[1]+temp[-1])
    easy_plot(temp.T, f"{target_label} 样本", target_label, bad_rate_col)
    
    temp[bad_rate_col] = temp[bad_rate_col].apply(lambda x: "{:.2%}".format(x))
    display(temp.T)