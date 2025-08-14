
import pandas as pd
import numpy as np


def cal_ks(tpr, fpr, thresholds):
    KS_max=0
    best_thr=0
    for i in range(len(tpr)):
        if(i==0):
            KS_max = tpr[i] - fpr[i]
            thr = thresholds[i]
        elif (tpr[i] - fpr[i] > KS_max):
            KS_max = tpr[i] - fpr[i]
            best_thr = thresholds[i]
    return KS_max, best_thr



def ks_weight(y_pred, y_true, weight = None):
    if weight == None:
        weight = np.ones(len(y_true))
    table = pd.crosstab(y_pred*weight, y_true*weight).cumsum()/pd.crosstab(1, y_true*weight).iloc[0]
    return abs(table[0]-table[1]).max()


def cal_woe_iv(dataset):
    
    dataset['Total'] = dataset.Good + dataset.Bad
    dataset['%Good'] = dataset.Good/ dataset.Good.sum()
    dataset['%Bad'] = dataset.Bad/ dataset.Bad.sum()
    dataset['%Total'] = dataset.Total/ dataset.Total.sum()  
    dataset['Bad_Rate'] = dataset.Bad/ dataset.Total
    dataset['WOE'] = np.log( dataset['%Good'] / dataset['%Bad'] )
    dataset['IV'] = dataset['WOE'] *(dataset['%Good'] - dataset['%Bad']  )
    dataset['Odds'] = dataset.Good / dataset.Bad
    return dataset


def cal_score(proba, scale_method='min-max', base_score=750, base_odd=20, pdo=60, _min_=0.0001, _max_=0.9999):
    if scale_method == 'pdo-odds':
        factor = score_step / np.log(2)
        offset = base_score - factor * np.log(base_odd)
        score = np.round(offset - factor * np.log(proba /(1 - proba)))
    elif scale_method == 'min-max':
        score = np.round(300 + 600* (proba-_min_) / (_max_ - _min_))

    if score >= 900:
        score = 900
    elif score <= 300:
        score = 300
    
    return score


def get_oofset_factor(base_score=750, base_odd=20, score_step=60):
    factor = score_step / np.log(2)
    offset = base_score - factor * np.log(base_odd)
    return factor, offset


def cal_psi(actual, expect, bins=10):
    """
    功能: 计算PSI值，并输出实际和预期占比分布
    :param actual: Array或series，代表真实数据，如测试集模型得分 # test
    :param expect: Array或series，代表期望数据，如训练集模型得分 # base
    :param bins: 分段数
    :return:
        psi: float，PSI值
        psi_df:DataFrame

    """
    #分箱
    expect_min = expect.min()  # 实际中的最小值
    expect_max = expect.max()  # 实际中的最大值
    binlen = (expect_max - expect_min) / bins  #箱体宽度
    cuts = [expect_min + i * binlen for i in range(1, bins)] #设定切点
    cuts.insert(0, -float("inf"))  #在切点左侧加入-float("inf")扩展左边界
    cuts.append(float("inf"))  #在切点右侧加入float("inf")扩展右边界
    #箱内计数，合并为一个数据框
    expect_cuts = np.histogram(expect, bins=cuts)[0]#将expect等宽分箱并计数
    actual_cuts = np.histogram(actual, bins=cuts)[0]#将actual按expect的分组等宽分箱并计数
    actual_df = pd.DataFrame(actual_cuts,columns=['actual'])
    expect_df = pd.DataFrame(expect_cuts,columns=['expect'])
    psi_df = pd.merge(expect_df,actual_df,right_index=True,left_index=True)
    #计算箱内频数
    psi_df['actual_rate'] = (psi_df['actual'] + 1) / psi_df['actual'].sum()#计算占比，分子加1，防止计算PSI时分子分母为0
    psi_df['expect_rate'] = (psi_df['expect'] + 1) / psi_df['expect'].sum()
    #计算每个箱内的数值
    psi_df['psi'] = (psi_df['actual_rate'] - psi_df['expect_rate']) * np.log(psi_df['actual_rate'] / psi_df['expect_rate'])
    #得到PSI
    psi = psi_df['psi'].sum()
    return psi, psi_df



def cal_score_rank_distribution(df, score_col, label_col, n_groups=10, weight_col=None, asc=True,
                                bin_str=None, cut_by_bad=False):
    '''
    :param df:
    :param score_col:
    :param label_col: y variable
    :param n_groups:
    :param weight_col:
    :param asc: the default cum order is from low score to high score
    :param bin_str: given bin edges
    :param cut_by_bad: if True cut score by number of bad, default False
    :return: pandas data frame
    '''
    df_new = df[[score_col, label_col]].copy()
    if weight_col is None:
        df_new['weight'] = 1
    else:
        df_new['weight'] = df[weight_col]
    df_new['range'], _ = snbin.analysis_var_numeric(df[score_col], df[label_col], score_col, n_groups,
                                                    sample_weight=df_new['weight'], bin_str=bin_str,
                                                    cut_by_bad=cut_by_bad)
    df_new['weighted_y'] = df_new[label_col] * df_new['weight']

    # non-cum values
    df_result = df_new.groupby('range').agg({'weight': np.sum, 'weighted_y': np.sum}) \
        .rename(columns={'weight': '#total', 'weighted_y': '#bad'})

    df_result['max'] = df_new.groupby('range').agg({score_col: np.max})
    df_result['min'] = df_new.groupby('range').agg({score_col: np.min})
    df_result['range'] = '(' + (df_result['min'] - 1).astype(int).astype(str) + ',' + \
                         df_result['max'].astype(int).astype(str) + ']'

    df_result['#good'] = df_result['#total'] - df_result['#bad']
    df_result['per_good'] = round(100 * df_result['#good'] / df_result['#good'].sum(), 2).astype(str) + '%'
    df_result['per_bad'] = round(100 * df_result['#bad'] / df_result['#bad'].sum(), 2).astype(str) + '%'
    df_result['per_total'] = round(100 * df_result['#total'] / df_result['#total'].sum(), 2).astype(str) + '%'
    df_result['bad_rate'] = round(100 * df_result['#bad'] / df_result['#total'], 2).astype(str) + '%'

    # cal index for sorting
    df_result['index'] = df_new.groupby('range').agg({score_col: np.max}).rename(columns={score_col: 'index'})
    if not asc:
        df_result.sort_values(by='index', inplace=True, ascending=False)
    # cum values
    df_result['#cum_total'] = df_result['#total'].cumsum().astype(int)
    df_result['#cum_bad'] = df_result['#bad'].cumsum().astype(int)
    df_result['#cum_good'] = df_result['#good'].cumsum().astype(int)

    df_result['per_cum_bad'] = round(100 * df_result['#cum_bad'] / df_result['#bad'].sum(), 2).astype(str) + '%'
    df_result['per_cum_good'] = round(100 * df_result['#cum_good'] / df_result['#good'].sum(), 2).astype(str) + '%'
    df_result['per_cum_total'] = round(100 * df_result['#cum_total'] / df_result['#total'].sum(), 2).astype(str) + '%'
    df_result['cum_bad_rate'] = round(100 * df_result['#cum_bad'] / df_result['#cum_total'], 2).astype(str) + '%'
    # round counting values
    df_result['#total'] = df_result['#total'].astype(int)
    df_result['#good'] = df_result['#good'].astype(int)
    df_result['#bad'] = df_result['#bad'].astype(int)

    return_col = ['range', '#cum_good', '#cum_bad', '#cum_total', 'per_cum_good', 'per_cum_bad', 'per_cum_total',
                  'cum_bad_rate', '#good', '#bad', '#total', 'per_good', 'per_bad', 'per_total', 'bad_rate']
    # re-sort
    if not asc:
        df_result.sort_values(by='index', inplace=True)
    return df_result[return_col]
