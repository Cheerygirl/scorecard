# encoding=utf 8

import pandas as pd
import math
import MySQLdb
import numpy as np


#x为dataframe或其series，返回分箱的list，list中的成员为各个分箱的dataframe（里面存了分箱中的x元素）
def create_bins_uniform(col_name, df):
    if (df[col_name].dtype == np.int64)|(df[col_name].dtype == np.float):
       bins_num = raw_input('Please enter bins number:')
       bins_num = float(bins_num)
       df_missing = df[pd.isnull(df[col_name])]
       df_values = df[~pd.isnull(df[col_name])]

       bins_list = []
       bins_list.append(df_missing)

       xlst = df_values[col_name].tolist()
       max_x = float(max(xlst))
       min_x = float(min(xlst))

       range_x = max_x - min_x
       print max_x,min_x,range_x
# 求分箱的节点
       i = 0
       point = []
       while i < bins_num:
           point.append(min_x + range_x / bins_num * i )
           i += 1
       point.append(max_x)
# 求每个分箱中的dataframe，并返回包含几个分箱的列表
       i = 0
       while i < bins_num:
           df_bin = df_values[(df_values[col_name].astype(float) >= point[i]) & (df_values[col_name].astype(float) < point[i + 1])]
           bins_list.append(df_bin)
           i += 1
    else:
        df_missing = df[pd.isnull(df[col_name])]
        df_values = df[~pd.isnull(df[col_name])]

        bins_list = []
        bins_list.append(df_missing)

        x_list = df_values[col_name].tolist()
        x_list = list(set(x_list)) #去重
        bins_num = x_list.count()
        i = 0
        while i<bins_num:
            df_bin = df_values[df_values[col_name] == x_list[i]]
            bins_list.append(df_bin)
            i+=1

    return bins_list

def create_bins_percentile(col_name, df):
    uni_var = 'Null'
    if (df[col_name].dtype == np.int64) | (df[col_name].dtype == np.float):
        #bins_num = raw_input('Please enter bins number:')
        #bins_num = float(bins_num)
        bins_num = 10.0
        df_missing = df[pd.isnull(df[col_name])]
        df_values = df[~pd.isnull(df[col_name])]

        bins_list = []
        bins_list.append(df_missing)
        # 求分箱的节点
        j = 0
        point = []
        while j <= bins_num:
            point.append(df_values[col_name].quantile(j/bins_num))
            j+=1
        # 求每个分箱中的dataframe，并返回包含几个分箱的列表
        i = 0
        while i < bins_num-1:
            df_bin = df_values[(df_values[col_name].astype(float) >= point[i]) & (
            df_values[col_name].astype(float) < point[i + 1])]
            bins_list.append(df_bin)
            i += 1
        df_bin = df_values[(df_values[col_name].astype(float) >= point[9]) & (
            df_values[col_name].astype(float) <= point[10])]
        bins_list.append(df_bin)

        if df_values[col_name].quantile(0.25) ==df_values[col_name].quantile(1):
            uni_var = col_name
    else:
        df_missing = df[pd.isnull(df[col_name])]
        df_values = df[~pd.isnull(df[col_name])]

        bins_list = []
        bins_list.append(df_missing)

        x_list = df_values[col_name].tolist()
        x_list = list(set(x_list))  # 去重

        bins_num = len(x_list)
        i = 0
        while i < bins_num:
            df_bin = df_values[df_values[col_name] == x_list[i]]
            bins_list.append(df_bin)
            i += 1
        point = bins_list
    #print col_name,bins_list
    return bins_list,col_name,point, uni_var




#转换Y为0/1

def Y_trans(df_Y):
    i = 0
    while i < df_Y.count()[1]:
        if df_Y.ix[i,1] > 30:
            df_Y.ix[i,1] = 1
        else:
            df_Y.ix[i,1] = 0
        i+=1
    return df_Y



def cal_info_value(Y, bins_list,col_name, point):
    i = 0
    goods = []
    bads = []
    goods_bads = []
    info_odds = []
    bad_rate = []
    good_rate = []
    woe = []
    info_value = 0
    info_value_list = []
    bins_num = len(bins_list)
    goods_total_num = 0
    bads_total_num = 0
    for i in range(0,bins_num):
        df_bin =bins_list[i]
        goods.append(float(df_bin[df_bin[Y] == 1][Y].count()))
        bads.append(float(df_bin[df_bin[Y] == 0][Y].count()))
        goods_bads.append(goods[i]+bads[i])
        goods_total_num = goods_total_num + goods[i]
        bads_total_num = bads_total_num + bads[i]
    #print goods_total_num, bads_total_num
    for i in range (0,bins_num):
        if(goods_total_num== 0 or bads_total_num ==0):
            bad_rate.append(0.0)
            good_rate.append(0.0)
        else:
            bad_rate.append(float(bads[i]) / float(bads_total_num))
            good_rate.append(float(goods[i]) / float(goods_total_num))
        if (bad_rate[i] == 0 or good_rate[i] == 0):
            info_odds.append(0.0)
            woe.append(0.0)
        else:
            info_odds.append(good_rate[i] / bad_rate[i])
            woe.append(math.log(info_odds[i]))
        info_value = float(info_value + (good_rate[i] - bad_rate[i]) * woe[i])
    info_value_list.append(info_value)

    #df_GB = pd.DataFrame({'Category':col_name[0:3],'Var':col_name,'bins list':bins_list,'goods number':goods, 'bads number':bads, 'good rate':good_rate,'bad rate':bad_rate,'info odds':info_odds,'WOE':woe,'point':point, 'good + bad no':goods_bads})
    df_IV = pd.DataFrame({'Category':col_name[0:3],'Var':col_name,'IV':info_value_list})
    df_GB = pd.DataFrame(
        {'Category': col_name[0:3], 'Var': col_name, 'goods number': goods, 'bads number': bads,
         'good rate': good_rate, 'bad rate': bad_rate, 'info odds': info_odds, 'WOE': woe, 'point': point,
         'good bad no': goods_bads})
    return df_IV,df_GB

'''
df=pd.read_excel('C:/JBW/JBW_c/PycharmProjects/untitled/ans_y_45_0518_equally-interval/bin_pars_selected.xlsx')
column.append('overdue_days')
column.append('gb_flag')
flag_df=scd3.Set_GB(final_data_df,'overdue_days',45)
flag_df=flag_df[column]
col_name_list=[flag_df.columns.tolist()[i] for i in range(len(flag_df.columns.tolist()))]
col_name_list.remove('overdue_days')
col_name_list.remove('gb_flag')

List=[]
Item=[]
for item in col_name_list:
    flag_df_notna=flag_df[~pd.isnull(flag_df[item])]
    if  df[df['var_name']==item]['val_range_adjusted'].count()==0:
        if (flag_df_notna[item].dtype == np.int64) or (flag_df_notna[item].dtype == np.float):
            i_max=max(flag_df_notna[item].tolist())
            i_min=min(flag_df_notna[item].tolist())
            bins=[(i_min,i_max)]
            bins.append('missing')
            List.append(bins)
            Item.append(item)
            print item + '0'
            
            '''
'''
 i_max = max(flag_df_notna[item].tolist())
            i_min = min(flag_df_notna[item].tolist())
            percent = [flag_df_notna[item].quantile(0.1 * i) for i in range(1, 10)]
            bins_start = [i_min] + percent
            bins_end = percent + [i_max]
            bins = zip(bins_start, bins_end)
            bins.append('missing')
            List.append(bins)
            Item.append(item)
            print item +'0' 
'''
'''
else:
            i_unique_list = np.unique(flag_df_notna[item]).tolist()
            i_str_list = [str(i_unique_list[i]) for i in range(len(i_unique_list))]
            bins = i_str_list
            bins.append('missing')
            List.append(bins)
            Item.append(item)
            print item +'1'


    elif df[df['var_name']==item]['val_range_adjusted'].dtype == np.float:
        #type(df[df['var_name']==item]['val_range_adjusted'][0])==float:
        i_max=max(flag_df_notna[item].tolist())
        i_min=min(flag_df_notna[item].tolist())
        i_range = df[['var_name', 'val_range_adjusted']]
        cond1 = i_range['var_name'] == item
'''
