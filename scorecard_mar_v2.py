# encoding=utf 8

import pandas as pd
import math
import MySQLdb
import F
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pylab as pl
import sys

import numpy as np
import openpyxl
import scipy
from scipy.stats import spearmanr
#from scipy import stats
import FF
import statsmodels.api as sm
from sklearn import metrics
from ggplot import *
import xlrd
#reload(sys)
#sys.setdefaultencoding("utf-8")

cnx = MySQLdb.connect(user='etl', passwd='etl@DCFTest#5234%',
                      host='application2.datacube.dcf.net',
                      charset='utf8')
# 创建df_x_table_list 和 df_xy_table_list，分别存储所有的x表和xyjoin起来的表
x_table = ['modelcentre.score_card_1','modelcentre.score_card_2','modelcentre.score_card_3','modelcentre.score_card_4','modelcentre.score_card_5','modelcentre.score_card_6',
           'modelcentre.score_card_7','modelcentre.score_card_8','modelcentre.score_card_9','modelcentre.score_card_10','modelcentre.score_card_11','modelcentre.score_card_12',
               'modelcentre.score_card_13','modelcentre.score_card_14','modelcentre.score_card_15','modelcentre.score_card_16']
df_Y = pd.read_sql('select * from modelcentre.score_card_y_value ', cnx)
F.Y_trans(df_Y)

df_x_table = pd.DataFrame()
df_x_table_list = []
df_xy_table_list = []
x_col_num = []
xy_col_num = []
for item in x_table:
    df_x_table = pd.read_sql('select * from ' + item, cnx)
    print 'x table',df_x_table.shape[1]
    x_col_num.append(df_x_table.shape[1])
    df_x_table_list.append(df_x_table)
    df_xy_table = pd.merge(df_x_table, df_Y, on='win_uuid',how='left')
    xy_col_num.append(df_xy_table.shape[1])
    print 'xy table', df_xy_table.shape[1]
    df_xy_table_list.append(df_xy_table)
print x_col_num, xy_col_num
#返回16个scorecard x table的列数与16个scorecard xy table的列数  以及每个xtable与xytable放进dateframe的list中

#以scorecard_1为例，创建x1-x86与Y组建的两列table


df_missing_count_total = pd.DataFrame()
for j in range(0,16):
    for i in range(7, x_col_num[j] - 1):
        df = df_x_table_list[j].ix[:,[0,i]]
        df_missing = df[pd.isnull(df.ix[:,1])]
        data = {'Var':df_missing.columns[1],'Missing count':len(df_missing)}
        df_missing_count = pd.DataFrame(data,index=[0])  #构造missing的Var和其missing的个数的DF
        df_missing_count_total = df_missing_count_total.append(df_missing_count)
print df_missing_count_total


df_final_GB = pd.DataFrame()
df_final_IV = pd.DataFrame()
uni_var_list = []
for j in range(0,16):
    df_intermediate_GB = pd.DataFrame()
    df_intermediate_IV = pd.DataFrame()
    uni_var_intermediate_list = []
    for i in range(7, x_col_num[j] - 1):
        df_single_x = df_xy_table_list[j].ix[:, [i, -2]]
        x_col_name = df_single_x.columns[0]
        uni_var = 0
        bins, column_name, point, uni_var= F.create_bins_percentile(x_col_name, df_single_x)
        IV, GB = F.cal_info_value('overdue_days', bins, column_name, point)
        df_intermediate_IV = df_intermediate_IV.append(IV)
        df_intermediate_GB = df_intermediate_GB.append(GB)
        if uni_var == column_name:
            uni_var_intermediate_list.append(uni_var)
    for item in uni_var_intermediate_list:
        uni_var_list.append(item)
    df_final_GB = df_final_GB.append(df_intermediate_GB)
    df_final_IV = df_final_IV.append(df_intermediate_IV)
print uni_var_list,len(uni_var_list)  #去除25分位等于100分位的变量
df_final_IV = pd.merge(df_final_IV,df_missing_count_total, on='Var',how='left' )
df_final_GB = pd.merge(df_final_GB,df_missing_count_total, on='Var',how='left' )
df_final_IV.sort_values(by=['IV'], ascending=False, inplace=True)
#df_final_GB.sort_values(by='Category', ascending=False, inplace=True)
df_final_IV.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\IV.xlsx', encoding='gbk')
#df_final_GB.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\GB.xlsx', encoding='gbk')

'''
check error
pub_GB = df_final_GB.loc[df_final_GB['Var'] == 'pub_11_cv',:]
pub_GB.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\pub_GB.xlsx', encoding='gbk')
'''


#df_final_GB.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\GB.xlsx', encoding='gbk')

#去掉所有关于逾期的指标


#去掉25分位与100分位相同的变量
a = df_final_IV['Var'].tolist()
for item in uni_var_list:
    a.remove(item)
df_final_IV_uni_remove = df_final_IV[df_final_IV.ix[:,2].isin(a)]


df_overdue = pd.read_excel('overdue_var.xlsx')
overdue_list = df_overdue.iloc[:,0].tolist()
print len(overdue_list)
b = df_final_IV_uni_remove['Var'].tolist()
for item in overdue_list:
    if item in b:
        b.remove(item)
df_final_IV_uni_remove = df_final_IV[df_final_IV.ix[:,2].isin(b)]

#choose only IV >= 0.2 and missing count < half
Category_list = list(set(df_final_IV_uni_remove['Category'].tolist()))
df_category = pd.DataFrame()
for item in Category_list:
    df_cat = df_final_IV_uni_remove[df_final_IV_uni_remove['Category'] == item]
    df_cat = df_cat[df_cat['IV']>0.2]
    df_cat = df_cat[df_cat['Missing count'] < 1563]
    df_category = df_category.append(df_cat)
df_category.sort_values(by=['IV'], ascending=False, inplace=True)
print len(df_category)



df_final_IV_filter = df_category
df_final_IV_filter_list = []
df_final_IV_filter_list =df_final_IV_filter.iloc[:,2].tolist()
#print df_final_IV_filter.head()
# combine all x table in one
df_all_x_table = df_x_table_list[0]
for i in range (0,15):
    df_all_x_table = pd.merge(df_all_x_table, df_x_table_list[i+1], on='win_uuid', how='left')
print df_all_x_table.shape
print df_all_x_table.describe()
'''
#check error
overdue_table = df_xy_table_list[0].ix[:, [0, -2]]
df_total_xy = pd.merge(df_all_x_table, overdue_table, on='win_uuid', how='left')
print df_total_xy.describe()
df_total_xy.T.drop_duplicates().T
print df_total_xy.describe()
df_total_xy.ix[:,['pub_11_cv','overdue_days']].describe()
bins, column_name, point, uni_var= F.create_bins_percentile('pub_11_cv', df_total_xy.ix[:,['pub_11_cv','overdue_days']])
IV, GB = F.cal_info_value('overdue_days', bins, column_name, point)
'''

# choose df from all_x where it's in x_filter
#df_all_x_filter_table = df_all_x_table[[df_final_IV_filter_list]]
df_all_x_filter_table = df_all_x_table.ix[:,df_final_IV_filter_list]
df_corr_spearman = df_all_x_filter_table.corr('spearman')
df_corr_spearman.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\corr_spearman.xlsx', encoding='gbk')
df_corr_p = pd.DataFrame(spearmanr(df_all_x_filter_table)[1])
df_corr_p.index = df_corr_spearman.index
df_corr_p.columns = df_corr_spearman.columns
df_corr_p.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\corr_p.xlsx', encoding='gbk')


# corr_threhold = raw_input("please input the corr_threhold:")
# corr_threhold  = float(corr_threhold)
corr_threhold  = 0.3
#不存在var,其所有的P值都小于0.05，即完全独立
index_list = df_corr_p.index
not_corr_list_1 = []
for i in range(0,len(df_corr_p)):
    if all(df_corr_p.ix[i,:]>0.05):
        #print 'good!!!'
        not_corr_list_1 =not_corr_list_1.append(index_list[i])
    #else:
        #print "N"
print not_corr_list_1
'''
index_list = df_corr_spearman.index
not_corr_list_2 = []
for i in range(0,len(df_corr_spearman)):
    if sum(abs(df_corr_spearman.ix[i, :]) > corr_threhold)<2:
        #print "Good!!!"
        not_corr_list_2.append(index_list[i])
    #else:
        #print "N"
print not_corr_list_2
#不存在var,其所有的相关系数<0.3 & P>0.05
'''


'''
a = set(pd.Series(df_corr_spearman.columns).tolist())
b = set(not_corr_list_2).union(set(not_corr_list_1))
c = list(a.difference(b))
df2 = pd.DataFrame()
for item in c:
    df1 = df_final_IV_filter[df_final_IV_filter['Var']==item]
    df2 = df2.append(df1)
df2.sort_values(by=['Category','IV'], ascending=False, inplace=True)
df2_list = df2['Var'].tolist()  #有问题需要调试的

df_temp_1st_spearman = df_corr_spearman.ix[a.difference(b), a.difference(b)]
'''

#丢掉相关性较大的变量
df_temp_1st_spearman = df_corr_spearman
df_corr_p_judge = df_corr_p
temp_lst = []
win_var_list = []
lose_var_list = []
i = df_temp_1st_spearman.columns[0]
while len(df_temp_1st_spearman) > 1:
    for j in range(0,len(df_temp_1st_spearman)):
        if abs(df_temp_1st_spearman.ix[i, j]) > corr_threhold and df_temp_1st_spearman.ix[i, j]<1 and df_corr_p_judge.ix[i,j]<0.05:
            temp_lst.append(i)
            #temp_lst.append(df_temp_1st_spearman.columns[j])
            break
        else:
            continue
    if j < len(df_temp_1st_spearman)-1:
        a = df_final_IV[df_final_IV['Var'] == i]['IV'].values[0]
        b = df_final_IV[df_final_IV['Var'] == df_temp_1st_spearman.columns[j]]['IV'].values[0]
        if a > b:
            win_var = i
            lose_var = df_temp_1st_spearman.columns[j]
            lose_var_list.append(df_temp_1st_spearman.columns[j])
        elif a == b:
            if df_final_IV[df_final_IV['Var'] == i]['Missing count'].values[0] < df_final_IV[df_final_IV['Var'] == df_temp_1st_spearman.columns[j]]['Missing count'].values[0]:
                win_var = i
                lose_var = df_temp_1st_spearman.columns[j]
                lose_var_list.append(df_temp_1st_spearman.columns[j])
            else:
                win_var = df_temp_1st_spearman.columns[j]
                lose_var = i
                lose_var_list.append(i)
        else:
            win_var = df_temp_1st_spearman.columns[j]
            lose_var = i
            lose_var_list.append(i)
        df_temp_1st_spearman = df_temp_1st_spearman.drop(lose_var)
        del df_temp_1st_spearman[lose_var]
        df_corr_p_judge = df_corr_p_judge.drop(lose_var)
        del df_corr_p_judge[lose_var]
        i = win_var
        #print j, win_var, lose_var,'great'
    else:
        win_var_list.append(i)
        df_temp_1st_spearman = df_temp_1st_spearman.drop(i)
        del df_temp_1st_spearman[i]
        df_corr_p_judge = df_corr_p_judge.drop(i)
        del df_corr_p_judge[i]
        i = df_temp_1st_spearman.columns[0]
        #print j, i,'bad'
lose_var_list.append(i)
print len(win_var_list),len(lose_var_list)
win_var_list.extend(not_corr_list_1)
#win_var_list.extend(not_corr_list_2)
win_var_list = list(set(win_var_list))
lose_var_list = list(set(lose_var_list))
print win_var_list
print lose_var_list
print len(win_var_list)
#a = pd.Series(win_var_list)
#b = pd.Series(lose_var_list)
#a.to_csv('C:/Users/admin/Desktop/Python Output/V2/a.csv', encoding='gbk')
#b.to_csv('C:/Users/admin/Desktop/Python Output/V2/b.csv', encoding='gbk')
df_win_var = df_all_x_table.ix[:,win_var_list]
df_win_var_spearman = df_win_var.corr('spearman')
df_win_var_spearman.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\win_var_spearman.xlsx', encoding='gbk')
df_corr_p_win = pd.DataFrame(spearmanr(df_win_var)[1])
df_corr_p_win.index = df_win_var_spearman.index
df_corr_p_win.columns = df_win_var_spearman.columns
df_corr_p_win.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\corr_p_win.xlsx', encoding='gbk')

df_temp_1st_spearman.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\df_temp_1st_spearman.xlsx', encoding='gbk')




# win list中所有变量按照10个分箱画图
for i in range (0,len(win_var_list)/20+1):
    for j in range(0, 20):
        plt.figure(i)
        axe = plt.subplot(4,5,j+1)
        ptdata_1 = df_final_GB[df_final_GB.ix[:, 'Var'] == win_var_list[i * 20 + j]].ix[:, ['bads number', 'goods number']]
        axe = ptdata_1.plot(kind='bar', stacked=True, ax=axe, width=0.8, use_index=True, legend = False)
        axe.set_xticklabels(np.arange(11), rotation=0)
        #axe.set_ylabel('Goods/Bads Number')
        axf = axe.twinx()
        ptdata_2 = df_final_GB[df_final_GB.ix[:, 'Var'] == win_var_list[i * 20 + j]]['WOE']
        # axf.plot(axe.get_xticks(),ptdata_2, linestyle='-', marker='o',linewidth=2.0)
        axf.plot(axe.get_xticks(), ptdata_2, linestyle='-', marker='o', linewidth=2.0,color='#87CEFA')
        #axf.set_ylim((-4, 4))
        axe.set_title(win_var_list[i * 20 + j])
    plt.legend(loc='lower left', bbox_to_anchor=(1, 1.05))

# win list中所有变量按照5个分箱画图（首先分箱）
for i in range(0,15):
    df_x_table_list[i + 1] = pd.merge(df_x_table_list[i], df_x_table_list[i+1], on='win_uuid', how='left')
print df_x_table_list[15]
df_xy_table_FF = pd.merge(df_x_table_list[15], df_Y, on='win_uuid',how='left')
df_final_GB_1 = pd.DataFrame()
df_final_IV_1 = pd.DataFrame()
for item in win_var_list:
    df_single_x = df_xy_table_FF.ix[:,[item,'overdue_days']]
    bins, column_name, point = FF.create_bins_percentile(item, df_single_x)
    #bins_list.append(bins)
    IV, GB= FF.cal_info_value('overdue_days', bins, column_name, point)
    df_final_GB_1 = df_final_GB_1.append(GB)
    df_final_IV_1 = df_final_IV_1.append(IV)
    # print df_final_GB,df_final_IV
df_final_IV_1 = pd.merge(df_final_IV_1,df_missing_count_total, on='Var',how='left' )
df_final_GB_1 = pd.merge(df_final_GB_1,df_missing_count_total, on='Var',how='left' )
#df_final_IV_1.sort_values(by=['IV'], ascending=False, inplace=True)
#df_final_GB.sort_values(by='Category', ascending=False, inplace=True)
df_final_IV_1.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\IV_1.xlsx', encoding='gbk')
df_final_GB_1.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\GB_1.xlsx', encoding='gbk')

for i in range (0,len(win_var_list)/20+1):
    for j in range(0, 20):
        plt.figure(i)
        axe = plt.subplot(4,5,j+1)
        ptdata_1 = df_final_GB_1[df_final_GB_1.ix[:, 'Var'] == win_var_list[i * 20 + j]].ix[:, ['bads number', 'goods number']]
        #ptdata_1 = df_final_GB[df_final_GB.ix[:, 'Var'] == win_var_list[i * 20 + j]].ix[:, ['bads number', 'goods number']]
        axe = ptdata_1.plot(kind='bar', stacked=True, ax=axe, width=0.9, use_index=True, legend = False)
        axe.set_xticklabels(np.arange(11), rotation=0)
        #axe.set_ylabel('Goods/Bads Number')
        axf = axe.twinx()
        ptdata_2 = df_final_GB_1[df_final_GB_1.ix[:, 'Var'] == win_var_list[i * 20 + j]]['WOE']
        # axf.plot(axe.get_xticks(),ptdata_2, linestyle='-', marker='o',linewidth=2.0)
        axf.plot(axe.get_xticks(), ptdata_2, linestyle='-', marker='o', linewidth=2.0,color='#87CEFA')
        #axf.set_ylim((-4, 4))
        axe.set_title(win_var_list[i * 20 + j])
    plt.legend(loc='lower left', bbox_to_anchor=(1, 1.05))
    plt.show()





#df_choose_GB = df_final_GB[df_final_GB['Var'].isin (final_choose_list)]
#df_choose_GB.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\GB_choose.xlsx', encoding='gbk')

df_win_GB = df_final_GB[df_final_GB['Var'].isin (win_var_list)]
df_win_GB.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\GB_win0.xlsx', encoding='gbk')

#手动分箱并读取算IV和GB
df_adjust = pd.read_excel('C:\Users\marciahuang\Desktop\Python Output\V2\GB_win.xlsx')
overdue_table = df_xy_table_list[0].ix[:, [0, -2]]
df_total_xy = pd.merge(df_all_x_table, overdue_table, on='win_uuid', how='left')
df_final_GB_ad = pd.DataFrame()
df_final_IV_ad = pd.DataFrame()
bins_ad = []
for item in win_var_list:
    binnum = df_adjust[df_adjust['Var'] == item]['adjusted binning'].count()
    point_ad = df_adjust[df_adjust['Var'] == item]['adjusted binning'].dropna(axis=0, how='all').tolist()
    df_missing_ad = df_total_xy[pd.isnull(df_total_xy [item])]
    df_values_ad = df_total_xy[~pd.isnull(df_total_xy[item])]
    bins_list_ad = []
    bins_list_ad.append(df_missing_ad)
    i = 0
    while i < binnum-2:
        df_bin_ad = df_values_ad[(df_values_ad[item].astype(float) >= point_ad[i]) & (
        df_values_ad[item].astype(float) < point_ad[i + 1])]
        bins_list_ad.append(df_bin_ad)
        i += 1
    df_bin_ad = df_values_ad[(df_values_ad[item].astype(float) >= point_ad[binnum-2]) & (
        df_values_ad[item].astype(float) <= point_ad[binnum-1])]
    bins_list_ad.append(df_bin_ad)
    #print col_name,bins_list
    IV_ad, GB_ad = F.cal_info_value('overdue_days', bins_list_ad, item, point_ad)
    df_final_IV_ad = df_final_IV_ad.append(IV_ad)
    df_final_GB_ad = df_final_GB_ad.append(GB_ad)
    bins_ad.append(bins_list_ad)
print len(bins_ad)
df_final_IV_ad = pd.merge(df_final_IV_ad, df_missing_count_total, on='Var', how='left')
#df_final_GB_ad = pd.merge(df_final_GB_ad, df_missing_count_total, on='Var', how='left')
#df_final_IV_ad.sort_values(by=['IV'], ascending=False, inplace=True)
df_final_IV_ad.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\IV_ad.xlsx', encoding='gbk')
df_final_GB_ad.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\GB_ad.xlsx', encoding='gbk')

# 参照图来选择最终变量
#final_choose_list = ['run_4_max', 'pub_1_r7_u', 'run_6_r30_agr', 'god_5_u',  'pub_11_cv', 'hbo_6_agr']


for i in range (0,len(final_choose_list)/20+1):
    for j in range(0, 20):
        plt.figure(i)
        axe = plt.subplot(4,5,j+1)
        ptdata_1 = df_final_GB_ad[df_final_GB_ad.ix[:, 'Var'] == final_choose_list[j]].ix[:, ['bads number', 'goods number']]
        axe = ptdata_1.plot(kind='bar', stacked=True, ax=axe, width=0.9, use_index=True, legend = False)
        axe.set_xticklabels(np.arange(11), rotation=0)
        #axe.set_ylabel('Goods/Bads Number')
        axf = axe.twinx()
        ptdata_2 = df_final_GB_ad[df_final_GB_ad.ix[:, 'Var'] == final_choose_list[j]]['WOE']
        # axf.plot(axe.get_xticks(),ptdata_2, linestyle='-', marker='o',linewidth=2.0)
        axf.plot(axe.get_xticks(), ptdata_2, linestyle='-', marker='o', linewidth=2.0)
        axf.set_ylim((-4, 4))
        axe.set_title(final_choose_list[i * 20 + j])
    plt.legend(loc='lower left', bbox_to_anchor=(1, 1.05))
    plt.show()


#final_choose_list = ['ibm_3_agr','run_6_agr','pub_11_cv','pub_9_max','god_1_u','run_4_r30_cv','hbo_6_agr']
final_choose_list = win_var_list
pre_dummy_list = []
df_pre_dummy = pd.DataFrame()
df_adjust = pd.read_excel('C:\Users\marciahuang\Desktop\Python Output\V2\GB_win.xlsx')
overdue_table = df_xy_table_list[0].ix[:, [0, -2]]
df_total_xy = pd.merge(df_all_x_table, overdue_table, on='win_uuid', how='left')
for item in final_choose_list:
    #df_single_x = df_xy_table_FF.ix[:, [item, 'overdue_days']]
    #bins, column_name, point = FF.create_bins_percentile(item, df_single_x)
    binnum = df_adjust[df_adjust['Var'] == item]['adjusted binning'].count()
    point_ad = df_adjust[df_adjust['Var'] == item]['adjusted binning'].dropna(axis=0, how='all').tolist()
    df_missing_ad = df_total_xy[pd.isnull(df_total_xy[item])]
    df_values_ad = df_total_xy[~pd.isnull(df_total_xy[item])]
    bins_list_ad = []
    bins_list_ad.append(df_missing_ad)
    i = 0
    while i < binnum - 2:
        df_bin_ad = df_values_ad[(df_values_ad[item].astype(float) >= point_ad[i]) & (
            df_values_ad[item].astype(float) < point_ad[i + 1])]
        df_bin_ad.loc[:,'1'] = df_bin_ad.index.tolist()
        df_bin_ad.reset_index(drop=True, inplace=True)
        bins_list_ad.append(df_bin_ad)
        i += 1
    df_bin_ad = df_values_ad[(df_values_ad[item].astype(float) >= point_ad[binnum - 2]) & (
        df_values_ad[item].astype(float) <= point_ad[binnum - 1])]
    df_bin_ad.loc[:, '1'] = df_bin_ad.index.tolist()
    df_bin_ad.reset_index(drop=True,inplace=True)
    bins_list_ad.append(df_bin_ad)
    bins = bins_list_ad
    #print bins, point
    df_sig_dummy = pd.DataFrame()
    for i in range(0,len(bins)):
        #bins[i]['1'] = bins[i].index.tolist()
        #bins[i].loc[:,item].replace(bins[i].loc[:,item], i, inplace = True) #将这个var相应的值改成01234类型
        bins[i].loc[:, item] = pd.Series([i]*(bins[i].shape[0]))
        print bins[i]
        df_sig_dummy = df_sig_dummy.append(bins[i])  #
        df_sig_dummy.sort_values(by=['1'], ascending=True, inplace=True)#所有样本的该Var值都变成了01234类型
        df = df_sig_dummy.ix[:,[item,'1']]
    #pre_dummy_list.append(df_sig_dummy)
    pre_dummy_list.append(df)
df_pre_dummy = pd.merge(pre_dummy_list[0], pre_dummy_list[1], on='1', how='left')
for i in range(2,len(final_choose_list)):
    df_pre_dummy = pd.merge(df_pre_dummy,pre_dummy_list[i], on='1',how='left')
df_pre_dummy_ready = pd.merge(df_pre_dummy,df_sig_dummy.ix[:,['overdue_days','1']], on='1',how='left')
#list_mid = final_choose_list + ['overdue_days_x']
#df_pre_dummy_ready = df_pre_dummy[list_mid]
#df_pre_dummy_ready = df_pre_dummy_ready.T.drop_duplicates().T
#df_pre_dummy_ready = df_pre_dummy_ready.rename(columns = {'overdue_days_x':'overdue_days'})

print df_pre_dummy_ready.describe()

dummy_list= []
for item in final_choose_list:
    dummy_ranks = pd.get_dummies(df_pre_dummy_ready[item],prefix=item)
    print dummy_ranks.head()
    dummy_list.append(dummy_ranks.ix[:, 1:])
#df_dummy = dummy_list[0].join(dummy_list[1])
df_dummy = pd.DataFrame()
for i in range(0,len(final_choose_list)):
    df_dummy = dummy_list[i].join(df_dummy)
df_dummy = df_dummy.join(df_pre_dummy_ready['overdue_days'])
print df_dummy.head()

train_cols = df_dummy.columns[:-1]
logit = sm.Logit(df_dummy['overdue_days'],df_dummy.ix[:,:-1])
result = logit.fit()

print result.summary()
print result.conf_int()
print np.exp(result.params)

params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
print np.exp(conf)

'''
   binnum = df_adjust[df_adjust['Var'] == item]['adjusted binning'].count()
    point_ad = df_adjust[df_adjust['Var'] == item]['adjusted binning'].dropna(axis=0, how='all').tolist()
    df_missing_ad = df_total_xy[pd.isnull(df_total_xy [item])]
    df_values_ad = df_total_xy[~pd.isnull(df_total_xy[item])]
    bins_list_ad = []
    bins_list_ad.append(df_missing_ad)
    i = 0
    while i < binnum-2:
        df_bin_ad = df_values_ad[(df_values_ad[item].astype(float) >= point_ad[i]) & (
        df_values_ad[item].astype(float) < point_ad[i + 1])]
        bins_list_ad.append(df_bin_ad)
        i += 1
    df_bin_ad = df_values_ad[(df_values_ad[item].astype(float) >= point_ad[binnum-2]) & (
        df_values_ad[item].astype(float) <= point_ad[binnum-1])]
    bins_list_ad.append(df_bin_ad)
    bins = bins_list_ad
'''

df_score_pre = result.fittedvalues

#df_score_pre = ((df_score_pre - min(df_score_pre.tolist()))* 100).astype('int')
PDO = 20
sita = 0.001
P0 = 500
B = PDO/math.log(2)
A = P0 + B*math.log(sita)
df_score_pre = (A - B * df_score_pre).astype(int)

print min(df_score_pre.tolist()), max(df_score_pre.tolist())
df_pre_dummy_ready['Score'] = df_score_pre
df_score_bin = df_pre_dummy_ready.ix[:,-2:]


#每个样本的score分箱
bins_num = 10.0
df_values = df_score_bin
col_name = 'Score'
bins_list = []
# 求分箱的节点
j = 0
point = []
df_score = pd.DataFrame()
while j <= bins_num:
    point.append(df_values[col_name].quantile(j / bins_num))
    j += 1
# 求每个分箱中的dataframe，并返回包含几个分箱的列表
i = 0
while i < bins_num - 1:
    df_bin = df_values[
        (df_values[col_name].astype(float) >= point[i]) & (df_values[col_name].astype(float) < point[i + 1])]
    df_bin['Bin'] = [i]*len(df_bin)
    df_score = df_score.append(df_bin)
    bins_list.append(df_bin)
    i += 1
df_bin = df_values[(df_values[col_name].astype(float) >= point[9]) & (
    df_values[col_name].astype(float) <= point[10])]
df_bin['Bin'] = 9
bins_list.append(df_bin)
df_score = df_score.append(df_bin)
df_score['Bin'] = 9 - df_score['Bin']

df_final_IV_final = df_final_IV_1[df_final_IV_1['Var'].isin(final_choose_list)]
df_final_IV_final.sort_values(by=['IV'], ascending=False, inplace=True)
#df_final_GB.sort_values(by='Category', ascending=False, inplace=True)
df_final_IV_final.to_excel('C:/Users/admin/Desktop/Python Output/V2/IV_final.xlsx', encoding='gbk')

from sklearn import metrics
y = np.array(df_score['overdue_days'].tolist())
scores = np.array(df_score['Score'].tolist())
fpr, tpr, threholds = metrics.roc_curve(y, scores, pos_label = 1)
roc_score = metrics.roc_auc_score(y,scores)
df_roc = pd.DataFrame()
df_roc['fpr'] = fpr
df_roc['tpr'] = tpr
df_roc.to_excel('C:/Users/admin/Desktop/Python Output/V2/df_roc.xlsx', encoding='gbk')
print fpr, tpr, threholds, roc_score

'''
#分箱
bins_num = 10.0
df_values = df_score_bin
col_name = 'Score'
bins_list = []
# 求分箱的节点
j = 0
point = []
df_score = pd.DataFrame()
while j <= bins_num:
    point.append(df_values[col_name].quantile(j / bins_num))
    j += 1
# 求每个分箱中的dataframe，并返回包含几个分箱的列表
i = 0
while i < bins_num - 1:
    df_bin = df_values[(df_values[col_name].astype(float) >= point[i]) & (
        df_values[col_name].astype(float) < point[i + 1])]
    df_bin['Bin'] = [i]*len(df_bin)
    df_score = df_score.append(df_bin)
    bins_list.append(df_bin)
    i += 1
df_bin = df_values[(df_values[col_name].astype(float) >= point[9]) & (
    df_values[col_name].astype(float) <= point[10])]
df_bin['Bin'] = 9
bins_list.append(df_bin)
df_score = df_score.append(df_bin)
'''


# df_ks_cal = DataFrame(columns = ['Bin','Goods No','Bads No','Goods Per','Bads Per','Goods culm','Bads culm','Sample No','KS'])
df_ks_cal = pd.DataFrame()
score_bin = pd.Series(range(0,10))
df_ks_cal['score_bin'] = score_bin
bads_total_num =df_score['overdue_days'].groupby(df_score['Bin']).sum()
sample_num =df_score['overdue_days'].groupby(df_score['Bin']).count()
goods_total_num =df_score[df_score['overdue_days']==0]['overdue_days'].groupby(df_score[df_score['overdue_days']==0]['Bin']).count()
df_ks_cal['Goods No'] = goods_total_num
df_ks_cal['Bads No'] = bads_total_num
df_ks_cal['Sample No'] = sample_num
df_ks_cal['Bads Per'] = bads_total_num/df_score['overdue_days'].sum()
df_ks_cal['Goods Per'] = goods_total_num/df_score[df_score['overdue_days']==0]['overdue_days'].count()
#df_ks_cal['point'] = point
df_ks_cal['Goods_culm'] = df_ks_cal['Goods Per'].cumsum()
df_ks_cal['Bads_culm'] = df_ks_cal['Bads Per'].cumsum()
df_ks_cal['KS'] = df_ks_cal['Goods_culm']-df_ks_cal['Bads_culm']
print df_ks_cal['KS'].max()
df_ks_cal.to_excel('C:/Users/admin/Desktop/Python Output/V2/df_ks_cal2.xlsx', encoding='gbk')






'''
i = 0
j = 0
df_corr_count = pd.DataFrame()

for i in range(0,len(df_corr_spearman)):
    t = 0  # abs<0.3计数
    f = 0  # abs>=0.5计数
    m = 0  # 0.3<=abs<0.5计数
    for j in range(0,len(df_corr_spearman)):
        if abs(df_corr_spearman.ix[i,j]) >= 0.5 :
            f+=1
        elif abs(df_corr_spearman.ix[i,j]) < 0.3:
            t+= 1
        else:
            m+=1
    df = pd.DataFrame({'Var':df_corr_spearman.index[i], 'abs>=0.5':f, 'abs <0.3':t, '0.3<=abs <0.5':m},index=[0])
    df_corr_count = df_corr_count.append(df)
df_corr_count = pd.merge(df_category, df_corr_count, on='Var', how='left')
df_corr_count.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\corr_count.xlsx', encoding='gbk')


#只保留相关系数较大的变量中的一个，且该变量的missing值尽可能小
i = 0
j = 0
df_notnull_corr_obvious = pd.DataFrame() #存相关系数明显且非空的DF
corr_iv_compare_list = [] #相关系数明显且非空的DF中Var名称转成list
corr_iv_compare_list_total = []
max_iv_compare = []
corr_iv_compare_max_list = []
df_corr_iv_compare_max_total = pd.DataFrame()
df_corr_iv_compare_missingcount_total = pd.DataFrame()
for i in range(0,len(df_final_IV_filter_list)):
    df_notnull_corr_obvious= df_corr_obvious[~pd.isnull(df_corr_obvious.ix[:,i])]
    df_notnull_corr_p_obvious = df_corr_p_obvious[~pd.isnull(df_corr_p_obvious.ix[:, i])]
    corr_iv_compare_list = df_notnull_corr_obvious.index.tolist()
    corr_p_compare_list = df_notnull_corr_p_obvious.index.tolist()
    #df_corr_iv_compare = df_final_IV[df_final_IV['Var'].isin(corr_iv_compare_list)]  #在原来的IV表中找到在需compare的Var及其对应的IV值
    same = list(set(corr_iv_compare_list).intersection(set(corr_p_compare_list)))
    df_corr_iv_compare = df_final_IV[df_final_IV['Var'].isin(same)]
    #df_corr_iv_compare_missingcount = pd.merge(df_corr_iv_compare, df_missing_count_total, on='Var', how='left')
    #df_corr_iv_compare_missingcount['compare list no'] = i
    #f_corr_iv_compare_missingcount_total = df_corr_iv_compare_missingcount_total.append(df_corr_iv_compare_missingcount)

    # df_corr_iv_compare_max= df_corr_iv_compare[df_corr_iv_compare['IV'] == df_corr_iv_compare['IV'].max(2)]  #找到最大的Var及其对应的IV值
    df_corr_iv_compare.sort_values(by='IV', ascending=False, inplace=True)
    df_corr_iv_compare_top = df_corr_iv_compare[0:3]
    df_corr_iv_compare_top.sort_values(by='Missing count', ascending=True, inplace=True)
    df_corr_iv_compare_max = df_corr_iv_compare_top[0:1]


    df_corr_iv_compare_max_total = df_corr_iv_compare_max_total.append(df_corr_iv_compare_max)  #最大值的Var及其IV存在一张DF中
    #corr_iv_compare_list_total.append(df_corr_iv_compare_missingcount)     #把所有需要比较的Var及其对应的IV值存在list中
    #corr_iv_compare_max_list.append(df_corr_iv_compare_max)   #把所有最大的Var及其对应的IV值

#print corr_iv_compare_list_total, corr_iv_compare_max_list,df_corr_iv_compare_max_total
df_corr_iv_compare_max_total = df_corr_iv_compare_max_total.drop_duplicates()  #选出每组IV值最大的Var，并且把重复的Var去掉，余下的Var存进DF中
df_corr_iv_compare_max_total.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\ 1st Var.xlsx', encoding='gbk')

df_corr_lst = df_corr_iv_compare_max_total['Var'].tolist()

df_1st_corr_table = df_all_x_table.ix[:,df_corr_lst]
df_1st_corr_spearman = df_1st_corr_table.corr('spearman')
df_1st_corr_spearman.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\ 1st_corr_spearman.xlsx', encoding='gbk')

df_corr_p_1st = pd.DataFrame(stats.spearmanr(df_1st_corr_table)[1])
df_corr_p_1st.index = df_1st_corr_spearman.index
df_corr_p_1st.columns = df_1st_corr_spearman.columns
df_corr_p_1st.to_excel('C:\Users\marciahuang\Desktop\Python Output\V2\ corr_p_1st.xlsx', encoding='gbk')



for i in range (0,len(win_var_list)/20+1):
    plt.figure(i)
    #plt.subplot(4,5,20)
    trylist = [451, 452, 453, 454, 455, 456, 457, 458, 459, 4510, 4511, 4512, 4513, 4514, 4515, 4516, 4517, 4518, 4519, 4520]
    for j in range(0,20):
        plt.subplot(4,5,j+1)
        plt.bar(np.arange(11),
                    np.array(df_final_GB[df_final_GB.ix[:, 'Var'] == win_var_list[i*20+j]]['bads number']),
                    width=1, color='#FF7F50', label='bads number')
        plt.bar(np.arange(11),
                    np.array(df_final_GB[df_final_GB.ix[:, 'Var'] == win_var_list[i*20+j]]['goods number']),
                    width=1, color='#87CEFA',
                    bottom=np.array(df_final_GB[df_final_GB.ix[:, 'Var'] == win_var_list[i*20+j]]['bads number']),
                    label='goods number')
        plt.subplot(4, 5, j + 1).set_title(win_var_list[i*20+j])
        plt.show()
    plt.legend(loc='lower left', bbox_to_anchor=(1, 1.05))

'''


