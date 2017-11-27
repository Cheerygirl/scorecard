# encoding=utf 8

import pandas as pd
import math
import MySQLdb
import F
import package
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pylab as pl
import sys

import numpy as np
import openpyxl
import scipy
from scipy.stats import spearmanr
import FF
import statsmodels.api as sm
from sklearn import metrics
from sklearn import tree
from ggplot import *
import xlrd

cnx = MySQLdb.connect(user='etl', passwd='etl@DCFTest#5234%',
                      host='application2.datacube.dcf.net',
                      charset='utf8')
# 创建df_x_table_list 和 df_xy_table_list，分别存储所有的x表和xyjoin起来的表
x_table = ['modelcentre.score_card_1','modelcentre.score_card_2','modelcentre.score_card_3','modelcentre.score_card_4','modelcentre.score_card_5','modelcentre.score_card_6',
           'modelcentre.score_card_7','modelcentre.score_card_8','modelcentre.score_card_9','modelcentre.score_card_10','modelcentre.score_card_11','modelcentre.score_card_12',
               'modelcentre.score_card_13','modelcentre.score_card_14','modelcentre.score_card_15','modelcentre.score_card_16']

df_Y = pd.read_sql('select * from modelcentre.score_card_y_value ', cnx)
df_Y =df_Y.ix[:,['win_uuid','overdue_days']]
package.Y_trans(df_Y)

#准备所有x与y存进两张df中
df_x = pd.DataFrame()
df_x_lst= []
for item in x_table:
    df_x = pd.read_sql('select * from ' + item, cnx)
    #df_x = df_x.T.drop(df_x.ix[:, 1:7].columns).T.ix[:,:-1]
    df_x = df_x.drop(df_x.ix[:, 1:7].columns,axis=1).ix[:, :-1]
    #print 'x table',df_x.shape[1]
    df_x_lst.append(df_x)
df_x_all= df_x_lst[0]
for i in range (0,15):
    df_x_all = pd.merge(df_x_all, df_x_lst[i+1],on='win_uuid',how='left')
print df_x_all.shape
df_xy_all = pd.merge(df_x_all, df_Y.ix[:,['win_uuid','overdue_days']], on='win_uuid',how='left')
df_xy_all = df_xy_all[~(df_xy_all['overdue_days'] == 2)]  #去掉不确定的样本

#去掉missing超过一半的变量,并映射到原有的
nomis_var_lst = package.nomis_var(df_xy_all)
df_xy_nomis = df_xy_all[['win_uuid'] +nomis_var_lst+['overdue_days']]
#判定变量种类
conti_var_lst,chara_var_lst,nan_var_lst = package.var_identify(df_xy_nomis)
#len(conti_var_lst),len(chara_var_lst),len(nan_var_lst)

noovd_conti_var=[]
for item in conti_var_lst:
    if 'bob' not in item and 'dfc' not in item and 'etl' not in item and 'kok' not in item and 'lol' not in item and 'mum'not in item and 'pub_38'not in item and 'pub_39'not in item and 'run_2'not in item:
        noovd_conti_var.append(item)

item = 'tvb_1_min'
item = 'jbw_30_agr'
df_x_dt = df_xy_all.ix[:, [item, 'overdue_days']]
df_x_value = df_x_dt[~pd.isnull(df_x_dt.ix[:, 0])]

['jbw_1_agr', 'jbw_2_agr', 'jbw_9_agr', 'jbw_15_agr', 'jbw_30_agr']


#计算IV值
gb = pd.DataFrame()
IV = pd.DataFrame()
for item in noovd_conti_var:
    df_x_dt = df_xy_all.ix[:, [item, 'overdue_days']]
    df_x_value = df_x_dt[~pd.isnull(df_x_dt.ix[:,0])]
    sample_num = len(df_x_dt.index)  #sample_num = len(df_x_value.index)
    nodes_all = package.DT(df_x_value)
    break_nodes = []#专存break的节点
    break_nodes = package.noding(df_x_value,nodes_all,sample_num,break_nodes)
    df_gb = package.con_gb(break_nodes,df_x_dt)
    df_IV = package.IV_cal(df_gb)
    gb = gb.append(df_gb)
    IV = IV.append(df_IV)
for item in chara_var_lst:
    #item  = chara_var_lst[1]
    df_x_dt = df_xy_all.ix[:, [item, 'overdue_days']]
    df_gb = package.cha_gb(df_x_dt)
    df_IV = package.IV_cal(df_gb)
    gb = gb.append(df_gb)
    IV = IV.append(df_IV)
gb.to_excel('C:\Users\marciahuang\Desktop\DraftDT\gb.xlsx', encoding='gbk')
IV.to_excel('C:\Users\marciahuang\Desktop\DraftDT\IV.xlsx', encoding='gbk')
gb_try = gb

IV[IV.ix[:,1]>0.1]

var_choice = ['pub_12_cv', 'run_4_max', 'pub_30_max', 'pub_11_r30_cv', 'pub_9_r30_u', 'fbi_3_r30_u', 'pub_32_std', 'sun_9_agr', 'god_5_max', 'tvb_1_min', 'tvb_5_agr', 'run_13_std', 'pub_19_max', 'pub_3_min', 'usa_6_agr', 'qab_15_agr', 'qab_9_agr', 'sun_8_agr', 'usa_9_agr', 'ibm_3_agr', 'pub_32_min', 'qab_12_agr', 'hbo_12_agr']
gb_try = gb_try[gb_try['1_var'].isin(var_choice)]
gb_try.to_excel('C:\Users\marciahuang\Desktop\DraftDT\gb_try.xlsx', encoding='gbk')


for i in range(1,len(df_xy_all.columns)-1):
    df_x_dt = df_xy_all.ix[:, [df_xy_all.columns[i], 'overdue_days']]
    if (df_x_dt.ix[:, 0].dtype == np.int64) | (df_x_dt.ix[:, 0].dtype == np.float64):
        conti_var_lst.append(df_x_dt.columns[0])

#创建原始需分箱的变量df，并且按从小到大顺序排列
#df_x_dt = df_xy_all.ix[:, ['fbi_1_min', 'overdue_days']]








