# encoding=utf 8

import pandas as pd
import math
import MySQLdb
import numpy as np


def Y_trans(df_Y):  #定义好坏及不确定样本
    i = 0
    while i < df_Y.count()[1]:
        if df_Y.ix[i,1] > 45:
            df_Y.ix[i,1] = 1
        elif df_Y.ix[i,1] <= 0:
            df_Y.ix[i, 1] = 0
        else:
            df_Y.ix[i, 1] = 2
        i += 1
    return df_Y

def nomis_var(df_xy_all):  #去掉missing超过一半的变量
    nomis_var_lst = []
    for i in range(1,len(df_xy_all.columns)-1):
        df_x_dt = df_xy_all.ix[:, i]
        mis_count = len(df_x_dt[pd.isnull(df_x_dt)].get_values())
        if mis_count<len(df_xy_all.index)*0.5:
            nomis_var_lst.append(df_xy_all.columns[i])
    return nomis_var_lst

def var_identify(df_xy_all):#判定哪些变量是连续的，哪些是名义的，哪些全是null值
    conti_var_lst = [] #数值变量
    chara_var_lst = [] #名义变量
    nan_var_lst = [] #全是null的变量
    for i in range(1,len(df_xy_all.columns)-1):
        if (df_xy_all.ix[:,i].dtype == np.int64) | (df_xy_all.ix[:,i].dtype == np.float64):
            conti_var_lst.append(df_xy_all.columns[i])
        elif pd.isnull(df_xy_all.ix[:,i]).all():
            nan_var_lst.append(df_xy_all.columns[i])
        else:
            chara_var_lst.append(df_xy_all.columns[i])
    #print len(conti_var_lst),len(chara_var_lst),len(nan_var_lst)
    return conti_var_lst,chara_var_lst,nan_var_lst

def Gini_cal(df_x_dt,nodes_all,num):  #计算Gini并找到Gini最小的节点作为break node
    if (len(df_x_dt.index)<0.1*num)| (len(nodes_all) <2) : #因为每个箱不低于5%的所有个数，因此当个数低于10%的个数时就不能往下分了
        gi_min = np.nan
        bre_node = np.nan
    else:
        x = pd.Series(0, index=[0, 1])
        df_gi = pd.DataFrame(columns = ['bre_point','Gini_left','Gini_right','Gini_weighted','Gount_left','Gount_right','Gount_0_left','Gount_1_left','Gount_0_right','Gount_1_right'])
        for i in range(0,len(nodes_all)):
            #mea = (df_x_dt.ix[i:i + 1, 0]).mean()
            df_x_dtl = df_x_dt[df_x_dt.ix[:, 0] <= nodes_all[i]]
            df_x_dtr = df_x_dt[df_x_dt.ix[:, 0] > nodes_all[i]]
            a = df_x_dtl['overdue_days'].groupby(df_x_dtl['overdue_days']).count()
            b = float(len(df_x_dtl.index))
            c = df_x_dtr['overdue_days'].groupby(df_x_dtr['overdue_days']).count()
            d = float(len(df_x_dtr.index))
            e = x.add(a,fill_value = 0)
            f = x.add(c,fill_value = 0)
            '''
                       if b<0.05*num or d<0.05*num:
                continue
            else:
                try:
                    gi_l = 1 - (float(e[0]) / b) ** 2 - (float(e[1]) / b) ** 2
                    gi_r = 1 - (float(f[0]) / d) ** 2 - (float(f[1]) / d) ** 2
                except:
                    print "0 denominator"
                    #break
                else:
                    gi = b / (b + d) * gi_l + d / (b + d) * gi_r
                    df_gi = df_gi.append(pd.DataFrame({'bre_point': nodes_all[i], 'Gini_left': gi_l, 'Gini_right': gi_r, 'Gini_weighted': gi,'Gount_left':b,'Gount_right':d,'Gount_0_left':e[0],'Gount_1_left':e[1],'Gount_0_right':f[0],'Gount_1_right':f[1]},index=[i]))
            '''
            if b>0.05*num and d>0.05*num:
                try:
                    gi_l = 1 - (float(e[0]) / b) ** 2 - (float(e[1]) / b) ** 2
                    gi_r = 1 - (float(f[0]) / d) ** 2 - (float(f[1]) / d) ** 2
                except:
                    print "0 denominator",b,d,num
                    #break
                else:
                    gi = b / (b + d) * gi_l + d / (b + d) * gi_r
                    df_gi = df_gi.append(pd.DataFrame({'bre_point': nodes_all[i], 'Gini_left': gi_l, 'Gini_right': gi_r, 'Gini_weighted': gi,'Gount_left':b,'Gount_right':d,'Gount_0_left':e[0],'Gount_1_left':e[1],'Gount_0_right':f[0],'Gount_1_right':f[1]},index=[i]))
        if df_gi.empty:
            gi_min = np.nan
            bre_node = np.nan
        else:
            gi_min = df_gi['Gini_weighted'].min()
            bre_node = df_gi[df_gi['Gini_weighted'] == gi_min]['bre_point'].get_values()[0]
    return gi_min,bre_node

def noding(df_x_dt,nodes_all,sample_num,break_nodes):  #采用递归算法，算出所有的break nodes
    #如果分不出来bre_node，说明已经分完了，跳出递归
    gini_min, bre_node = Gini_cal(df_x_dt,nodes_all,sample_num)
    #print bre_node
    if ~np.isnan(bre_node):     #~(bre_node ==np.nan):
        #print bre_node
        break_nodes.append(bre_node)
        nodes_left = nodes_all[0:nodes_all.index(bre_node)+1]
        nodes_right =nodes_all[nodes_all.index(bre_node) + 1:len(nodes_all)+1]
        df_x_right = df_x_dt[df_x_dt.ix[:, 0] > bre_node]
        df_x_left = df_x_dt[df_x_dt.ix[:, 0] <= bre_node]
        noding(df_x_left, nodes_left,sample_num,break_nodes), noding(df_x_right, nodes_right,sample_num,break_nodes)
    break_nodes.sort()
    return break_nodes

def DT(df_x_dt):
  #单个变量计算出分的节点
    df_x_dt.sort_values(by=df_x_dt.columns[0], ascending=True, inplace=True)
    df_x_dt.index = range(len(df_x_dt.index))
    #计算所有的nodes存进nodes_all里面
    df_x_dt_dropD = df_x_dt.ix[:, 0].drop_duplicates()
    df_x_dt_dropD.index = range(len(df_x_dt_dropD.index))
    nodes_all= []
    for i in range(0,len(df_x_dt_dropD)-1):
        dtmean = (df_x_dt_dropD[i]+df_x_dt_dropD[i+1])/2
        nodes_all.append(dtmean)
    #print nodes_all
    return nodes_all
    #传进需计算的变量df，所有的节点
    #noding(df_x_dt,nodes_all,sample_num)


def con_gb(break_nodes,df_x_dt): #计算连续变量根据break nodes划分出来的样本分类
    x =df_x_dt.ix[:,0].min()-1 #控制
    bins_list = []
    #df_gb = pd.DataFrame(columns = ['1_var','2_bin','3_good_no','4_bad_no','5_total_no'])
    df_gb = pd.DataFrame()
    df_missing = df_x_dt[pd.isnull(df_x_dt.ix[:,0])]
    df_x_dt = df_x_dt[~pd.isnull(df_x_dt.ix[:,0])]
    break_nodes.append(df_x_dt.ix[:,0].max())
    for item in break_nodes:
        df_bin = df_x_dt[(df_x_dt.ix[:, 0] <= item) & (df_x_dt.ix[:, 0] > x)]
        bins_list.append(df_bin)
        x = item
        good_no = float(df_bin[df_bin.ix[:,1] ==0].count()[1])
        bad_no = float(df_bin[df_bin.ix[:, 1] == 1].count()[1])
        df_gb = df_gb.append([{'1_var':df_x_dt.columns[0],'2_bin':break_nodes.index(item),'3_good_no':good_no,'4_bad_no':bad_no,'5_total_no':good_no+bad_no,'6_break_node':item}])

    df_gb = df_gb.append([{'1_var': df_x_dt.columns[0], '2_bin': len(break_nodes), '3_good_no': df_missing[df_missing.ix[:,1] ==0].count()[1],
                           '4_bad_no': df_missing[df_missing.ix[:,1] ==1].count()[1], '5_total_no': (df_missing[df_missing.ix[:,1] ==0].count()[1]) + (df_missing[df_missing.ix[:,1] ==1].count()[1]),'6_break_node':'missing'}])
    df_gb.index = range(len(df_gb.index))
    return df_gb



def cha_gb(df_x_dt):  #计算名义变量的样本分类
    bins_list = []
    df_gb = pd.DataFrame()
    df_missing = df_x_dt[pd.isnull(df_x_dt.ix[:,0])]
    df_x_dt = df_x_dt[~pd.isnull(df_x_dt.ix[:,0])]
    #df_x_dt.ix[:,0].describe
    cha_val_lst = list(set(df_x_dt.ix[:,0].tolist()))
    for item in cha_val_lst:
        df_bin = df_x_dt[df_x_dt.ix[:,0] == item]
        bins_list.append(df_bin)
        good_no = float(df_bin[df_bin.ix[:,1] ==0].count()[1])
        bad_no = float(df_bin[df_bin.ix[:, 1] == 1].count()[1])
        df_gb = df_gb.append([{'1_var': df_x_dt.columns[0], '2_bin': cha_val_lst.index(item), '3_good_no': good_no,
                               '4_bad_no': bad_no, '5_total_no': good_no + bad_no,'6_break_node':item}])
    bins_list.append(df_missing)
    df_gb = df_gb.append([{'1_var': df_x_dt.columns[0], '2_bin': len(cha_val_lst), '3_good_no': df_missing[df_missing.ix[:,1] ==0].count()[1],
                           '4_bad_no': df_missing[df_missing.ix[:,1] ==1].count()[1], '5_total_no': (df_missing[df_missing.ix[:,1] ==0].count()[1]) + (df_missing[df_missing.ix[:,1] ==1].count()[1]),'6_break_node':'missing'}])
    df_gb.index = range(len(df_gb.index))
    return df_gb

def IV_cal(df_gb):  #名义变量和连续变量都可用此计算IV值
    if df_gb['3_good_no'].sum()==0.0:
        df_gb['good_rate'] = 0.0
    else:
        df_gb['good_rate'] = df_gb['3_good_no']/(df_gb['3_good_no'].sum())
    if df_gb['4_bad_no'].sum()==0.0:
        df_gb['bad_rate'] = 0.0
    else:
        df_gb['bad_rate'] = df_gb['4_bad_no']/(df_gb['4_bad_no'].sum())

    for i in range(0,len(df_gb.index)):
        if df_gb.ix[i,'bad_rate'] ==0.0:
            df_gb.ix[i, 'info_odds'] = 0
            df_gb.ix[i, 'woe'] = 0
            df_gb.ix[i, 'info'] = 0
        else:
            df_gb.ix[i, 'info_odds'] = df_gb.ix[i, 'good_rate'] / df_gb.ix[i, 'bad_rate']
            if df_gb.ix[i, 'info_odds']  ==0:
                df_gb.ix[i, 'woe'] = 0
                df_gb.ix[i, 'info'] = 0
            else:
                df_gb.ix[i, 'woe'] = np.log(df_gb.ix[i, 'info_odds'])
                df_gb.ix[i, 'info'] = (df_gb.ix[i, 'good_rate'] - df_gb.ix[i, 'bad_rate'])*df_gb.ix[i, 'woe']
    info_value = df_gb['info'].sum()
    df_IV = pd.DataFrame([{'1_var': df_gb.ix[0,'1_var'],'2_IV': info_value}])
    return df_IV

'''
def noding(df_x_dt,nodes_all,sample_num):
    #如果分不出来bre_node，说明已经分完了，跳出递归
    break_nodes = [] #专存break的节点
    gini_min, bre_node = Gini_cal(df_x_dt,nodes_all,sample_num)
    if ~np.isnan(bre_node):     #~(bre_node ==np.nan):
        #print bre_node
        break_nodes.append(bre_node)
        nodes_left = nodes_all[0:nodes_all.index(bre_node)+1]
        nodes_right =nodes_all[nodes_all.index(bre_node) + 1:len(nodes_all)+1]
        df_x_right = df_x_dt[df_x_dt.ix[:, 0] > bre_node]
        df_x_left = df_x_dt[df_x_dt.ix[:, 0] <= bre_node]
        noding(df_x_left, nodes_left,sample_num), noding(df_x_right, nodes_right,sample_num)
        return break_nodes.sort()
    else:
        return break_nodes.sort()

def DT(df_x_dt):  #单个变量计算出分的节点
    df_x_dt.sort_values(by=df_x_dt.columns[0], ascending=True, inplace=True)
    df_x_dt.index = range(len(df_x_dt.index))
    #计算所有的nodes存进nodes_all里面
    df_x_dt_dropD = df_x_dt.ix[:, 0].drop_duplicates()
    df_x_dt_dropD.index = range(len(df_x_dt_dropD.index))
    nodes_all= []
    sample_num = len(df_x_dt.index)
    for i in range(0,len(df_x_dt_dropD)-1):
        dtmean = (df_x_dt_dropD[i]+df_x_dt_dropD[i+1])/2
        nodes_all.append(dtmean)
    #print nodes_all
    #传进需计算的变量df，所有的节点
    noding(df_x_dt,nodes_all,sample_num)
'''




