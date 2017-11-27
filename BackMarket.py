#!/usr/bin/python
# -*- coding: utf-8 -*
__author__ = 'Cheery'

#Imports
#dataprocessing & viturlization
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='notebook',style="ticks",palette="GnBu_d",font_scale=1.5,font='ETBembo',rc={"figure.figsize": (10, 6)})
#plt.rcParams['figure.figsize']=(15,10)
import warnings
warnings.filterwarnings('ignore') #为了整洁，去除弹出的warnings
pd.set_option('precision', 5) #设置精度
pd.set_option('display.float_format', lambda x: '%.5f' % x) #为了直观的显示数字，不采用科学计数法
pd.options.display.max_rows = 200 #最多显示200行

#machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#data extracting
rdata = pd.read_csv('bank-full.csv',sep=';',dtype={'balance':np.float64})
radata = pd.read_csv('bank-additional-full.csv',sep=';',dtype={'balance':np.float64})

##########################data_check#########################################
radata.columns
radata.info()
radata.describe()

radata.describe().astype(np.int64).T
radata.select_dtypes(include=['O']).describe().T.assign(missing_pct=radata.apply(lambda x : (len(x)-x.count())/len(x)))
(radata.select_dtypes(include=['float64']).describe().T) .drop(['25%','50%','75%'],axis=1)\
    .assign(missing_pct=radata.apply(lambda x: (len(x)-x.count())/len(x)),nunique = radata.apply(lambda x: x.nunique()),
            pct_10=radata.select_dtypes(include=['float64']).apply(lambda x: x.dropna().quantile(.1)),
            pct_25=radata.select_dtypes(include=['float64']).apply(lambda x: x.dropna().quantile(.25)),
            pct_50=radata.select_dtypes(include=['float64']).apply(lambda x: x.dropna().quantile(.5)),
            pct_75=radata.select_dtypes(include=['float64']).apply(lambda x: x.dropna().quantile(.75)),
            pct_90=radata.select_dtypes(include=['float64']).apply(lambda x: x.dropna().quantile(.9)))
demo_data = radata.loc[:,radata.columns.str.contains('num')] #取包含‘num’的所有变量
sns.boxplot(data=demo_data,orient="h",color="c")

sns.despine(trim=True,offset=10)
############################记录重复值检查与处理#############################
radata[radata.duplicated()==True].count()
radata[radata.duplicated()==True]
radata.drop_duplicates()

##########################空值检查与处理######################################
radata.isnull().any()
radata[radata.isnull().values==True]

#缺失值较多的特征处理
# 定义工资改变特征缺失值处理函数，将有变化设为Yes，缺失设为No
def set_salary_change(df):
    df.loc[(df.salary_change.notnull()), 'salary_change'] = 'Yes'
    df.loc[(df.salary_change.isnull(df)), 'salary_change'] = 'No'
    print df
    pd.df()
    print df
    return df
radata = set_salary_change(radata)

#缺失值较少的特征处理
# 将所有行用各自的均值填充
radata.fillna(radata.mean())
# 也可以指定某些行进行填充
radata.fillna(radata.mean()['browse_his', 'card_num'])
# 用前一个数据代替NaN：method='pad'
radata.fillna(method='pad')
# 与pad相反，bfill表示用后一个数据代替NaN
radata.fillna(method='bfill')
# 插值法就是通过两点（x0，y0），（x1，y1）估计中间点的值
radata.interpolate()
#用算法拟合进行填充
# 定义browse_his缺失值预测填充函数
def set_missing_browse_his(df):
    # 把已有的数值型特征取出来输入到RandomForestRegressor中
    process_df = df[['browse_his', 'gender', 'job', 'edu', 'marriage', 'family_type']]
    # 乘客分成已知该特征和未知该特征两部分
    known = process_df[process_df.browse_his.notnull()].as_matrix()
    unknown = process_df[process_df.browse_his.isnull()].as_matrix()
    # X为特征属性值
    X = known[:, 1:]
    # y为结果标签值
    y = known[:, 0]
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000,  n_jobs=-1)
    rfr.fit(X,y)
    # 用得到的模型进行未知特征值预测
    predicted = rfr.predict(unknown[:, 1::])
    # 用得到的预测结果填补原缺失数据
    df.loc[(df.browse_his.isnull()), 'browse_his'] = predicted
    return df, rfr
radata, rfr = set_missing_browse_his(radata)

##
radata[radata.isnull()==True].count()
radata[radata.notnull()==True].count()

radata.fillna(0)
radata.dropna()
radata['loan_amnt']=radata['loan_amnt'].fillna(radata['total_pymnt']-radata['total_rec_int']).astype(np.int64)
radata['annual_inc']=radata['annual_inc'].fillna(radata['annual_inc'].mean())

##################异常值检查与处理###############################################
#连续变量
radata['age'].value_counts()
sns.distplot(radata['age'])
radata.boxplot(column='age')
radata.boxplot(column='age',by='education')
#分类变量
sns.stripplot(x='education',y='age',data=radata,jitter=True)
sns.stripplot(x='education',y='age', hue="y",data=radata, jitter=True)

#ata.replace([A, B], [A_R, B_R])  data.replace({A:A_R, B:B_R})
radata['loan_amnt'].replace([100000,36],radata['loan_amnt'].mean())
radata['age'].replace([158, 6], np.nan)
sigp = radata['age'].mean() + 3 * radata['age'].std()
sigm = radata['age'].mean() - 3 * radata['age'].std()
radata['age'].replace(radata[radata['age']>sigp].age,sigp)
radata['age'].replace(radata[radata['age']<sigm].age,sigm)

#检查和去字符空格
radata['education'].value_counts()
radata['term'] = radata['term'].map(str.strip)
radata['term'] = radata['term'].map(str.lstrip)
radata['term'] = radata['term'].map(str.rstrip)
#调大小写
radata['term']=radata['term'].map(str.upper)
radata['term']=radata['term'].map(str.lower)
radata['term']=radata['term'].map(str.title)
#检查每个变量类型是否一致
radata['emp_length'].apply(lambda x: x.isalpha())
radata['emp_length'].apply(lambda x: x. isalnum ())
radata['emp_length'].apply(lambda x: x. isdigit ())

##############修整数据格式#####################
radata['loan_amnt']=radata['loan_amnt'].astype(np.int64)
radata['issue_d']=pd.to_datetime(radata['issue_d'])
radata.dtypes

##############数据分箱######################
bins = [0, 5, 10, 15, 20]
group_names = ['A', 'B', 'C', 'D']
radata['categories'] = pd.cut(radata['open_acc'], bins, labels=group_names)

cutPoint=[0,30, 40, 50,80]
groupLabel=[0,1,2,3]
pd.cut(radata['age'],cutPoint)
radata['ageGroup'] =pd.cut(radata['age'],cutPoint,labels=groupLabel)
#按分位数
radata['ageGroup'] =pd.qcut(radata['age'],2)#分两组
pd.get_dummies(radata['ageGroup'] )
radata =pd.merge(radata, pd.get_dummies(radata['ageGroup'], prefix='age' ),right_index=True, left_index=True)
###########一个变量分裂############
grade_split = pd.DataFrame((x.split('-') for x in radata.grade),index=radata.index,columns=['grade','sub_grade'])
radata=pd.merge(radata,grade_split,right_index=True, left_index=True)

