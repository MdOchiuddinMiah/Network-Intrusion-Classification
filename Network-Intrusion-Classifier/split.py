import numpy as np
from numpy import  genfromtxt
import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.model_selection import train_test_split,KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier,RandomForestRegressor,RandomForestClassifier,BaggingClassifier,AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from itertools import permutations
from sklearn.feature_selection import f_regression
from matplotlib import pyplot as plt
import seaborn as sns


df=pd.read_csv('kddcup.csv',names=range(1,43))
df.rename(columns={42:'class'},inplace=True)
X=df

lb_style = LabelBinarizer()
classes = {"back.": 1, "land.": 2, "pod.": 3, "normal.": 4, "neptune.": 5, "smurf.": 6, "teardrop.": 7, "phf.": 8, "spy.": 9, "ftp_write.": 10,
            "guess_passwd.": 11, "imap.": 12, "multihop.": 13, "nmap.": 14, "perl.": 15, "rootkit.": 16, "buffer_overflow.": 17, "loadmodule.": 18,
             "warezclient.": 19, "warezmaster.": 20, "portsweep.": 21, "ipsweep.": 22, "satan.": 23}
X['class'].replace(classes, inplace=True)
lb_result2=pd.DataFrame(lb_style.fit_transform(X[2]),columns=lb_style.classes_)
lb_result3=pd.DataFrame(lb_style.fit_transform(X[3]),columns=lb_style.classes_)
lb_result4=pd.DataFrame(lb_style.fit_transform(X[4]),columns=lb_style.classes_)

X.drop([2,3,4],1,inplace=True)
X=pd.concat([X,lb_result2,lb_result3,lb_result4],1)


temp = pd.Series(X['class']).value_counts().sort_values(ascending=False)
majority = minority = pd.DataFrame()
ind = list(temp.index)
for i in ind:
    if(temp[i] < 100):
        X_temp = X[X['class'] == i]
        minority = minority.append(X_temp)
    else:
        X_temp = X[X['class'] == i]
        majority = majority.append(X_temp)
    

majority.to_csv('majority.csv', sep=',',index=None, encoding='utf-8')
minority.to_csv('minority.csv', sep=',',index=None, encoding='utf-8')
