from kdd_cup_2_layer import X_Majority
import numpy as np
from numpy import  genfromtxt
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder,LabelBinarizer,StandardScaler,Normalizer
from sklearn.model_selection import train_test_split,KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans,DBSCAN
from sklearn.ensemble import ExtraTreesClassifier,RandomForestRegressor,RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from datetime import datetime
import seaborn as sns
from matplotlib import pyplot as plt
from imblearn.under_sampling import ClusterCentroids,RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import make_classification
from collections import Counter


majority = df=pd.read_csv('majority.csv')
minority = df=pd.read_csv('minority.csv')



y_majority=majority['class']
y_minority=minority['class']


cc = ClusterCentroids(random_state=0)
rus = RandomUnderSampler(random_state=0,ratio=0.05)
X_resampled_majority, y_resampled_majority = rus.fit_resample(majority, y_majority)

ros = RandomOverSampler(random_state=0)
X_resampled_minority, y_resampled_minority = ros.fit_resample(majority, y_minority)

df1 = pd.concat([pd.DataFrame(X_resampled_majority),pd.DataFrame(y_resampled_majority)],1)
df2 = pd.concat([pd.DataFrame(X_resampled_minority),pd.DataFrame(y_resampled_minority)],1)
df = pd.concat([pd.DataFrame(df1),pd.DataFrame(df2)],1)

df.to_csv('kdd_train_balanced.csv', sep=',',index=None, encoding='utf-8')
