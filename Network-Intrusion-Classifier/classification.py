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
import itertools




def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize = 56)
    plt.yticks(tick_marks, classes,fontsize = 56)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=56)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()




df=pd.read_csv('kddcup.data.corrected',names=range(1,43))
df.rename(columns={42:'class'},inplace=True)
X=df.drop(['class'],1)
y=df['class']


lb_style = LabelBinarizer()
classes = {"back.": 1, "land.": 2, "pod.": 3, "normal.": 4, "neptune.": 5, "smurf.": 6, "teardrop.": 7, "phf.": 8, "spy.": 9, "ftp_write.": 10,
            "guess_passwd.": 11, "imap.": 12, "multihop.": 13, "nmap.": 14, "perl.": 15, "rootkit.": 16, "buffer_overflow.": 17, "loadmodule.": 18,
             "warezclient.": 19, "warezmaster.": 20, "portsweep.": 21, "ipsweep.": 22, "satan.": 23}
y.replace(classes, inplace=True)
lb_result2=pd.DataFrame(lb_style.fit_transform(X[2]),columns=lb_style.classes_)
lb_result3=pd.DataFrame(lb_style.fit_transform(X[3]),columns=lb_style.classes_)
lb_result4=pd.DataFrame(lb_style.fit_transform(X[4]),columns=lb_style.classes_)

X.drop([2,3,4],1,inplace=True)
X=pd.concat([X,lb_result2,lb_result3,lb_result4],1)

y= pd.Series(y)
df=pd.read_csv('kdd_train_balanced.csv')
df.columns = range(1,43)
df.rename(columns={42:'class'},inplace=True)
X_test=df.drop(['class'],1)
y_test=df['class']


lb_style = LabelBinarizer()
y_test.replace(classes, inplace=True)
lb_result2=pd.DataFrame(lb_style.fit_transform(X_test[2]),columns=lb_style.classes_)
lb_result3=pd.DataFrame(lb_style.fit_transform(X_test[3]),columns=lb_style.classes_)
lb_result4=pd.DataFrame(lb_style.fit_transform(X_test[4]),columns=lb_style.classes_)

X_test.drop([2,3,4],1,inplace=True)
X_test=pd.concat([X_test,lb_result2,lb_result3,lb_result4],1)

y_test= pd.Series(y_test)

X, X_test = X.align(X_test, join='outer', axis=1)
X.fillna(0,inplace=True)
X_test.fillna(0,inplace=True)


uniqueList = []
unique  = y.unique()
size  = unique.size

 
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X,y)
y_pred=clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
accuracy_by_class = []
classified_by_class = []
misclassified_by_class = []




for i in range(0,size):
    temp = y_test[:] == unique[i]
    
    for c,u in classes.items():
        if(u == unique[i]): print(i," ",c)
     
    res = y_test[temp] == y_pred[temp]
    classified_by_class.append(sum(res))
    misclassified_by_class.append(y_test[temp].shape[0]-sum(res))
    accuracy_by_class.append(accuracy_score(y_test[temp],y_pred[temp]))

class_names = []

for c,u in classes.items():
    class_names.append(c)


cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

plt.figure(figsize=(100, 100),dpi=100)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')


plt.show()
