# -*- coding: utf-8 -*-

"""
Created on Fri Mar  6 22:06:48 2020

@author: vbhoj
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as pt
import seaborn as sb
import pickle 

df = pd.read_csv("diabetes.csv")
# 1. Data collection - done..!!

# 2. Data preprocessing - done..!!

# 3. EDA - 


'********************************************************'
'*************EDA data preprocessing*********************'
'********************************************************'

# Step 1 checking missing values
df.info()

'checking stats values for missing check or zeros values becz mean cannot be zero'
df.describe().T

'creating the deep of dataset'
diabetes_copy = df.copy(deep=True)

diabetes_copy[[i for i in df.columns if not i.startswith("Outcome")]] = diabetes_copy[[i for i in df.columns if not i.startswith("Outcome")]].replace(0,np.nan)
# X = X.replace(0,np.nan)


# checking the missing values
print(diabetes_copy.isnull().sum())

# Checking missing values percenatage
miss_per = pd.DataFrame({'Total_Missing' : diabetes_copy.isnull().sum(), 'per': diabetes_copy.isnull().sum()/diabetes_copy.shape[0], 'Skew':diabetes_copy.skew()})

#'Now filling the nan values with some values as per the histogram'
graphs = diabetes_copy.hist()


'filling missing values with require statistics as per the distribution of data'
'MISSING VALUE TREATMENT'
'Pregnancies'

diabetes_copy['Pregnancies'].fillna(diabetes_copy['Pregnancies'].median(), inplace=True)


'Glucose'
diabetes_copy['Glucose'].fillna(diabetes_copy['Glucose'].mean(), inplace=True)


'BloodPressure'
diabetes_copy['BloodPressure'].fillna(diabetes_copy['BloodPressure'].mean(), inplace=True)


'SkinThickness'
diabetes_copy['SkinThickness'].fillna(diabetes_copy['SkinThickness'].median(), inplace = True)


'Insulin'
diabetes_copy['Insulin'].fillna(diabetes_copy['Insulin'].median(), inplace = True)

'BMI'
diabetes_copy['BMI'].fillna(diabetes_copy['BMI'].median(), inplace = True)


'Age and DiabetesPedigreeFunction has no missing values'


'Step 2 :-checking the ouput varaible whether it is balanced or not'
bal_out = pd.DataFrame({'total values':diabetes_copy.Outcome.value_counts(),
                        'zero_percentage': diabetes_copy.Outcome.value_counts()[0]/diabetes_copy.shape[0],
                        'one_percentage' : diabetes_copy.Outcome.value_counts()[1]/diabetes_copy.shape[0]
                        })



'Step 3 : Checking for outliers and if outliers is present then we will remove those using z-score'
z = np.abs(stats.zscore(diabetes_copy))
print(z)

# threshold = 3
print(np.where(z > 3))

diabetes_copy_df = diabetes_copy[(z < 3).all(axis=1)]



'''
''## Null count analysis'
import missingno as msno
p=msno.bar(diabetes_copy)

'''
X = diabetes_copy_df.iloc[:,0:-1]
y = diabetes_copy_df.iloc[:,-1]


'''Feature Selection using two techniques '''
# Feature selection / feature engineering
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

fmodel = SelectKBest(score_func=chi2,k=5)
fmodel.fit(X,y)
features = fmodel.scores_
flist = pd.DataFrame({'columns':X.columns,'Scores':fmodel.scores_})

'''selecting the scores'''
fivelist = flist.nlargest(5,'Scores')



'''Feature selection using feature importance'''
from sklearn.ensemble import ExtraTreesClassifier
classifier = ExtraTreesClassifier()
classifier.fit(X,y)
vars = classifier.feature_importances_

newlist = pd.DataFrame({'col_name':X.columns,'specs':classifier.feature_importances_})
top5results = newlist.nlargest(5,'specs')




'Step 3'
'scatter matrix of uncleaned data'
from pandas.plotting import scatter_matrix
sm = scatter_matrix(diabetes_copy_df,figsize=(25,25))

'heatmap for clean data'
sb.heatmap(diabetes_copy_df.corr(),annot=True,cmap='RdYlGn',vmin=0.30)


'pariplot for clean data'
sb.pairplot(diabetes_copy_df,hue='Outcome')

'dropping unwanted columns'
X.drop(['DiabetesPedigreeFunction','BloodPressure', 'SkinThickness'],axis=1,inplace=True)




#columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#      'BMI', 'DiabetesPedigreeFunction', 'Age']
#X = pd.DataFrame(ssc.fit_transform(diabetes_copy.drop(['Outcome'],axis = 1)), 
#                  columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#                             'BMI', 'DiabetesPedigreeFunction', 'Age'])

'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(diabetes_copy.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
'''


'Step 4'
'splitting the data Fit,predict'

X = X.values
y = y.values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=.25,random_state=0)


'Step 5'
'Scalling the data'

from sklearn.preprocessing import StandardScaler
ssc = StandardScaler()
xtrain = ssc.fit_transform(xtrain)
xtest = ssc.fit_transform(xtest)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=15,metric="minkowski",p=1)
classifier.fit(xtrain,ytrain)
y_pred = classifier.predict(xtest)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest,y_pred)

# 'classifier score as per the default settings'
classifier.score(xtest,ytest)


# finding the best parameters for optimum result
from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[i for i in range(1,25)],'metric':['euclidean','minkowski','manhattan'],'p':[1,2,3]}


model = GridSearchCV(classifier,params, cv=3)
gsresults = model.fit(xtrain,ytrain)
model.best_params_
model.best_score_

'step 6'
'''MODEL PERFORMANCE ANALYSIS'''
'1. CONFUSION MATRIX'
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ytest,y_pred)
p = sb.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" , fmt='g')
pt.title('Confusion Matrix',loc='center')
pt.xlabel('Predicted values')
pt.ylabel('Actual values')


'2. CLASSFICATION REPORT-PRECION/RECALL REPORT'
from sklearn.metrics import classification_report
c_report = classification_report(ytest, y_pred)
TP = cm[1,1]
FP = cm[1,0]
FN = cm[0,1]
TN = cm[0,0]

Precion = TP/TP+FP

Recall = TN/FN+TN

print(c_report)

'3. AUC/ROC curve'
from sklearn.metrics import roc_curve
y_pred_proba = classifier.predict_proba(xtest)[:,1]
y_pred_proba


fpr,tpr,thresholds = roc_curve(ytest, y_pred_proba)
print("FPR",fpr)
print("FPR",tpr)
print("FPR",thresholds)


'plotting the roc curve'
pt.plot([0,1],[0,1],'k--')
pt.plot(fpr,tpr,label="KNN")
pt.xlabel('fpr')
pt.ylabel('tpr')
pt.title('K-Nearest Neighbour k=9 ROC Curve')


#Area under ROC curve
from sklearn.metrics import roc_auc_score
print(roc_auc_score(ytest,y_pred_proba))


# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

#comparing the results
model = pickle.load(open('model.pkl','rb'))
# val = pd.values

print(classifier.predict([[1,85,125,26.6,31]]))








