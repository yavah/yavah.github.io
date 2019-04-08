---
layout: post
title: Model selection for capstone project - Kiva
tags: [Classification, Model Selection, Python]
excerpt_separator: <!--more-->
---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")
from time import time

```python
import warnings
warnings.filterwarnings('ignore')
```


```python
import sys, os
sys.path.insert(0, os.path.abspath('..'))
```


```python
from Ingestion.kivadataloader import KivaDataLoader
m=KivaDataLoader()
cleaneduploans=m.get_clean_dataframe()
```

    Connection Failed
    The process takes about 5 minutes to run.
    ***** statement for get_clean_dataframe failed *****



```python
cleaneduploans.shape
```




    (1177384, 26)




```python
cleaneduploans.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>language_english</th>
      <th>description_length</th>
      <th>loan_amount</th>
      <th>loan_use_length</th>
      <th>currency_usd</th>
      <th>tags_exist</th>
      <th>num_borrowers_female_pct</th>
      <th>sector_name_Agriculture</th>
      <th>sector_name_Arts</th>
      <th>sector_name_Clothing</th>
      <th>...</th>
      <th>sector_name_Personal Use</th>
      <th>sector_name_Retail</th>
      <th>sector_name_Services</th>
      <th>sector_name_Transportation</th>
      <th>sector_name_Wholesale</th>
      <th>distribution_model_field_partner</th>
      <th>fol</th>
      <th>repayment_interval_bullet</th>
      <th>repayment_interval_irregular</th>
      <th>repayment_interval_weekly</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>539.0</td>
      <td>1000.0</td>
      <td>41.0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>858.0</td>
      <td>300.0</td>
      <td>37.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>875.0</td>
      <td>125.0</td>
      <td>86.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>706.0</td>
      <td>1025.0</td>
      <td>22.0</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>680.0</td>
      <td>1750.0</td>
      <td>15.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
cleaneduploans.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>language_english</th>
      <th>description_length</th>
      <th>loan_amount</th>
      <th>loan_use_length</th>
      <th>currency_usd</th>
      <th>tags_exist</th>
      <th>num_borrowers_female_pct</th>
      <th>sector_name_Agriculture</th>
      <th>sector_name_Arts</th>
      <th>sector_name_Clothing</th>
      <th>...</th>
      <th>sector_name_Personal Use</th>
      <th>sector_name_Retail</th>
      <th>sector_name_Services</th>
      <th>sector_name_Transportation</th>
      <th>sector_name_Wholesale</th>
      <th>distribution_model_field_partner</th>
      <th>fol</th>
      <th>repayment_interval_bullet</th>
      <th>repayment_interval_irregular</th>
      <th>repayment_interval_weekly</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
      <td>...</td>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
      <td>1.177384e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.905045e-01</td>
      <td>7.545755e+02</td>
      <td>7.579580e+02</td>
      <td>5.375018e+01</td>
      <td>1.559822e-01</td>
      <td>4.059449e-01</td>
      <td>7.920015e-01</td>
      <td>2.382485e-01</td>
      <td>2.239711e-02</td>
      <td>5.509757e-02</td>
      <td>...</td>
      <td>3.827383e-02</td>
      <td>2.000214e-01</td>
      <td>7.075941e-02</td>
      <td>2.567302e-02</td>
      <td>1.580623e-03</td>
      <td>9.941939e-01</td>
      <td>5.354905e-01</td>
      <td>6.841948e-02</td>
      <td>4.140663e-01</td>
      <td>3.541750e-04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.622858e-01</td>
      <td>4.052108e+02</td>
      <td>9.964608e+02</td>
      <td>2.825520e+01</td>
      <td>3.628387e-01</td>
      <td>4.910742e-01</td>
      <td>3.943464e-01</td>
      <td>4.260121e-01</td>
      <td>1.479713e-01</td>
      <td>2.281707e-01</td>
      <td>...</td>
      <td>1.918567e-01</td>
      <td>4.000162e-01</td>
      <td>2.564226e-01</td>
      <td>1.581579e-01</td>
      <td>3.972563e-02</td>
      <td>7.597622e-02</td>
      <td>4.987390e-01</td>
      <td>2.524645e-01</td>
      <td>4.925603e-01</td>
      <td>1.881621e-02</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.500000e+01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000e+00</td>
      <td>4.910000e+02</td>
      <td>2.750000e+02</td>
      <td>3.400000e+01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000e+00</td>
      <td>6.450000e+02</td>
      <td>4.750000e+02</td>
      <td>4.800000e+01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000e+00</td>
      <td>8.930000e+02</td>
      <td>9.000000e+02</td>
      <td>6.800000e+01</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000e+00</td>
      <td>1.161000e+04</td>
      <td>1.000000e+05</td>
      <td>2.149000e+03</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>...</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>



# Target and Features


```python
#Target
y=cleaneduploans['fol']
```


```python
y.shape
```




    (1177384,)




```python
#Features
X=cleaneduploans.drop('fol', axis=1)
```


```python
X.shape
```




    (1177384, 25)




```python
for i in X.isnull():
    if i == True:
        print (i)
print('no missing values')
```

    no missing values


# Feauture selection 


```python
from yellowbrick.features import Rank2D

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=X.columns, algorithm='pearson')

visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data
```


![image](https://github.com/yavah/yavah.github.io/tree/master/assets/img/m1.png)



```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge, Lasso, ElasticNet
```


```python
model = Lasso(alpha=0.5)
sfm = SelectFromModel(model)
sfm.fit(X, y)
print(list(X.iloc[:, sfm.get_support(indices=True)]))
```

    ['description_length', 'loan_amount', 'loan_use_length']



```python
model = Ridge(alpha=0.5)
sfm = SelectFromModel(model)
sfm.fit(X, y)
print(list(X.iloc[:, sfm.get_support(indices=True)]))
```

    ['tags_exist', 'num_borrowers_female_pct', 'sector_name_Arts', 'sector_name_Education', 'sector_name_Health', 'sector_name_Manufacturing', 'sector_name_Personal Use', 'distribution_model_field_partner']



```python
model = ElasticNet()
sfm = SelectFromModel(model)
sfm.fit(X, y)
print(list(X.iloc[:, sfm.get_support(indices=True)]))
```

    ['description_length', 'loan_amount', 'loan_use_length']



```python
from yellowbrick.features import Rank1D

visualizer = Rank1D(features=X.columns, algorithm='shapiro')

visualizer.fit(X, y)                
visualizer.transform(X)         
visualizer.poof()                     
```


![image](https://github.com/yavah/yavah.github.io/tree/master/assets/img/m2.png)



```python
from yellowbrick.features.importances import FeatureImportances
from sklearn.linear_model import LogisticRegression

fig = plt.figure()
ax = fig.add_subplot()

viz = FeatureImportances(LogisticRegression(), ax=ax)
viz.fit(X, y)
viz.poof()
```


![image](https://github.com/yavah/yavah.github.io/tree/master/assets/img/m3.png)



```python
from sklearn.ensemble import GradientBoostingClassifier

fig = plt.figure()
ax = fig.add_subplot()

viz = FeatureImportances(GradientBoostingClassifier(), ax=ax)
viz.fit(X, y)
viz.poof()
```


![image](https://github.com/yavah/yavah.github.io/tree/master/assets/img/m4.png)


# Train and Test


```python
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
```


```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=1)
```


```python
#scaler=StandardScaler()
#X_train=scaler.fit_transform(X_train)
#X_test=scaler.transform(X_test)
```

# Check balance in train and test data


```python
y.value_counts(normalize=True)
```




    1    0.535491
    0    0.464509
    Name: fol, dtype: float64




```python
c.value_counts(normalize=True)
```




    True     0.929635
    False    0.070365
    Name: loan_amount, dtype: float64




```python
y_train.value_counts(normalize=True)
```




    1    0.535681
    0    0.464319
    Name: fol, dtype: float64




```python
y_test.value_counts(normalize=True)
```




    1    0.534727
    0    0.465273
    Name: fol, dtype: float64



# Model 1: Naive Approach


```python
from sklearn.naive_bayes import GaussianNB
```


```python
nb = GaussianNB()
```


```python
nb.fit(X_train, y_train)
```




    GaussianNB(priors=None)




```python
expected   = y_test
predicted  = nb.predict(X_test)
classificationReport = classification_report(expected, predicted, target_names=['FOL','MTF'])
print(classificationReport)
```

                 precision    recall  f1-score   support
    
            FOL       0.54      0.76      0.63    109561
            MTF       0.67      0.44      0.53    125916
    
    avg / total       0.61      0.59      0.58    235477
    


# Model 2: Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
```


```python
lr = LogisticRegression(C=0.01)
```


```python
lr.fit(X_train, y_train)
```




    LogisticRegression(C=0.01, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
expected   = y_test
predicted  = lr.predict(X_test)
classificationReport = classification_report(expected, predicted, target_names=['FOL','MTF'])
print(classificationReport)
```

                 precision    recall  f1-score   support
    
            FOL       0.62      0.63      0.63    109561
            MTF       0.67      0.67      0.67    125916
    
    avg / total       0.65      0.65      0.65    235477
    



```python
from yellowbrick.classifier import ClassificationReport
fig = plt.figure()
ax = fig.add_subplot()
visualizer = ClassificationReport(lr, ax=ax, classes=['FOL', 'MTF'], support=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()
```


![image](https://github.com/yavah/yavah.github.io/tree/master/assets/img/m5.png)



```python
from yellowbrick.classifier import ROCAUC

visualizer = ROCAUC(LogisticRegression(), classes=['FOL', 'MTF'])

visualizer.fit(X_train, y_train)  
visualizer.score(X_test, y_test)
g = visualizer.poof()
```


![image](https://github.com/yavah/yavah.github.io/tree/master/assets/img/m6.png)



```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
lr=LogisticRegression()
params={'C':np.logspace(-4, 4, 5)}
clf = GridSearchCV(lr, params, scoring='neg_log_loss', refit='True', n_jobs=-1, cv=5)
clf.fit(X_train, y_train)
print("best params: " + str(clf.best_params_))
print("best scores: " + str(clf.best_score_))
estimates = clf.predict_proba(X_test)
acc = accuracy_score(y_test, clf.predict(X_test))
print("Accuracy: {:.4%}".format(acc))
```

    best params: {'C': 0.01}
    best scores: -0.6305215984299067
    Accuracy: 65.0879%


# Model 3: Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
rf = RandomForestClassifier(n_estimators=100)
```


```python
rf.fit(X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)




```python
expected   = y_test
predicted  = rf.predict(X_test)
classificationReport = classification_report(expected, predicted, target_names=['FOL','MTF'])
print(classificationReport)
```

                 precision    recall  f1-score   support
    
            FOL       0.62      0.60      0.61    109561
            MTF       0.66      0.68      0.67    125916
    
    avg / total       0.65      0.65      0.65    235477
    


# Model 4: SVM


```python
from sklearn.svm import LinearSVC
```


```python
svc = LinearSVC()
```


```python
svc.fit(X_train, y_train)
```




    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
         verbose=0)




```python
expected   = y_test
predicted  = svc.predict(X_test)

classificationReport = classification_report(expected, predicted, target_names=['FOL','MTF'])
print(classificationReport)
```

                 precision    recall  f1-score   support
    
            FOL       0.47      1.00      0.64    109561
            MTF       0.88      0.00      0.00    125916
    
    avg / total       0.69      0.47      0.30    235477
    


# Model 5: LDA


```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
```


```python
lda = LDA(n_components=2)
```


```python
lda.fit(X_train,y_train)
```




    LinearDiscriminantAnalysis(n_components=2, priors=None, shrinkage=None,
                  solver='svd', store_covariance=False, tol=0.0001)




```python
expected   = y_test
predicted  = lda.predict(X_test)
classificationReport = classification_report(expected, predicted, target_names=['FOL','MTF'])
print(classificationReport)
```

                 precision    recall  f1-score   support
    
            FOL       0.62      0.63      0.63    109561
            MTF       0.68      0.67      0.67    125916
    
    avg / total       0.65      0.65      0.65    235477
    


# Model 6: Gradient Boosting


```python
from sklearn.ensemble import GradientBoostingClassifier
```


```python
gbc = GradientBoostingClassifier()
```


```python
gbc.fit(X_train,y_train)
```




    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=2,
                  min_weight_fraction_leaf=0.0, n_estimators=100,
                  presort='auto', random_state=None, subsample=1.0, verbose=0,
                  warm_start=False)




```python
expected = y_test
predicted  = gbc.predict(X_test)
classificationReport = classification_report(expected, predicted, target_names=['FOL','MTF'])
print(classificationReport)
```

                 precision    recall  f1-score   support
    
            FOL       0.64      0.64      0.64    109561
            MTF       0.69      0.69      0.69    125916
    
    avg / total       0.66      0.66      0.66    235477
    



```python
from yellowbrick.classifier import ClassificationReport
fig = plt.figure()
ax = fig.add_subplot()
visualizer = ClassificationReport(gbc, ax=ax, classes=['FOL', 'MTF'], support=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()
```


![image](https://github.com/yavah/yavah.github.io/tree/master/assets/img/m7.png)


# Model 7: MLP


```python
from sklearn.neural_network import MLPClassifier
```


```python
mlp=MLPClassifier(alpha=1)
```


```python
mlp.fit(X_train,y_train)
```




    MLPClassifier(activation='relu', alpha=1, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)




```python
expected = y_test
predicted  = mlp.predict(X_test)
classificationReport = classification_report(expected, predicted, target_names=['FOL','MTF'])
print(classificationReport)
```

                 precision    recall  f1-score   support
    
            FOL       0.62      0.63      0.63    109561
            MTF       0.67      0.66      0.67    125916
    
    avg / total       0.65      0.65      0.65    235477
    


# Model 8: Bagging


```python
from sklearn.ensemble import BaggingClassifier
```


```python
bc=BaggingClassifier(n_estimators=100, oob_score=10)
```


```python
bc.fit(X_train,y_train)
```




    BaggingClassifier(base_estimator=None, bootstrap=True,
             bootstrap_features=False, max_features=1.0, max_samples=1.0,
             n_estimators=100, n_jobs=1, oob_score=10, random_state=None,
             verbose=0, warm_start=False)




```python
expected = y_test
predicted  = bc.predict(X_test)
classificationReport = classification_report(expected, predicted, target_names=['FOL','MTF'])
print(classificationReport)
```

                 precision    recall  f1-score   support
    
            FOL       0.62      0.61      0.61    109561
            MTF       0.66      0.68      0.67    125916
    
    avg / total       0.64      0.65      0.64    235477
    

