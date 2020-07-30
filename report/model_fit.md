

```python
# Import packages: 

import feather
import pandas as pd
import numpy as np

# sklearn :
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import random
%matplotlib inline

```

### Read Data from Feather


```python
dat = feather.read_dataframe('../data/sub_data.feather')
dat.head(5)
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
      <th>racenum</th>
      <th>pos</th>
      <th>hnum</th>
      <th>odds</th>
      <th>date</th>
      <th>name</th>
      <th>driver</th>
      <th>trainer</th>
      <th>seconds</th>
      <th>temp</th>
      <th>cond</th>
      <th>winner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>2.60</td>
      <td>2015-11-23</td>
      <td>Ryder</td>
      <td>Asher</td>
      <td>Quincy</td>
      <td>116.2</td>
      <td>24.0</td>
      <td>FT</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>4.85</td>
      <td>2015-11-23</td>
      <td>Ashlee</td>
      <td>Zane</td>
      <td>Carol</td>
      <td>117.2</td>
      <td>24.0</td>
      <td>FT</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>4.00</td>
      <td>2015-11-23</td>
      <td>Carmen</td>
      <td>Theresa</td>
      <td>Brian</td>
      <td>117.4</td>
      <td>24.0</td>
      <td>FT</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>28.60</td>
      <td>2015-11-23</td>
      <td>Rowland</td>
      <td>Taryn</td>
      <td>Quincy</td>
      <td>117.0</td>
      <td>24.0</td>
      <td>FT</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>0.30</td>
      <td>2015-11-23</td>
      <td>Noe</td>
      <td>Theresa</td>
      <td>Braylon</td>
      <td>118.0</td>
      <td>24.0</td>
      <td>FT</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
le = LabelEncoder()
```


```python
enc_cols = ['date', 'name', 'driver', 'trainer', 'cond']
exp_cols = ['date', 'name', 'driver', 'trainer', 'cond', 'temp', 'hnum']
```


```python
X = dat[exp_cols]    # these are the features
y = dat['winner']    # these are the labels (1: win 0: not win)
y_multi = dat['pos'] # these are the labels for multinomial logistic regression
X[enc_cols] = X[enc_cols].apply(le.fit_transform) # encode the string value columns
```

    /anaconda/lib/python3.6/site-packages/pandas/core/frame.py:2540: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self[k1] = value[k2]


### Fit Logistic Regression Model:


```python
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()

model.fit(X_train, y_train)
#multi_class=multinomial

print("Logistic Regression training error: %f" % (1.0-model.score(X_train,y_train)))
print("Logistic Regression test error: %f" % (1.0-model.score(X_validation,y_validation)))
```

    Logistic Regression training error: 0.175367
    Logistic Regression test error: 0.198697



```python
y_pred = model.predict(X_train)
```


```python
print("Predicted values are all 0 as the following ratio is", len(y_pred[y_pred == 0])/len(y_pred))
```

    Predicted values are all 0 as the following ratio is 1.0



```python
print("Not winner percentage in training set", round(len(y_train[y_train == 0])/len(y_train),2))
```

    Not winner percentage in training set 0.82


### Fit Multiclass Logistic Regression Model:


```python
X_train, X_validation, y_train, y_validation = train_test_split(X, y_multi, test_size=0.3)

model_multi = LogisticRegression(multi_class='multinomial', solver = 'lbfgs')

model_multi.fit(X_train, y_train)
#multi_class=multinomial

print("Multinomial Logistic Regression training error: %f" % (1.0-model_multi.score(X_train,y_train)))
print("Multinomial Logistic Regression test error: %f" % (1.0-model_multi.score(X_validation,y_validation)))
```

    Multinomial Logistic Regression training error: 0.798695
    Multinomial Logistic Regression test error: 0.843478



```python
y_pred_multi = model_multi.predict(X_train)
```


```python
print("Predicted winner as a percentage:", round(len(y_pred_multi[y_pred_multi == 1])/len(y_pred_multi),2))
print("Winner percentage in training set:", round(len(y_train[y_train == 1])/len(y_train),2))
```

    Predicted winner as a percentage: 0.24
    Winner percentage in training set: 0.17

