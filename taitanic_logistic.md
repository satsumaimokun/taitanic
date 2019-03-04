

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
```


```python
#データの読み込み(欠損値はnp.nanとして読み込む）
dataframe_train=pd.read_csv("/home/2240374924/Downloads/train.csv",na_values=[np.nan])
```


```python
#passengerIdをindexにする。
dataframe_train=dataframe_train.set_index(dataframe_train["PassengerId"])
```


```python
#不要な列の削除
dataframe_train=dataframe_train.drop(["Name","Embarked","PassengerId","Ticket","Cabin"],axis=1)
```


```python
#Survivedをtargetに設定。
dataframe_train_target=dataframe_train["Survived"]
```


```python
#残りをfeaturesを設定
dataframe_train_features=dataframe_train.drop("Survived",axis=1)
```


```python
#KNNが使えないのでAgeの欠損値をmedianで補完
Age_median=dataframe_train_features[dataframe_train_features["Age"].notnull()]["Age"].median()
dataframe_train_features["Age"]=dataframe_train_features["Age"].replace(np.nan,Age_median)
```


```python
#Sexをmaleを0,femaleを1に置き換える。
dataframe_train_features=dataframe_train_features.replace(["male","female"],[0,1])
```


```python
#Sibsp+Parch=Familyという引数を設定。
#dataframe_train_features["Family"]=dataframe_train_features["SibSp"]+dataframe_train_features["Parch"]
#dataframe_train_features=dataframe_train_features.drop(["SibSp","Parch"],axis=1)
```


```python
#SibSpの標準化
#SibSp_standardizer=scaler.fit(dataframe_train_features["SibSp"])
#standardized_SibSp=SibSp_standardizer.transform(dataframe_train_features["SibSp"])
#dataframe_train_features["SibSp"]=standardized_SibSp
```


```python
#Parchの標準化
#Parch_standardizer=scaler.fit(dataframe_train_features["Parch"])
#standardized_Parch=Parch_standardizer.transform(dataframe_train_features["Parch"])
#dataframe_train_features["Parch"]=standardized_Parch
```


```python
#Familyの標準化
#Family_standardizer=scaler.fit(dataframe_train_features["Family"])
#standardized_Family=Family_standardizer.transform(dataframe_train_features["Family"])
#dataframe_train_features["Family"]=standardized_Family
```


```python
#Ageの標準化
Age_standardizer=scaler.fit(dataframe_train_features["Age"])
standardized_Age=Age_standardizer.transform(dataframe_train_features["Age"])
dataframe_train_features["Age"]=standardized_Age
```


```python
#Fareの標準化
Fare_standardizer=scaler.fit(dataframe_train_features["Fare"])
standardized_Fare=Fare_standardizer.transform(dataframe_train_features["Fare"])
dataframe_train_features["Fare"]=standardized_Fare
```


```python
#探索器の作成
logistic_regression=LogisticRegressionCV(penalty="l2",Cs=100,random_state=0,n_jobs=-1,)
```


```python
#二重交差検証による性能評価
#cross_val_score(logistic_regression,dataframe_train_features,dataframe_train_target).mean()
```


```python
#探索機の訓練
model=logistic_regression.fit(dataframe_train_features,dataframe_train_target)
```


```python
#testデータの読み込み(欠損値はnp.nanとして読み込む）
dataframe_test=pd.read_csv("/home/2240374924/Downloads/test.csv",na_values=[np.nan])
```


```python
#passengerIdをindexにする。
dataframe_test=dataframe_test.set_index(dataframe_test["PassengerId"])
```


```python
#不要な列の削除しtest_featuresを作成
dataframe_test_features=dataframe_test.drop(["Name","Embarked","PassengerId","Ticket","Cabin"],axis=1)
```


```python
#KNNが使えないのでAgeの欠損値をmedianで補完
dataframe_test_features["Age"]=dataframe_test_features["Age"].replace(np.nan,Age_median)
```


```python
#test_dataでFareがNaNになっているところがある
dataframe_test_features[dataframe_test_features["Fare"].isnull()]
```


```python
#test_dataのFareのNaNをPcalss=3の平均で置き換える。
Pclass3_Fare_mean=dataframe_test_features[dataframe_test_features["Fare"].notnull()][dataframe_test_features[dataframe_test_features["Fare"].notnull()]["Pclass"]==3]["Fare"].mean()
dataframe_test_features["Fare"]=dataframe_test_features["Fare"].replace(np.nan,Pclass3_Fare_mean)
```


```python
#Sexをmaleを0,femaleを1に置き換える。
dataframe_test_features=dataframe_test_features.replace(["male","female"],[0,1])
```


```python
#Ageの標準化
standardized_Age=Age_standardizer.transform(dataframe_test_features["Age"])
dataframe_test_features["Age"]=standardized_Age
```


```python
#Fareの標準化
standardized_Fare=Fare_standardizer.transform(dataframe_test_features["Fare"])
dataframe_test_features["Fare"]=standardized_Fare
```


```python
#ターゲットベクトルを予想
target_predicted=model.predict(dataframe_test_features)
```


```python
#提出用のcsvファイルを作成
dataframe_target_predicted=pd.DataFrame()
dataframe_target_predicted["PassengerId"]=range(892,1310)
dataframe_target_predicted["Survived"]=target_predicted
dataframe_target_predicted=dataframe_target_predicted.set_index(dataframe_target_predicted["PassengerId"])
dataframe_target_predicted=dataframe_target_predicted.drop(["PassengerId"],axis=1)
dataframe_target_predicted.to_csv("taitanic_target_prediction.csv")
```
