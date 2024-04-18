# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SUDHARSANA KUMAR S R
RegisterNumber:  212223240162
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
## Placement data:
![image](https://github.com/rexlinrajan2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406566/0d65163e-89bd-4559-a827-b6c984a8b69c)
## Salary data:
![image](https://github.com/rexlinrajan2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406566/b50c9d81-07bf-4ea2-bd55-314ae5d7f113)
## Checking the null() function:
![image](https://github.com/rexlinrajan2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406566/27949900-92a0-468a-bb12-8dcb01a83ec0)
## Data duplicate:
![image](https://github.com/rexlinrajan2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406566/9eef8da7-cdf1-4ce6-86f0-9b6c587c6146)
## Print data:
![image](https://github.com/rexlinrajan2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406566/6ae4dffb-18b2-4b82-95dc-6c7f83b8c23d)
## Data-status:
![image](https://github.com/rexlinrajan2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406566/1810a3d9-e2d4-4dc5-8c46-0215a94b6f8d)
## Y_prediction array:
![image](https://github.com/rexlinrajan2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406566/91beecf1-7253-42fb-a1af-22205d0c25a5)
## Accuracy value:
![image](https://github.com/rexlinrajan2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406566/86e41762-900b-463d-814c-90a1d0e84355)
## Confusion array:
![image](https://github.com/rexlinrajan2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406566/5ec7c8f5-829c-4042-a9e1-a1c9b0595adb)
## Classification Report:
![image](https://github.com/rexlinrajan2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406566/2125bc90-ce93-4f7b-93b7-c6bf6786f51b)
## Prediction of LR:
![image](https://github.com/rexlinrajan2004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406566/024f856e-41c9-44aa-874b-c778bf7f1c28)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
