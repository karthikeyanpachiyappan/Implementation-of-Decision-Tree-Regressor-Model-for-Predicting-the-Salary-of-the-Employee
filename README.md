# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

8.Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KARTHIKEYAN P
RegisterNumber: 212223230102
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

data=pd.read_csv(r"C:\Users\astle\Downloads\Salary.csv")
data.head()
data.info()
data.isnull().sum()
print(data)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print(le)
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
print(x_train)
print(x_test)
print(y_train)
print(y_test)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(y_pred)
r2=metrics.r2_score(y_test,y_pred)
print(r2)
dt.predict([[5,6]])
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show() 
*/
```

## Output:
Data:

![image](https://github.com/user-attachments/assets/48ed97be-c1f6-4ac2-a8e2-b82b55fd8b56)

x_test:

![image](https://github.com/user-attachments/assets/6829adc5-efc0-4ec5-8d96-7058a19a921e)

y_test

![image](https://github.com/user-attachments/assets/4b705cd4-5eeb-48c6-b1cc-386404cef92f)

y_pred

![Screenshot 2025-04-23 042131](https://github.com/user-attachments/assets/29a0442f-5ab5-4f49-a567-cd02dda77b34)

r2

![image](https://github.com/user-attachments/assets/45fdf366-3306-465a-bd1b-00c2360c96b5)

plt.show()

![image](https://github.com/user-attachments/assets/1136c315-d4c6-4701-8d35-ddc53dda7425)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
