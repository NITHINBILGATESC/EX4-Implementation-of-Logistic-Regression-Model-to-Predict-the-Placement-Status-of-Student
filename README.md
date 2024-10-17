# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student
## DATE:

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook 

## Algorithm
1. We will use pandas for data manipulation, scikit-learn for building and evaluating the logistic regression model, and matplotlib/seaborn for data visualization.
2. Let’s assume you have a dataset with columns such as age, cgpa, internships, aptitude_score, and the target column placement_status (1 = Placed, 0 = Not Placed).
3. Before training the model, we need to preprocess the data. This includes handling missing values, encoding categorical variables, scaling the features, and splitting the data into features (X) and target (y).
4. We will initialize the Logistic Regression model and fit it to the training data.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NITHIN BILGATES C
RegisterNumber: 2305001022 
*/
import pandas as pd
data=pd.read_csv("/content/ex45Placement_Data (1).csv")
data.head()

data1=data.copy()
data1.head()

data1=data1.drop(['sl_no','salary'],axis=1)
data1

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1['gender']=le.fit_transform(data1['gender'])
data1['ssc_b']=le.fit_transform(data1['ssc_b'])
data1['hsc_b']=le.fit_transform(data1['hsc_b'])
data1['hsc_s']=le.fit_transform(data1['hsc_s'])
data1['degree_t']=le.fit_transform(data1['degree_t'])
data1['workex']=le.fit_transform(data1['workex'])
data1['specialisation']=le.fit_transform(data1['specialisation'])
data1['status']=le.fit_transform(data1['status'])
data1

x=data1.iloc[:,:-1]
x

y=data1.iloc[:,-1]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
y_pred,x_test

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy_score)
print("\nConfusion matrix:\n",confusion)
print("\nClassification report:\n",cr)

from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
```
## Output:

![image](https://github.com/user-attachments/assets/66102de6-f024-44a0-90e4-33a22eeef6b2)
![image](https://github.com/user-attachments/assets/92912e6e-bc02-4ebd-ac7c-fbae2fda4047)
![image](https://github.com/user-attachments/assets/022c5e38-a65f-44f9-bf47-f66400e7d3d1)
![image](https://github.com/user-attachments/assets/cd241534-ac32-40f4-9516-c866deb17f43)
![image](https://github.com/user-attachments/assets/24143369-c87f-4d81-9d28-52c83b62281f)
![image](https://github.com/user-attachments/assets/16f5b5a0-94df-4390-b06d-32c69ae1a775)
![image](https://github.com/user-attachments/assets/c257f7e6-80ac-4811-b808-8b11ec68031c)
![image](https://github.com/user-attachments/assets/487a468b-7cb1-4277-8b75-10c91b579f6a)
![image](https://github.com/user-attachments/assets/c33584e6-644b-4266-bbf0-b4df9d7c05c2)
![image](https://github.com/user-attachments/assets/6a445630-1099-4045-acdb-64969e8b59ce)
![image](https://github.com/user-attachments/assets/6e9f78ae-bad5-4648-b2ef-598970220031)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
