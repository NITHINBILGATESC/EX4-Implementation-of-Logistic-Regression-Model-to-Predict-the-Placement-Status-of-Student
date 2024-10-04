# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect data on various attributes that might influence placement
2. Identify and appropriately handle missing data through imputation or removal.
3. Scale features to ensure that they contribute equally to the model’s performance. This helps in speeding up the convergence of Gradient Descent.
4. Divide the data into training and testing subsets 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NITHIN BILGATES C
RegisterNumber:  2305001022
*/
```
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
## Output:
![image](https://github.com/user-attachments/assets/4313fe17-8b5e-459f-94d4-9b3f6e0b9aaf)
![image](https://github.com/user-attachments/assets/79654210-0a73-4fd2-b1c8-292310b23819)
![image](https://github.com/user-attachments/assets/65852e83-3886-4756-945a-ecec94fd171f)
![image](https://github.com/user-attachments/assets/85ac320e-45ab-474e-b4c7-efcae93a4164)
![image](https://github.com/user-attachments/assets/1565095e-635f-4fd8-a41a-6225f5b04bd5)
![image](https://github.com/user-attachments/assets/6ffb9a18-d1e0-41fa-b831-8bf6d89c508f)
![image](https://github.com/user-attachments/assets/284bf0eb-6169-42db-8ff9-c7b506656b3b)
![image](https://github.com/user-attachments/assets/142abed1-d4af-45e4-b126-207f80e16d5a)
![image](https://github.com/user-attachments/assets/7ddf2730-ce1b-4265-9a89-5a74ea1ce574)
![image](https://github.com/user-attachments/assets/37acbcc7-973d-41af-9a7a-10e51e4b22de)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
