import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = pd.read_csv("diabetes.csv")
print(data)

x = data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
y = data["Outcome"]
dt = LogisticRegression()
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25)
dt.fit(xtrain,ytrain)
ypred = dt.predict(xtest)
print(ypred)


acc = accuracy_score(ytest,ypred)
print(acc)

dt = LogisticRegression()
dt.fit(x,y)
# ypred = dt.predict([[8,176,90,34,300,33.7,0.467,58]])
# print(ypred)

print("Hello, Welcome to diabates detection system")
b= []
a = int(input("1. Pregnancies: "))
b.append(a)

a = int(input("2. Glucose: "))
b.append(a)


a = int(input("3. BloodPressure: "))
b.append(a)


a = int(input("4.SkinThickness: "))
b.append(a)


a = int(input("5. Insulin: "))
b.append(a)


a = float(input("6.BMI: "))
b.append(a)


a = float(input("7.DiabetesPedigreeFunction: "))
b.append(a)


a = int(input("8. Age: "))
b.append(a)

ypred= dt.predict([b])
print(ypred)
if ypred==1:
    print("Patient Suffers from Diabetes")
else:
    print("No Diabetes Detected")









