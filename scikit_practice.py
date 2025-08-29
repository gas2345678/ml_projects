import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score




df = pd.read_csv(r"c:\Users\PC\Downloads\Titanic-Dataset.csv")

df = pd.DataFrame(df)


df = df.dropna()


print(df.head())

print(df.shape)

#Dataframe overview


print("Key aspects")
print(df.describe())
print(df.info())


#Change sex values to 0 and 1

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

#Use the important columns for the analysis

x=  df.drop(columns=["Ticket","Cabin","Fare","Name","PassengerId","SibSp","Parch","Age","Survived","Embarked"])

y=df["Survived"]



#Divide our train and test data 


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)





print("Training data")
print(x_train.head())
print("Testing data")
print(x_test.head())

#Create an instance of the kNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)


# Fit the model with the  training data

knn.fit(x_train,y_train)


obj_predicted = knn.predict(x_test)



print("Results of predictions:")
print(obj_predicted)


print("Accuracy of the model:")
print(accuracy_score(y_test,obj_predicted))


























