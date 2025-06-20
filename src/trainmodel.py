# github.com/hussrepo
import pandas as panda
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

#Config/Loading Data
csv = "dataset/heart_disease.csv"
targetColumn = "target"
print("Loading Dataset...")
data = panda.read_csv(csv)
data = data[data["chol"] != 0]

#Set Target
if targetColumn not in data.columns:
    raise ValueError(f"Error 1: Target column '{targetColumn}' not found...")
x = data.drop(columns=[targetColumn])
y = data[targetColumn]
print("Columns That Will be used for training:", list(x.columns))

#TrainTestSplit
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Train
heartDiseaseModel = LogisticRegression(max_iter=1000)
heartDiseaseModel.fit(x_train, y_train)

#Eval the Model
print("Model Evaluation:")
y_predict = heartDiseaseModel.predict(x_test)
print(classification_report(y_test, y_predict))
heartDiseaseModel_accuracy = heartDiseaseModel.score(x_test, y_test)
print(f"Model Accuracy: {round(heartDiseaseModel_accuracy * 100, 2)}%")

#Save Model
joblib.dump({"heartDiseaseModel": heartDiseaseModel, "heartDiseaseModel_accuracy": heartDiseaseModel_accuracy}, "heartDiseaseModel.pkl")
print("Model Trained and saved as 'heartDiseaseModel.pkl'")