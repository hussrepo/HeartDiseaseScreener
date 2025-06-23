# github.com/hussrepo
import tkinter as tki
import pandas as panda
from PIL import Image, ImageTk
import os
import sys
from tkinter import messagebox
import joblib

#Set Path Directory Function
def getPath(fileName):
    if getattr(sys, "frozen", False):
        path = sys._MEIPASS
    else:
        path = os.path.dirname(__file__)
    return os.path.join(path, fileName)

#Initalize Model Path
modelPath = getPath("heartDiseaseModel.pkl")

#Load Model
heartDiseaseModel_data = joblib.load(modelPath)
heartDiseaseModel = heartDiseaseModel_data["heartDiseaseModel"]
heartDiseaseModel_accuracy = heartDiseaseModel_data["heartDiseaseModel_accuracy"]

#Define Inputs in Human Readable/Understandable way
variables = {
    "age": ("Age", "(EX: 24)"),
    "sex": ("Sex", "(1 = Male, 0 = Female)"),
    "cp": ("Chest Pain Type", "(1 = Typical Angina, 2 = Atypical Angina, 3 = Non-anginal, 4 = Asymptomatic)"),
    "restbps": ("Resting Blood Pressure (Systolic)", "(EX: 120)"),
    "chol": ("Serum Cholesterol (mg/dl)", "(EX: 212)"),
    "fbs": ("Fasting Blood Sugar > 120 mg/dl", "(1 = True, 0 = False)"),
    "restecg": ("Resting ECG Result", "(0 = Normal, 1 = ST-T Wave Abnormality, 2 = Probable or Definite Left Ventricular Hypertrophy)"),
    "maxhr": ("Maximum Measured Heart Rate", "(EX: 150)"),
    "exang": ("Exercise Induced Angina", "(1 = True, 0 = False)"),
    "stdep": ("ST Depression Induced by Exercise (Decimal)", "(EX: 0.5), 0 = None"),
    "slope": ("Slope of the Peak Exercise ST Segment", "(1 = Upsloping, 2 = Flat, 3 = Downsloping), (No ST Dep. = 1)"),
    "ca": ("Number of Major Vessels Colored by Fluoroscopy (0-3)", "(EX: 1), 0 = None"),
    "thal": ("Thalassemia Test Result", "(3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect)"),
}

#GUI
app = tki.Tk()
app.title("Heart Disease Screener")
app.geometry("1000x600")
app.configure(bg="#36454F")
inputs = {}

#Predict Risk Function
def prediction():
    try:
        vals = [float(inputs[variable].get()) for variable in variables]
        inputPanda = panda.DataFrame([vals], columns=list(variables.keys()))
        probability = heartDiseaseModel.predict_proba(inputPanda)[0][1]
        binaryPrediction = heartDiseaseModel.predict(inputPanda)[0]
        if binaryPrediction == 1:
            binaryPrediction_text = "Heart Disease Present"
        elif binaryPrediction == 0:
            binaryPrediction_text = "Heart Disease Not Present"
        diseaseRiskPercent = probability * 100
        if diseaseRiskPercent < 40:
            riskType = "Low"
        elif diseaseRiskPercent < 65:
            riskType = "Medium"
        else:
            riskType = "High"
        msg = (f"There is a {probability * 100:.2f}% chance you have heart disease.\n" 
               f"Risk Type: {riskType}\n"
               f"Prediction: {binaryPrediction_text}")
        messagebox.showinfo("Risk Result", msg)
    except ValueError:
        messagebox.showerror("Error: Invalid User Input", "Please Enter Valid Values...")

#Import, Format, and Display the model accuracy
def showAccuracy():
    accuracy = round(heartDiseaseModel_accuracy * 100, 2)
    messagebox.showinfo("Imported Model Accuracy", f"Model Accuracy: {accuracy}%", parent=visualWindow)

#Display Training Data Information Function
def trainingDataInfo():
    #Create Graphical Window
    global visualWindow
    visualWindow = tki.Toplevel(app)
    visualWindow.title("Model Training Information")
    visualWindow.configure(bg="#36454F")
    visualWindow.geometry=("700x500")
    visualWindow.resizable(True, True)
    #Display Text
    tki.Label(visualWindow, text="Training Data Visualizations", font=("Arial", 18, "bold"), bg="#36454F", fg="red").grid(row=0, column=0, columnspan=3, pady=20, padx=20)
    #Load Images
    imgRef = []
    imgFiles = ["cholheartdis.png", "datasexratio.png", "maxhrfreq.png"]
    for i in range(3):
        imagePath = getPath(f"datavisualization/{imgFiles[i]}")
        image = Image.open(imagePath)
        image = image.resize((450, 350), Image.Resampling.LANCZOS)
        imagetk = ImageTk.PhotoImage(image)
        imgRef.append(imagetk)
        imagelabel = tki.Label(visualWindow, image=imagetk, bg="#36454F")
        imagelabel.grid(row=(i//3) + 1, column=i%3, padx=10, pady=10)
    visualWindow.imgRef = imgRef
    tki.Button(visualWindow, text="Import Model Accuracy", width=25, height=1, font=("Arial", 12, "bold"), command=showAccuracy, bg="blue", fg="white").grid(row=2, column=1,pady=20)



#Input Fields Font Style
headers = ("Arial", 12, "bold")
labels = ("Arial", 10, "bold")
notes = ("Arial", 8)

#App Headers
appHeader = tki.Label(app, text="Heart Disease Screener", font=("Arial", 18, "bold"), bg="#36454F", fg="red")
appHeader.grid(row=0, column=0, columnspan=3, pady=(10, 0))
appSubHeader = tki.Label(app, text="Not Intended for Medical Usage\ngithub.com/hussrepo", font=labels, bg="#36454F", fg="red")
appSubHeader.grid(row=1, column=0, columnspan=3, pady=(0, 10))

#Input Fields
for i, (variable, (label, note)) in enumerate(variables.items(), start=2):
    #Label
    tki.Label(app, text=label + ":", font=labels, fg="white", bg="#36454F").grid(row=i, column=0, sticky="e", padx=10, pady=4)
    #Input Field
    entry = tki.Entry(app, width=25)
    entry.grid(row=i, column=1, pady=4)
    inputs[variable] = entry
    #Examples
    tki.Label(app, text=note, font=notes, fg="white", bg="#36454F").grid(row=i, column=2, sticky="w")

#Calculate Button
tki.Button(app, text="Calculate Probability", command=prediction, font=headers, bg="red", fg="white", padx=10).grid(row=len(variables) + 2, column=0, columnspan=3, pady=20)
tki.Button(app, text="Show Model Training Information", command=trainingDataInfo, font=headers, bg="blue", fg="white", padx=10).grid(row=len(variables) + 3, column=0, columnspan=3, pady=0)
app.mainloop()