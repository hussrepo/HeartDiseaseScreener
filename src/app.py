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
    "age": ("Age", "e.g., 24"),
    "sex": ("Sex", "1 = Male, 0 = Female"),
    "cp": ("Chest Pain Type", "1 = Typical Angina, 2 = Atypical Angina, 3 = Non-anginal, 4 = Asymptomatic"),
    "restbps": ("Resting Blood Pressure (Systolic)", "e.g., 120 mmHg"),
    "chol": ("Serum Cholesterol", "e.g., 212 mg/dL"),
    "fbs": ("Fasting Blood Sugar > 120 mg/dL", "1 = Yes, 0 = No"),
    "restecg": ("Resting ECG Result", "0 = Normal, 1 = ST-T Abnormality, 2 = LV Hypertrophy"),
    "maxhr": ("Maximum Heart Rate Achieved", "e.g., 150 bpm"),
    "exang": ("Exercise Induced Angina", "1 = Yes, 0 = No"),
    "stdep": ("ST Depression (Exercise)", "e.g., 0.5; 0 = None"),
    "slope": ("Slope of Peak Exercise ST Segment", "1 = Upsloping, 2 = Flat, 3 = Downsloping"),
    "ca": ("Major Vessels Colored by Fluoroscopy", "0–3; e.g., 1; 0 = None"),
    "thal": ("Thalassemia Test Result", "3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect"),
}

#GUI
app = tki.Tk()
app.title("Heart Disease Screener")
app.geometry("800x625")
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
    # Create Graphical Window
    global visualWindow
    visualWindow = tki.Toplevel(app)
    visualWindow.title("Model Training Information")
    visualWindow.configure(bg="#23272F")
    visualWindow.geometry("1150x500")
    visualWindow.resizable(True, True)

    # Header Frame for Title
    header_frame = tki.Frame(visualWindow, bg="#23272F")
    header_frame.pack(fill="x", pady=(20, 10))
    tki.Label(
        header_frame,
        text="Training Data Visualizations",
        font=("Segoe UI", 22, "bold"),
        bg="#23272F",
        fg="#FF5252",
        pady=8
    ).pack()
    tki.Label(
        header_frame,
        text="Visual insights from the model's training dataset",
        font=("Segoe UI", 12, "italic"),
        bg="#23272F",
        fg="#FFD369"
    ).pack()

    # Main Content Frame
    content_frame = tki.Frame(visualWindow, bg="#23272F")
    content_frame.pack(expand=True, fill="both", padx=30, pady=10)

    # Load and display images in a grid with padding and border
    imgRef = []
    imgFiles = ["cholheartdis.png", "datasexratio.png", "maxhrfreq.png"]
    captions = \
        [
        "Cholesterol vs Heart Disease",
        "Sex Ratio in Dataset",
        "Max Heart Rate Frequency"
        ]
    for i, imgFile in enumerate(imgFiles):
        imagePath = getPath(f"datavisualization/{imgFile}")
        image = Image.open(imagePath)
        image = image.resize((320, 240), Image.Resampling.LANCZOS)
        imagetk = ImageTk.PhotoImage(image)
        imgRef.append(imagetk)
        img_frame = tki.Frame(content_frame, bg="#393E46", bd=2, relief="ridge")
        img_frame.grid(row=0, column=i, padx=18, pady=8)
        tki.Label(img_frame, image=imagetk, bg="#393E46").pack()
        tki.Label(img_frame, text=captions[i], font=("Segoe UI", 10, "italic"), bg="#393E46", fg="#FFD369").pack(pady=(6, 2))

    visualWindow.imgRef = imgRef

    # Button Frame
    button_frame = tki.Frame(visualWindow, bg="#23272F")
    button_frame.pack(pady=(10, 20))
    tki.Button(button_frame, text="Show Model Accuracy", width=25, height=1, font=("Segoe UI", 12, "bold"), command=showAccuracy, bg="#0074D9", fg="white", activebackground="#005fa3", activeforeground="#FFD369", bd=0, relief="ridge").pack()


#Input Fields Font Style
headers = ("Arial", 12, "bold")
labels = ("Arial", 10, "bold")
notes = ("Arial", 8)

# App Headers - Improved Appearance/Alignment
header_frame = tki.Frame(app, bg="#23272F", bd=3, relief="groove", highlightbackground="#393E46", highlightthickness=2)
header_frame.grid(row=0, column=0, columnspan=3, pady=(20, 10), padx=30, sticky="ew")

appHeader = tki.Label(header_frame, text="❤ Heart Disease Screener ❤", font=("Segoe UI", 26, "bold"), bg="#23272F", fg="#FF5252", pady=0, anchor="center", justify="center", bd=0, highlightthickness=0)
appHeader.pack(fill="x")

appSubHeader = tki.Label(header_frame, text="Not Intended for Medical Usage\ngithub.com/hussrepo", font=("Segoe UI", 11, "italic"), bg="#23272F", fg="#FFD369", pady=0, anchor="center", justify="center", bd=0, highlightthickness=0)
appSubHeader.pack(fill="x")


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
tki.Button(app, text="Calculate Probability", command=prediction, font=("Segoe UI", 12, "bold"), bg="#FF5252", fg="white", activebackground="#005fa3", activeforeground="#FFD369", bd=0, relief="ridge", padx=10, width=25, height=1).grid(row=len(variables) + 2, column=0, columnspan=3, pady=15)
tki.Button(app, text="Model Training Information", command=trainingDataInfo, font=("Segoe UI", 12, "bold"), bg="#0074D9", fg="white", activebackground="#005fa3", activeforeground="#FFD369", bd=0, relief="ridge", padx=10, width=25, height=1).grid(row=len(variables) + 3, column=0, columnspan=3, pady=0)
app.mainloop()