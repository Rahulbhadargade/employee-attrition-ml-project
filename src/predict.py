import joblib
import pandas as pd

model = joblib.load("outputs/attrition_model.pkl")

def predict_employee(data):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    if prediction[0] == 1:
        print("Employee likely to leave")
    else:
        print("Employee likely to stay")