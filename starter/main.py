# Put the code for your API here.
import numpy as np
import pandas as pd
from fastapi import FastAPI
from typing import Union, List
from pydantic import BaseModel, Field
import uvicorn
from joblib import load
from sklearn.preprocessing import OneHotEncoder, StandardScaler

classifier = load("./model/my_model.joblib")

# instantiate the app
app = FastAPI()


class ClassifierFeatureIn(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


# Define a GET for greetings.
@app.get("/")
async def greet_user():
    return {"message": "Hello User!"}


# Greet the user with his/her name
@app.get("/{name}")
async def get_name(name: str):
    return {"Welcome to this app": f"{name}"}


@app.post("/predict")
async def predict(data1: ClassifierFeatureIn):
    data = data1.dict()
    data = pd.DataFrame(data, index=[0])
    data = data.rename(columns={
        "capital_gain": "capital-gain",
        "capital_loss": "capital-loss",
        "hours_per_week": "hours-per-week",
        "native_country": "native-country",
        "marital_status": "marital-status"
    })

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_categorical = data[cat_features].values
    X_continuous = data.drop(*[cat_features], axis=1)

    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    scaler = StandardScaler()

    X_categorical = encoder.fit_transform(X_categorical)
    X_continuous = scaler.fit_transform(X_continuous)
    X = np.concatenate([X_continuous, X_categorical], axis=1)

    preds = classifier.predict(np.array(X))

    return {
        "prediction": preds
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
