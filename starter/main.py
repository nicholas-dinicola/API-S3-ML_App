# Put the code for your API here.
import numpy as np
import pandas as pd
from fastapi import FastAPI
from typing import Union, List
from pydantic import BaseModel
import uvicorn
from joblib import load
from sklearn.preprocessing import OneHotEncoder, StandardScaler

classifier = load("./model/my_model.joblib")

# instantiate the app
app = FastAPI()


class ClassifierFeatureIn(BaseModel):
    age: Union[int, float]
    workclass: str
    fnlwgt: Union[int, float]
    education: str
    education_num: Union[int, float]
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: Union[int, float]
    capital_loss: Union[int, float]
    hours_per_week: Union[int, float]
    native_country: str


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
