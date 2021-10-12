# Put the code for your API here.
import numpy as np
import pandas as pd
import os
from fastapi import FastAPI
from typing import Union, List
from pydantic import BaseModel, Field
import uvicorn
from joblib import load
from demo.ml.data import process_data
from demo.ml.model import inference, load_from_file

# instantiate the app
app = FastAPI()

# Give Heroku the ability to pull in data from DVC upon app start up.
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


# models with pydantic
class ClassifierFeatureIn(BaseModel):
    age: int = Field(..., example=50)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2500, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


# pydantic output of the model
class ClassifierOut(BaseModel):
    # The forecast output will be either >50K or <50K
    forecast: str = "Income > 50k"


# Define a GET for greetings.
@app.get("/")
async def greet_user():
    return {"message": "Hello User!"}


# Greet the user with his/her name
@app.get("/{name}")
async def get_name(name: str):
    return {"Welcome to this app": f"{name}"}


@app.post("/predict", response_model=ClassifierOut, status_code=200)
async def predict(data1: ClassifierFeatureIn):
    data = pd.DataFrame.from_dict(data1.dict(by_alias=True))

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

    # Load the preprocessors and the classifier
    encoder = load_from_file('model/encoder')
    classifier = load_from_file('model/classsifer')
    lb = load_from_file('model/labelbinarizer')
    scaler = load_from_file('model/scaler')

    # Preprocess the data
    X, _, _, _, _ = process_data(
        data, categorical_features=cat_features, encoder=encoder, lb=lb, scaler=scaler, training=False
    )

    # Predict salary
    preds = inference(classifier, X.reshape(1, 108))

    # convert output from 0,1 to <50k,>50k
    preds = loaded_lb.inverse_transform(preds)

    return {
        "prediction": preds[0]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
