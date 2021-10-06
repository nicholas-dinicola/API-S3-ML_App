# Put the code for your API here.
from fastapi import FastAPI
from typing import Union, List
from pydantic import BaseModel
import uvicorn
from joblib import load

classifier = load("/model/my_model.joblib")
# instantiate the app
app = FastAPI()


class Classifier(BaseModel):
    age: Union[int, float]
    workclass: str
    fnlwgt: Union[int, float]
    education: str
    education_num: Union[int, float]
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: Union[int, float]
    capital_loss: Union[int, float]
    hours_per_week: Union[int, float]
    native_country: Union[int, float]


# Define a GET for greetings.
@app.get("/")
async def greet_user():
    return {"message": "Hello User!"}


# Greet the user with his/her name
@app.get("/{name}")
async def get_name(name: str):
    return {"Welcome to this app": f"{name}"}


@app.post("/predict")
async def predict(data: Classifier):
    return data
