import pytest
from fastapi.testclient import TestClient
import json
import sys

sys.path.insert(0, 'starter/demo/ml')
sys.path.insert(1, 'starter/model')
from starter.main import app

client = TestClient(app)


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == ["Hello User!"]


def test_path_two():
    r = client.get("/MyName")
    assert r.status_code == 200
    assert r.json() == ["Welcome to this app, MyName"]


def test_post_less_than_fifty():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    headers = {"Content-Type": "application/json"}

    r = client.post("/predict", data=json.dumps(data), headers=headers)
    assert r.status_code == 200
    assert r.json()["prediction"] == " <=50K", r"Prediction not expected"


# @pytest.mark.skip(reason="Pass for now")
def test_post_more_than_fifty():
    data = {
        "age": 50,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 500000,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    headers = {"Content-Type": "application/json"}

    r = client.post("/predict", data=json.dumps(data), headers=headers)
    assert r.status_code == 200
    assert r.json()["prediction"] == " >50K", r"Prediction not expected"



# PYTHONPATH=./starter pytest -v
