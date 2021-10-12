import pytest
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, './starter')
from starter.main import app

client = TestClient(app)


@pytest.fixture(scope='session')
def data_greater_than_fifty():
    data = {
        "age": 50,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "HS-grad",
        "education_num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2500,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    return data


@pytest.fixture(scope='session')
def data_less_than_fifty():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    return data


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == ["Hello User!"]


def test_path_two():
    r = client.get("/MyName")
    assert r.status_code == 200
    assert r.json() == ["Welcome to this app, MyName"]


@pytest.mark.skip(reason="Pass for now")
def test_post_more_than_fifty(data_greater_than_fifty):
    r = client.post("/predict/", json=data_greater_than_fifty)
    assert r.status_code == 200, r"Status code not ok"
    assert r.json() == ">50K", r"Prediction not expected"


@pytest.mark.skip(reason="Pass for now")
def test_post_less_than_fifty(data_less_than_fifty):
    r = client.post("/predict/", json=data_less_than_fifty)
    assert r.status_code == 200, r"Status code not ok"
    assert r.json() == "<=50K", r"Prediction not expected"


# PYTHONPATH=./starter pytest -v