# Import modules
import pandas as pd
import pytest
import os.path
import starter
import logging
from starter.demo.ml.data import process_data
from starter.demo.ml.model import train_model, compute_model_metrics, inference, save_to_file
from sklearn.model_selection import train_test_split

# root_dir = os.path.basename(os.path.abspath(starter.__file__))


# root_dir = os.path.abspath(starter.__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture()
def data():
    """
    Load the dataset used for the analysis
    Returns
    -------
    data: pd.DataFrame
    """
    # Add code to load in the data.
    # data = pd.read_csv(os.path.join(root_dir, "data", "census_no_spaces.csv"))
    data = pd.read_csv("./starter/data/census_no_spaces.csv")

    return data


def test_dataset(data):
    """
    Testing if the dataset is a pandas dataframe
    Returns
    -------

    """
    assert isinstance(data, pd.DataFrame), f"Dataset is not a pandas dataframe"
    assert data.shape == data.dropna().shape, f"Dataset has got missing values"


def test_process_data(data):
    """
    Testing train_model func,
    The model should be a RandomForestClassifier
    """
    # Split dataset into training and testing set
    train, test = train_test_split(data, test_size=0.20, random_state=42)

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

    X_train, y_train, encoder, lb, scaler = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, encoder, lb, scaler = process_data(
        test, categorical_features=cat_features, encoder=encoder, lb=lb, scaler=scaler, label="salary", training=False
    )

    assert X_train.shape[1] == 108, f"X_train: Expected 108 variables after encoding."
    assert X_test.shape[1] == 108, f"X_test: Expected 108 variables after encoding."
    assert y_train.shape[0] == 26048, f"y_train number of rows are different from expected."
    assert y_test.shape[0] == 6513, f"y_train number of rows are different from expected."


def test_train_model(data):
    # Split dataset into training and testing set
    train, test = train_test_split(data, test_size=0.20, random_state=42)

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

    X_train, y_train, encoder, lb, scaler = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    model = train_model(X_train, y_train)
    preds = model.predict(X_train)
    assert preds.shape[0] == y_train.shape[0], f"Number of predictions are different from expected."


def test_inference(data):
    # Split dataset into training and testing set
    train, test = train_test_split(data, test_size=0.20, random_state=42)

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

    X_train, y_train, encoder, lb, scaler = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, encoder, lb, scaler = process_data(
        test, categorical_features=cat_features, encoder=encoder, lb=lb, scaler=scaler, label="salary", training=False
    )

    model = train_model(X_train, y_train)
    preds = inference(model=model, X=X_test)
    assert preds.shape[0] == y_test.shape[0], f"Number of predictions are different from expected."


def test_compute_model_metrics(data):
    # Split dataset into training and testing set
    train, test = train_test_split(data, test_size=0.20, random_state=42)

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

    X_train, y_train, encoder, lb, scaler = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, encoder, lb, scaler = process_data(
        test, categorical_features=cat_features, encoder=encoder, lb=lb, scaler=scaler, label="salary", training=False
    )

    model = train_model(X_train, y_train)
    preds = inference(model=model, X=X_test)
    precision, recall, fbeta = compute_model_metrics(y=y_test, preds=preds)

    assert isinstance(precision, float), f"Precision dtype is different from expected."
    assert isinstance(recall, float), f"Precision dtype is different from expected."
    assert isinstance(fbeta, float), f"Precision dtype is different from expected."


@pytest.mark.skip(reason="Do not want to re-train the model for now")
def test_save_model(data):
    # Split dataset into training and testing set
    train, test = train_test_split(data, test_size=0.20, random_state=42)

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

    X_train, y_train, encoder, lb, scaler = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    model = train_model(X_train, y_train)
    # Save the mdoel.
    save_to_file(model.best_estimator_, os.path.join(root_dir, "model", "classifier"))

    my_file = os.path.join(root_dir, "my_model.joblib")
    assert os.path.isfile(my_file), f"Model not saved as expected"
