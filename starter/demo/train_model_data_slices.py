# Script to train machine learning model.
from starter.demo.ml.data import process_data
from starter.demo.ml.model import train_model, compute_model_metrics, inference, save_model
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv("/starter/data/census_no_spaces.csv")

for s in data.sex.unique():
    print(s)
    data["sex"] = data[data["sex"] == s]
    if s == " Male":
        data["sex"].replace(s, 0, inplace=True)
    else:
        data["sex"].replace(" Female", 1, inplace=True)

    # split dataset into training and testing
    train, test = train_test_split(data, test_size=0.20, random_state=42, stratify=data["salary"])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        #"sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb, scaler = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Process the test data with the process_data function.
    X_test, y_test, encoder, lb, scaler = process_data(
        test, categorical_features=cat_features, encoder=encoder, lb=lb, scaler=scaler, label="salary", training=False
    )

    # Train and save a model.
    model = train_model(X_train, y_train)

    # Test the model
    preds = inference(model=model, X=X_test)
    precision, recall, fbeta = compute_model_metrics(y=y_test, preds=preds)

    print(
        f"Train precision: {precision},\n"
        f"Train recall: {precision},\n"
        f"Train fbeta: {fbeta}")
