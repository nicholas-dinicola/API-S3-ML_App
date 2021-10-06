# Script to train machine learning model.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, save_model
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
import os.path
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv(os.path.join(root, "data", "census_no_spaces.csv"))

# Add code to load in the data.
#data = pd.read_csv("/starter/data/census_no_spaces.csv")

# Train the model on slices of data: cat features > sex ("Female", "Male")
for s in data.sex.unique():
    print(s)
    data_slice = data[data["sex"] == s]
    if s == " Male":
        data_slice["sex"] = data["sex"].replace(s, 0)
    else:
        data_slice["sex"] = data["sex"].replace(" Female", 1)

    # split dataset into training and testing
    train, test = train_test_split(data_slice, test_size=0.20, random_state=42, stratify=data_slice["salary"])

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
    if s == " Male":
        m = s
        precision_m, recall_m, fbeta_m = compute_model_metrics(y=y_test, preds=preds)

        print(
            f"Slice for: {m},\n"
            f"Precision: {precision_m},\n"
            f"Recall: {recall_m},\n"
            f"Fbeta: {fbeta_m}")
    else:
        f = s
        precision_f, recall_f, fbeta_f = compute_model_metrics(y=y_test, preds=preds)

        print(
            f"Slice for: {f},\n"
            f"Precision: {precision_f},\n"
            f"Recall: {recall_f},\n"
            f"Fbeta: {fbeta_f}")

sys.stdout = open(os.path.join(root, "model", "slice_output.txt"), "w")
print(
    f"Slice for: {m},\n"
            f"Precision: {precision_m},\n"
            f"Recall: {recall_m},\n"
            f"Fbeta: {fbeta_m}, \n\n\n"
            f"Slice for: {f},\n"
            f"Precision: {precision_f},\n"
            f"Recall: {recall_f},\n"
            f"Fbeta: {fbeta_f}"

)
sys.stdout.close()
