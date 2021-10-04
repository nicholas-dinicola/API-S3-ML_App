# Script to train machine learning model.
import sys
sys.path.insert(0, "/home/nicolas/Desktop/API-S3-ML_App/starter/starter/ml/")
from data import process_data
from model import train_model, compute_model_metrics, inference, save_model
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv("/home/nicolas/Desktop/API-S3-ML_App/starter/data/census_no_spaces.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42, stratify=data["salary"])

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

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb, scaler = process_data(
    test, categorical_features=cat_features, encoder=encoder, lb=lb, scaler=scaler, label="salary", training=False
)

# Save the encoded features
encoded_features = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_train)], axis=0)
encoded_features.to_csv("/home/nicolas/Desktop/API-S3-ML_App/starter/data/encoded_features.csv", index=False)

# Check the dataset have correctly been split
print(f"Split dataset {X_train.shape, X_test.shape, y_train.shape, y_test.shape}")

# Train and save a model.
model = train_model(X_train, y_train)

print(f"Split dataset {model.best_estimator_}")

# Test the model
preds = inference(model=model, X=X_test)
precision, recall, fbeta = compute_model_metrics(y=y_test, preds=preds)

print(
    f"Train precision: {precision},\n"
    f"Train recall: {precision},\n"
    f"Train fbeta: {fbeta}")

# Save the mdoel.
file_pth = "/home/nicolas/Desktop/API-S3-ML_App/starter/model/"

save_model(model=model, pth=file_pth, name="my_model")


