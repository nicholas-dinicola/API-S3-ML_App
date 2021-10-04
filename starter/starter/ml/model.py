from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import logging
from joblib import dump

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    logger.info("Instantiate the model")
    # instantiate the model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10]

    }

    model_cv = GridSearchCV(
        rf,
        param_grid=params,
        scoring="accuracy",
        cv=5,
        n_jobs=-1
    )

    logger.info("Fitting the GridSearchCV")
    # fit the model on the training set
    model_cv.fit(X_train, y_train)

    model = model_cv
    return model



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    logger.info("Computing the metrics")
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    logger.info("Running model on testing set")
    preds = model.predict(X)

    return preds

def save_model(model, pth: str, name: str):
    """

    Parameters
    ----------
    model: trained model
    pth: path where saving  the model

    Returns
    -------

    """
    logger.info(f"Savingthe model in {pth} as {name}")
    dump(model.best_estimator_, pth + name + ".joblib")
    logger.info("Model has been saved")