# TODO: add necessary import
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, train_model

# TODO: implement the first test. Change the function name and input as needed


def test_process_data():
    """
    Verify that process_data returns matching feature and label counts.
    """
    data = pd.read_csv("data/census.csv")

    train, _ = train_test_split(
        data,
        test_size=0.20,
        random_state=42
    )

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

    X, y, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    assert X.shape[0] == y.shape[0]
    assert encoder is not None
    assert lb is not None

# TODO: implement the second test. Change the function name and input as needed


def test_train_model():
    """
    Verify that train_model returns a trained model object.
    """
    data = pd.read_csv("data/census.csv")

    train, _ = train_test_split(
        data,
        test_size=0.20,
        random_state=42
    )

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

    X, y, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    model = train_model(X, y)

    assert model is not None

# TODO: implement the third test. Change the function name and input as needed


def test_compute_model_metrics():
    """
    Verify that metric calculations return expected values.
    """
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 1, 0]

    precision, recall, fbeta = compute_model_metrics(
        y_true,
        y_pred
    )

    assert round(precision, 2) == 1.00
    assert round(recall, 2) == 0.67
    assert round(fbeta, 2) == 0.80
