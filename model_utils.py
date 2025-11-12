import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import mlflow
import datetime

def build_catboost(df, target_col="price_egp", model_save_path=r"Models\catboost_model.pkl", experiment_name="CatBoost_Regression"):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_features = [
        "district", "seller_type", "finishing_type",
        "view_type", "compound_name"
    ]

    for c in categorical_features:
        if c in X.columns:
            X[c] = X[c].astype("category")

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, random_state=42, test_size=0.2
    )

    params = {
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 3,
        "l2_leaf_reg": 6,
        "subsample": 0.8,
        "bagging_temperature": 0.5,
        "random_strength": 0.8,
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "verbose": False,
        "random_seed": 42
    }

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="catboost_training"):
        for k, v in params.items():
            mlflow.log_param(k, v)

        model = CatBoostRegressor(**params)
        model.fit(x_train, y_train, cat_features=categorical_features, eval_set=(x_test, y_test))

        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)

        mlflow.log_metric("r2_train", r2_train)
        mlflow.log_metric("r2_test", r2_test)
        mlflow.log_metric("mae_train", mae_train)
        mlflow.log_metric("mae_test", mae_test)

        mlflow.sklearn.log_model(model, "catboost_model")

    joblib.dump(model, model_save_path)

    metrics = {
        "r2_train": r2_train,
        "r2_test": r2_test,
        "mae_train": mae_train,
        "mae_test": mae_test
    }

    return model, metrics



def predict_and_monitor(model, input_df):
    mlflow.set_experiment("CatBoost_Prediction_Monitoring")

    with mlflow.start_run(run_name="prediction"):

        mlflow.log_param("timestamp", str(datetime.datetime.now()))
        
        for col in input_df.columns:
            mlflow.log_param(f"input_{col}", input_df[col].iloc[0])

        prediction = model.predict(input_df)[0]

        mlflow.log_metric("prediction", prediction)

    return prediction

def load_model(path = "Models/catboost_model.pkl"):
    model = joblib.load(path)
    return model
