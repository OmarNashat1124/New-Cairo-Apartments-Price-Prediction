from Preprocessing_utils import (
    load_data,
    save_data,
    preprocess
)
from model_utils import build_catboost


def main():
    df = load_data()

    df = preprocess(df)

    save_data(df)

    model, metrics = build_catboost(df)

    print(metrics)


if __name__ == "__main__":
    main()
