from src.data_loading import load_data
from src.preprocessing import preprocess_data
from src.model_training import (
    train_logistic_regression,
    train_random_forest,
    plot_feature_importance,
    save_model
)


def main():

    df = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("\nTraining shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    train_logistic_regression(X_train, X_test, y_train, y_test)

    rf_model = train_random_forest(X_train, X_test, y_train, y_test)

    save_model(rf_model)

    plot_feature_importance(rf_model, X_train)


if __name__ == "__main__":
    main()