from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
import matplotlib.pyplot as plt
import joblib


def train_logistic_regression(X_train, X_test, y_train, y_test):

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n===== Logistic Regression Results =====")
    print("\nAccuracy:", accuracy)
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    return model


def train_random_forest(X_train, X_test, y_train, y_test):

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n===== Random Forest Results =====")
    print("\nAccuracy:", accuracy)
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    return model


def plot_feature_importance(model, X_train):

    importance = model.feature_importances_

    feature_importance = pd.Series(importance, index=X_train.columns)

    feature_importance = feature_importance.sort_values(ascending=False)

    print("\nTop 10 Important Features:\n")
    print(feature_importance.head(10))

    plt.figure()

    feature_importance.head(10).plot(kind="bar")

    plt.title("Top 10 Features Influencing Employee Attrition")

    plt.ylabel("Importance Score")

    plt.xlabel("Features")

    plt.tight_layout()

    plt.show()


def save_model(model):

    joblib.dump(model, "outputs/attrition_model.pkl")

    print("\nModel saved to outputs/attrition_model.pkl")