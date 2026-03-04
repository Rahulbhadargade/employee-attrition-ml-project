import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(df):

    # Convert target variable to numeric
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # Drop useless columns
    df = df.drop(columns=[
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours"
    ])

    # Separate features and target
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test