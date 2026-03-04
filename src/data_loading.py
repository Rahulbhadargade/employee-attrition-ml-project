import pandas as pd


def load_data():
    path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":

    df = load_data()

    print("Dataset Shape:", df.shape)
    print("\nColumns:\n", df.columns)
    print("\nFirst 5 rows:\n", df.head())