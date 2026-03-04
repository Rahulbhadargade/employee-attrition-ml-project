import matplotlib.pyplot as plt
import seaborn as sns


def basic_info(df):

    print("\nDataset Info\n")
    print(df.info())

    print("\nStatistical Summary\n")
    print(df.describe())

    print("\nMissing Values\n")
    print(df.isnull().sum())
    
    print("\nAttrition Counts\n")
    print(df["Attrition"].value_counts())


def attrition_distribution(df):

    plt.figure()

    sns.countplot(x="Attrition", data=df)

    plt.title("Employee Attrition Distribution")

    plt.show()


def overtime_vs_attrition(df):

    plt.figure()

    sns.countplot(x="OverTime", hue="Attrition", data=df)

    plt.title("Attrition vs Overtime")

    plt.show()