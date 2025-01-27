import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

"""
    ASSIGNMENT 2 (STUDENT VERSION):
    Using pandas to explore Titanic data from Kaggle (titanic_to_student.csv) and answer the questions.
    (Note that the following functions already take the Titanic dataset as a DataFrame, so you don’t need to use read_csv.)

"""


def Q1(df: DataFrame):
    """
    Problem 1:
        How many rows are there in the "titanic_to_student.csv"?
    """
    return df.shape[0]


def Q2(df: DataFrame):
    """
    Problem 2:
        Drop unqualified variables
        Drop variables with missing > 50%
        Drop categorical variables with flat values > 70% (variables with the same value in the same column)
        How many columns do we have left?
    """
    half_missing = df.isnull().sum() > df.shape[0] / 2
    df.drop(df.columns[half_missing], axis=1, inplace=True)

    flat_cols = df.select_dtypes(include="object").apply(
        lambda col: col.value_counts(normalize=True).max() > 0.7
    )
    df.drop(columns=flat_cols[flat_cols].index, inplace=True)

    return df.shape[1]


def Q3(df: DataFrame):
    """
    Problem 3:
         Remove all rows with missing targets (the variable "Survived")
         How many rows do we have left?
    """
    df.dropna(subset=["Survived"], inplace=True)

    return df.shape[0]


def Q4(df: DataFrame):
    """
    Problem 4:
         Handle outliers
         For the variable “Fare”, replace outlier values with the boundary values
         If value < (Q1 - 1.5IQR), replace with (Q1 - 1.5IQR)
         If value > (Q3 + 1.5IQR), replace with (Q3 + 1.5IQR)
         What is the mean of “Fare” after replacing the outliers (round 2 decimal points)?
         Hint: Use function round(_, 2)
    """
    q25, q75 = np.percentile(df["Fare"], [25, 75])
    iqr = q75 - q25

    min_outlier = q25 - 1.5 * iqr
    max_outlier = q75 + 1.5 * iqr

    df.loc[df["Fare"] < min_outlier, "Fare"] = min_outlier
    df.loc[df["Fare"] > max_outlier, "Fare"] = max_outlier

    return round(df["Fare"].mean(), 2)


def Q5(df: DataFrame):
    """
    Problem 5:
         Impute missing value
         For number type column, impute missing values with mean
         What is the average (mean) of “Age” after imputing the missing values (round 2 decimal points)?
         Hint: Use function round(_, 2)
    """
    num_imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    df[["Pclass", "Age", "SibSp"]] = pd.DataFrame(
        num_imp.fit_transform(df[["Pclass", "Age", "SibSp"]])
    )

    return round(df["Age"].mean(), 2)


def Q6(df: DataFrame):
    """
    Problem 6:
        Convert categorical to numeric values
        For the variable “Embarked”, perform the dummy coding.
        What is the average (mean) of “Embarked_Q” after performing dummy coding (round 2 decimal points)?
        Hint: Use function round(_, 2)
    """
    enc = OneHotEncoder(handle_unknown="ignore")
    # passing bridge-types-cat column (label encoded values of bridge_types)
    nominal_columns = ["Embarked"]
    enc_df = pd.DataFrame(enc.fit_transform(df[nominal_columns]).toarray())
    unique_vals = enc.categories_[0]
    new_col_names = [f"Embarked_{val}" for val in unique_vals]
    enc_df.columns = new_col_names

    df = pd.concat([df, enc_df], axis=1)

    mean_embarked_q = round(enc_df["Embarked_Q"].mean(), 2)

    return mean_embarked_q


def Q7(df: DataFrame):
    """
    Problem 7:
        Split train/test split with stratification using 70%:30% and random seed with 123
        Show a proportion between survived (1) and died (0) in all data sets (total data, train, test)
        What is the proportion of survivors (survived = 1) in the training data (round 2 decimal points)?
        Hint: Use function round(_, 2), and train_test_split() from sklearn.model_selection,
        Don't forget to impute missing values with mean.
    """
    num_imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    df["Survived"] = pd.DataFrame(num_imp.fit_transform(df[["Survived"]]))
    df["Survived"] = df["Survived"].apply(lambda x: 1.0 if x > 0.5 else 0.0)

    y = df.pop("Survived")
    X = df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=123
    )
    survived_train = y_train.sum() / y_train.shape[0]

    return round(survived_train, 2)
