import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import warnings  # DO NOT modify this line
from sklearn.exceptions import ConvergenceWarning  # DO NOT modify this line

warnings.filterwarnings(
    "ignore", category=ConvergenceWarning
)  # DO NOT modify this line


class BankLogistic:
    def __init__(self, data_path):  # DO NOT modify this line
        self.data_path = data_path
        self.df = pd.read_csv(data_path, sep=",")
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def Q1(self):  # DO NOT modify this line
        """
        Problem 1:
            Load ‘bank-st.csv’ data from the “Attachment”
            How many rows of data are there in total?

        """
        return self.df.shape[0]

    def Q2(self):  # DO NOT modify this line
        """
        Problem 2:
            return the tuple of numeric variables and categorical variables are presented in the dataset.
        """
        return (
            self.df.select_dtypes(include=["int64", "float64"]).shape[1],
            self.df.select_dtypes(include=["object"]).shape[1],
        )

    def Q3(self):  # DO NOT modify this line
        """
        Problem 3:
            return the tuple of the Class 0 (no) followed by Class 1 (yes) in 3 digits.
        """
        return tuple(round(self.df["y"].value_counts() / self.df.shape[0], 3))

    def Q4(self):  # DO NOT modify this line
        """
        Problem 4:
            Remove duplicate records from the data. What are the shape of the dataset afterward?
        """
        self.df.drop_duplicates(inplace=True)
        return self.df.shape

    def Q5(self):  # DO NOT modify this line
        """
        Problem 5:
            5. Replace unknown value with null
            6. Remove features with more than 99% flat values.
                Hint: There is only one feature should be drop
            7. Split Data
            -	Split the dataset into training and testing sets with a 70:30 ratio.
            -	random_state=0
            -	stratify option
            return the tuple of shapes of X_train and X_test.

        """
        self.Q4()
        self.df.replace("unknown", None, inplace=True)
        for col in self.df.columns:
            if self.df[col].value_counts(normalize=True).values[0] > 0.99:
                self.df.drop(columns=col, inplace=True)

        X = self.df.drop(columns="y")
        y = self.df["y"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y
        )
        return tuple(self.X_train.shape), tuple(self.X_test.shape)

    def Q6(self):
        """
        Problem 6:
            8. Impute missing
                -	For numeric variables: Impute missing values using the mean.
                -	For categorical variables: Impute missing values using the mode.
                Hint: Use statistics calculated from the training dataset to avoid data leakage.
            9. Categorical Encoder:
                Map the nominal data for the education variable using the following order:
                education_order = {
                    'illiterate': 1,
                    'basic.4y': 2,
                    'basic.6y': 3,
                    'basic.9y': 4,
                    'high.school': 5,
                    'professional.course': 6,
                    'university.degree': 7}
                Hint: Use One hot encoder or pd.dummy to encode nominal category
            return the shape of X_train.

        """
        self.Q5()
        self.X_train.fillna(self.X_train.mean(numeric_only=True), inplace=True)
        self.X_train.fillna(self.X_train.mode().iloc[0], inplace=True)

        self.X_test.fillna(self.X_train.mean(numeric_only=True), inplace=True)
        self.X_test.fillna(self.X_train.mode().iloc[0], inplace=True)

        education_order = {
            "illiterate": 1,
            "basic.4y": 2,
            "basic.6y": 3,
            "basic.9y": 4,
            "high.school": 5,
            "professional.course": 6,
            "university.degree": 7,
        }
        self.X_train["education"] = self.X_train["education"].map(education_order)
        self.X_test["education"] = self.X_test["education"].map(education_order)

        cat_cols = self.X_train.select_dtypes(include="object").columns
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        X_train_cat = pd.DataFrame(encoder.fit_transform(self.X_train[cat_cols]))
        X_test_cat = pd.DataFrame(encoder.fit_transform(self.X_test[cat_cols]))

        self.X_train.drop(columns=cat_cols, inplace=True)
        self.X_test.drop(columns=cat_cols, inplace=True)

        self.X_train = pd.DataFrame(np.hstack([self.X_train, X_train_cat]))
        self.X_test = pd.DataFrame(np.hstack([self.X_test, X_test_cat]))

        # self.X_train = pd.get_dummies(self.X_train, columns=self.X_train.select_dtypes(include='object').columns)
        # self.X_test = pd.get_dummies(self.X_test, columns=self.X_test.select_dtypes(include='object').columns)
        return self.X_train.shape

    def Q7(self):
        """Problem7: Use Logistic Regression as the model with
        random_state=2025,
        class_weight='balanced' and
        max_iter=500.
        Train the model using all the remaining available variables.
        What is the macro F1 score of the model on the test data? in 2 digits
        """
        self.Q6()
        model = LogisticRegression(
            random_state=2025, class_weight="balanced", max_iter=500
        )
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return round(f1_score(self.y_test, y_pred, average="macro"), 2)
