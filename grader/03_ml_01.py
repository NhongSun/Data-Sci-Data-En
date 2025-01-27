import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class MushroomClassifier:
    def __init__(self, data_path):  # DO NOT modify this line
        self.data_path = data_path
        self.df = pd.read_csv(data_path)

    def Q1(self):  # DO NOT modify this line
        """
        1. (From step 1) Before doing the data prep., how many "na" are there in "gill-size" variables?
        """
        return self.df["gill-size"].isna().sum()

    def Q2(self):  # DO NOT modify this line
        """
        2. (From step 2-4) How many rows of data, how many variables?
        - Drop rows where the target (label) variable is missing.
        - Drop the following variables:
        'id','gill-attachment', 'gill-spacing', 'gill-size','gill-color-rate','stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring-rate','stalk-color-below-ring-rate','veil-color-rate','veil-type'
        - Examine the number of rows, the number of digits, and whether any are missing.
        """
        drop_list = [
            "id",
            "gill-attachment",
            "gill-spacing",
            "gill-size",
            "gill-color-rate",
            "stalk-root",
            "stalk-surface-above-ring",
            "stalk-surface-below-ring",
            "stalk-color-above-ring-rate",
            "stalk-color-below-ring-rate",
            "veil-color-rate",
            "veil-type",
        ]

        self.df.dropna(subset="label", inplace=True)
        self.df.drop(drop_list, axis=1, inplace=True)

        return self.df.shape

    def Q3(self):  # DO NOT modify this line
        """
        3. (From step 5-6) Answer the quantity class0:class1
        - Fill missing values by adding the mean for numeric variables and the mode for nominal variables.
        - Convert the label variable e (edible) to 1 and p (poisonous) to 0 and check the quantity. class0: class1
        - Note: You need to reproduce the process (code) from Q2 to obtain the correct result.
        """
        self.Q2()

        numeric_cols = self.df.select_dtypes(include=["number"]).columns
        nominal_cols = self.df.select_dtypes(include=["object"]).columns

        numeric_imputer = SimpleImputer(strategy="mean")
        nominal_imputer = SimpleImputer(strategy="most_frequent")

        self.df[numeric_cols] = numeric_imputer.fit_transform(self.df[numeric_cols])
        self.df[nominal_cols] = nominal_imputer.fit_transform(self.df[nominal_cols])

        self.df["label"] = self.df["label"].apply(lambda l: 1 if l == "e" else 0)

        ans = self.df["label"].value_counts().tolist()
        return (ans[0], ans[1])

    def Q4(self):  # DO NOT modify this line
        """
        4. (From step 7-8) How much is each training and testing sets
        - Convert the nominal variable to numeric using a dummy code with drop_first = True.
        - Split train/test with 20% test, stratify, and seed = 2020.
        - Note: You need to reproduce the process (code) from Q2, Q3 to obtain the correct result.
        """
        self.Q3()

        nominal_cols = self.df.select_dtypes(include=["object"]).columns.tolist()
        self.df = pd.get_dummies(self.df, columns=nominal_cols, drop_first=True)

        Y = self.df.pop("label")
        X = self.df
        self.random_state = 2020

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, stratify=Y, test_size=0.2, random_state=self.random_state
        )

        return (self.X_train.shape, self.X_test.shape)

    def Q5(self):
        """
        5. (From step 9) Best params after doing random forest grid search.
        Create a Random Forest with GridSearch on training data with 5 CV.
        - 'criterion':['gini','entropy']
        - 'max_depth': [2,3]
        - 'min_samples_leaf':[2,5]
        - 'N_estimators':[100]
        - 'random_state': 2020
        - Note: You need to reproduce the process (code) from Q2, Q3, Q4 to obtain the correct result.
        """
        self.Q4()
        params = {
            "criterion": ["gini", "entropy"],
            "max_depth": [2, 3],
            "min_samples_leaf": [2, 5],
            "n_estimators": [100],
        }

        self.grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=self.random_state),
            param_grid=params,
            cv=5,
            n_jobs=-1,
        )
        self.grid_search.fit(self.X_train, self.Y_train)
        self.best_params = self.grid_search.best_params_

        return (
            self.best_params["criterion"],
            self.best_params["max_depth"],
            self.best_params["min_samples_leaf"],
            self.best_params["n_estimators"],
            self.random_state,
        )

    def Q6(self):
        """
        5. (From step 10) What is the value of macro f1 (2 digits)?
        Predict the testing data set with confusion_matrix and classification_report,
        using scientific rounding (less than 0.5 dropped, more than 0.5 then increased)
        - Note: You need to reproduce the process (code) from Q2, Q3, Q4, Q5 to obtain the correct result.
        """
        self.Q5()

        best_model = self.grid_search.best_estimator_
        Y_pred = best_model.predict(self.X_test)
        report = classification_report(self.Y_test, Y_pred, output_dict=True)
        f1_class0 = round(report["0"]["f1-score"], 2)
        f1_class1 = round(report["1"]["f1-score"], 2)

        return (f1_class0, f1_class1)
        return
