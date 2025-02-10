import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class Clustering:
    def __init__(self, file_path):  # DO NOT modify this line
        # Add other parameters if needed
        self.file_path = file_path
        self.df = pd.read_csv(file_path, sep=",")

    def Q1(self):  # DO NOT modify this line
        """
        Step1-4
            1. Load the CSV file.
            2. Choose edible mushrooms only.
            3. Only the variables below have been selected to describe the distinctive
               characteristics of edible mushrooms:
               'cap-color-rate','stalk-color-above-ring-rate'
            4. Provide a proper data preprocessing as follows:
                - Fill missing with mean
                - Standardize variables with Standard Scaler
        """
        self.df = self.df[self.df["label"] == "e"]
        self.df = self.df[["cap-color-rate", "stalk-color-above-ring-rate"]]
        self.df.fillna(self.df.mode().iloc[0], inplace=True)

        scaler = StandardScaler()
        self.scaled_features = scaler.fit_transform(self.df)
        self.scaler = scaler

        return self.scaled_features.shape

    # self.df.to_csv("output.csv", index=True)
    def Q2(self):  # DO NOT modify this line
        """
        Step5-6
            5. K-means clustering with 5 clusters (n_clusters=5, random_state=0, n_init='auto')
            6. Show the maximum centroid of 2 features ('cap-color-rate' and 'stalk-color-above-ring-rate') in 2 digits.
        """
        self.Q1()
        kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto")
        kmeans.fit(self.scaled_features)

        self.centroids = kmeans.cluster_centers_
        max_centroid = np.max(self.centroids, axis=0)
        # print(self.scaled_features)
        # print(self.centroids)
        return np.round(max_centroid, 2)

        kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto")
        kmeans.fit(self.scaled_features)

        # Store centroids for Q3
        self.centroids = kmeans.cluster_centers_
        # Find the maximum centroid for both features
        max_centroid = np.max(self.centroids, axis=0)

    def Q3(self):  # DO NOT modify this line
        """
        Step7
            7. Convert the centroid value to the original scale, and show the minimum centroid of 2 features in 2 digits.
        """
        self.Q2()
        original_centroids = self.scaler.inverse_transform(self.centroids)

        min_centroid = np.min(original_centroids, axis=0)
        return np.round(min_centroid, 2)
