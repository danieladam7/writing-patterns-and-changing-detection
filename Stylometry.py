import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os


class Stylometry:
    def __init__(self):
        self.pca = None
        self.kmeans = None
        self.n_components = None
        self.n_clusters = None
        self.train_data = None
        self.test_data = None
        self.reduced_train_data = None
        self.reduced_test_data = None
        self.train_labels = None
        self.test_labels = None

    
    def _choose_n_components(self, data, variance_threshold=0.95):
        pca = PCA()
        pca.fit(data)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.where(cumulative_variance >= variance_threshold)[0][0] + 1
        return n_components

    def _choose_n_clusters(self, data, k_range=range(2, 10)):
        scores = []
        ssd = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            scores.append(score)
            ssd.append(kmeans.inertia_)

        
        self._plot_elbow_method(ssd, k_range)
        optimal_k = k_range[np.argmax(scores)]
        return optimal_k

    def _plot_elbow_method(self, ssd, k_range):
        plt.figure(figsize=(8, 4))
        plt.plot(k_range, ssd, 'bx-')
        plt.xlabel('k (number of clusters)')
        plt.ylabel('Sum of squared distances')
        plt.title('Elbow Method For Optimal k')
        plots_directory = r"Visualizations\Plots of clustering"
        if not os.path.exists(plots_directory):
            os.makedirs(plots_directory)
        file_path = os.path.join(plots_directory, f'Elbow Method for Optimal k.png')
        plt.savefig(file_path)   
        plt.show()

    def fit(self, train_data):
        self.train_data = train_data
        self.n_components = self._choose_n_components(train_data)
        self.pca = PCA(n_components=self.n_components)
        self.reduced_train_data = self.pca.fit_transform(train_data)
        self.n_clusters = self._choose_n_clusters(self.reduced_train_data)
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.train_labels = self.kmeans.fit_predict(self.reduced_train_data)

    def predict(self, test_data):
        if self.pca is None or self.kmeans is None:
            raise ValueError("Must fit on train data before predicting.")
        self.test_data = test_data
        self.reduced_test_data = self.pca.transform(test_data)
        self.test_labels = self.kmeans.predict(self.reduced_test_data)
        return self.test_labels


    def get_reduced_date(self):
        return self.reduced_train_data, self.reduced_test_data
    
    def get_labels(self):
        return self.train_labels, self.test_labels

    
    