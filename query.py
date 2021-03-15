import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from tensorflow.keras import backend as K
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances
from modAL.uncertainty import uncertainty_sampling


class Query(ABC):
    """
    Abstract class for active learning query techniques
    """

    def __init__(
        self,
        n_instances: int,
    ) -> None:
        """
        Define the number of instances returned by query techniques
        """
        self.n_instances = n_instances

    @abstractmethod
    def __call__(self, classifier, pool):
        """
        Abstract method called by active learning techniques
        self-> instance of AL technique
        classifier -> ML model
        pool-> set of unlabeled images to be used in AL technique
        """


class RandomSampling(Query):
    """
    Base AL technic, select images from unlabeled dataset randomly
    """

    def __call__(self, classifier, X):
        """
        self-> instance of AL technique
        classifier -> ML model
        pool-> set of unlabeled images to be used in AL technique

        returns as images and corresponding indexes
        """
        # Select random indices
        query_idx = np.random.choice(
            range(len(X)),
            size=self.n_instances,
            replace=True,
        )
        # Return query indices and unlabeled data at those
        # indices
        return query_idx, X[query_idx]


class UncertaintySampling(Query):
    """
    Uncertainty AL technic, select images from unlabeled that has confidence prediction closer from 50% (decision boundary)
    """

    def __call__(self, classifier, X):
        """
        self-> instance of Uncertainty AL technic
        classifier -> ML model
        pool-> set of unlabeled images to be used in Uncertainty AL technic

        returns as images and corresponding indexes
        """
        # Select indices based on uncertainty
        query_idx = uncertainty_sampling(
            classifier,
            X,
            n_instances=self.n_instances,
        )
        # Return query indices and unlabeled data at those
        # indices
        return query_idx, X[query_idx]


class ClusterBasedSampling(Query):
    """
    Cluster AL technic, split imagens in clusters and select one image by each cluster randomly
    """

    def __call__(self, classifier, X):
        """
        self-> instance of AL technique
        classifier -> ML model
        pool-> set of unlabeled images to be used in AL technique

        returns as images and corresponding indexes
        """
        X = X.reshape(len(X), -1)

        # Instantiate KMeans object
        # With the number of clusters, equal
        # to the number of instances
        kmeans = MiniBatchKMeans(n_clusters=self.n_instances, random_state=0)

        # Fit the data
        kmeans.fit(X)

        # Get one index from each cluster
        query_idx = self.get_one_index_from_each_cluster(kmeans)

        return query_idx, X[query_idx]

    def get_one_index_from_each_cluster(self, kmeans):
        """
        Selects a random point from each cluster.
        The number of clusters is determined by `n_instances`.
        """
        # For each cluster:
        # (1) Find all the points from X that are assigned to the cluster.
        # (2) Choose 1 point from tese points randomly.
        # The number of clusters is `self.n_instances`

        return [
            int(np.random.choice(np.where(kmeans.labels_ == i)[0], size=1))
            for i in range(self.n_instances)
        ]


# class OutlierSampling(Query):
#     """
#     Docs here
#     """

#     def __init__(self, n_instances: int, x_validation: np.ndarray) -> None:
#         """
#         Docs here
#         """
#         # Save validation data
#         self.x_validation = x_validation
#         super().__init__(n_instances)

#     def __call__(self, classifier, X):
#         # Get per-neuron scores from validation data
#         validation_rankings = self.get_validation_rankings(
#             classifier, self.X_validation
#         )


# class OutlierSampling(Query):
#     """
#     Docs here
#     """

#     def __init__(self, unlabeled_data: np.ndarray, n_instances: int,**kwargs) -> None:
#         """
#         Docs here
#         """
#         self.X_validation=kwargs.get("X_validation")


#         super().__init__(unlabeled_data, n_instances)

#     def __call__(self,learner, X_pool, *args, **kwargs):
#         """
#         Docs here
#         """
#         self.unlabeled_data = X_pool,

#         if learner is None:
#             raise ValueError("learner_data param is missing")
#         if self.unlabeled_data is None:
#             raise ValueError("unlabeled_data param is missing")
#         if self.n_instances is None:
#             raise ValueError("n_instances param is missing")
#         if self.X_validation is None:
#             raise ValueError("X_validation param is missing")

#         model = learner.estimator.model
#         # Get per-neuron scores from validation data
#         validation_rankings = self.get_validation_rankings(model, self.X_validation)

#         index = 0
#         # outliers = {}
#         outliers_rank = {}
#         for item in self.unlabeled_data:

#             item = item[np.newaxis, ...]
#             # get logit of item

#             # keras_function =  K.function([model.layers[0].input],[model.layers[-1].output])
#             keras_function = K.function([model.get_input_at(0)], [model.layers[-1].output])
#             neuron_outputs = keras_function([item, 1])

#             n = 0
#             ranks = []
#             for output in neuron_outputs:
#                 rank = self.get_rank(output, validation_rankings[n])
#                 ranks.append(rank)
#                 n += 1

#             outliers_rank[index] = 1 - (sum(ranks) / len(neuron_outputs))  # average rank
#             index = index + 1

#         outliers_rank = sorted(outliers_rank.items(), key=lambda x: x[1], reverse=True)

#         query_idx = []
#         for outlier in outliers_rank[:self.n_instances:]:
#             query_idx.append(outlier[0])

#         return query_idx, self.unlabeled_data[query_idx]

#     def get_rank(self,value, rankings):
#         """get the rank of the value in an ordered array as a percentage

#         Keyword arguments:
#             value -- the value for which we want to return the ranked value
#             rankings -- the ordered array in which to determine the value's ranking

#         returns linear distance between the indexes where value occurs, in the
#         case that there is not an exact match with the ranked values
#         """

#         index = 0  # default: ranking = 0

#         for ranked_number in rankings:
#             if value < ranked_number:
#                 break  # NB: this O(N) loop could be optimized to O(log(N))
#             index += 1

#         if index >= len(rankings):
#             index = len(rankings)  # maximum: ranking = 1

#         elif index > 0:
#             # get linear interpolation between the two closest indexes

#             diff = rankings[index] - rankings[index - 1]
#             perc = value - rankings[index - 1]
#             linear = perc / diff
#             index = float(index - 1) + linear

#         absolute_ranking = index / len(rankings)

#         return absolute_ranking

#     def get_validation_rankings(self,model, validation_data):
# validation_rankings = (
#     []
# )  # 2D array, every neuron by ordered list of output on validation data per neuron
# v = 0
# for item in validation_data:

#     item = item[np.newaxis, ...]
#     # get logit of item
#     print(type(model.layers[0].input[0]))
#     print(model.layers[0].input[0])

#     keras_function =  K.function([model.layers[0].input[0]],[model.layers[-1].output])
#     #keras_function = K.function([model.get_input_at(0)], [model.layers[-1].output])
#     neuron_outputs = keras_function([item, 1])

#     # initialize array if we haven't yet
#     if len(validation_rankings) == 0:
#         for output in neuron_outputs:
#             validation_rankings.append([0.0] * len(validation_data))

#     n = 0
#     for output in neuron_outputs:
#         validation_rankings[n][v] = output
#         n += 1

#     v += 1

# # Rank-order the validation scores
# v = 0
# for validation in validation_rankings:
#     validation.sort()
#     validation_rankings[v] = validation
#     v += 1

# return validation_rankings


class RepresentativeSampling(Query):
    """
    Representative Sampling is a technic that calculates the difference
    between the training data and the unlabeled data. 
    For each unlabeled image is calculated the training score and unlabeled. 
    The score is the cosine similarity mean between that image and each image of the respective image dataset. 
    After calculating both scores, representativity is obtained by the dif-ference between the unlabeled score and training score. 
    """

    def __call__(self, classifier, X):
        """
        self-> instance of AL technique
        classifier -> ML model
        pool-> set of unlabeled images to be used in AL technique

        returns as images and corresponding indexes
        """
        # Get training vector
        (batch_size, length, height, depth) = classifier.X_training.shape
        train_vector = classifier.X_training.reshape(
            (
                batch_size,
                length * height * depth,
            )
        )

        # Get unlabeled vector
        (batch_size, length, height, depth) = X.shape
        unlabeled_vector = X.reshape(
            (
                batch_size,
                length * height * depth,
            )
        )

        # Calculate similarities
        train_similarity = pairwise_distances(
            unlabeled_vector, train_vector, metric="cosine"
        )

        unlabeled_similarity = pairwise_distances(
            unlabeled_vector, unlabeled_vector, metric="cosine"
        )

        representativity = np.fromiter(
            (
                np.mean(unlabeled_sim) - np.mean(train_sim)
                for train_sim, unlabeled_sim in zip(
                    train_similarity, unlabeled_similarity
                )
            ),
            dtype=float,
        )

        # Select first n elements (`n_instances`)
        # (Most representative ones)
        query_idx = (-representativity).argsort()[: self.n_instances]

        return query_idx, X[query_idx]


class UncertaintyWithClusteringSampling(Query):
    """
    Least Confidence Sampling with Clustering-based Sampling
    Combining Uncertainty and Diversity sampling means applying one technique and then another.
    this allow select images in different positions of the boarder
    """

    def __init__(self, n_instances: int) -> None:
        """
        Init UncertantySampling, ClusterBasedSampling and abstract class
        """
        self.uncertainty_sampling = UncertaintySampling(
            n_instances=500,
        )
        self.clustering_based_sampling = ClusterBasedSampling(
            n_instances=n_instances,
        )
        super().__init__(n_instances)

    def __call__(self, classifier, X):
        """
        self-> instance of AL technique
        classifier -> ML model
        pool-> set of unlabeled images to be used in AL technique

        returns as images and corresponding indexes
        """
        indices, _ = self.uncertainty_sampling.__call__(
            classifier,
            X,
        )

        query_idx, _ = self.clustering_based_sampling(
            classifier,
            X[indices],
        )

        return indices[query_idx], X[indices[query_idx]]

# def Uncertainty_With_ModelOutliers_sampling(**kwargs):
#     """
#     Uncertainty Sampling with Model-based Outliers

#     When Combining Uncertainty Sampling with Model-based Outliers, you are maximizing your model’s current confusion.
#     You are looking for items near the decision boundary and making sure that their features are relatively unknown
#     to the current model.
#     """
#     n_Uncertainty_instances = kwargs.get("n_Uncertainty_instances", 500)
#     learner = kwargs.get("learner")
#     unlabeled_data = kwargs.get("unlabeled_data")
#     validation_data = kwargs.get("validation_data")
#     n_instances = kwargs.get("n_instances")
#     if learner is None:
#         raise ValueError("Learner param is missing")
#     if unlabeled_data is None:
#         raise ValueError("unlabeled_data param is missing")
#     if n_instances is None:
#         raise ValueError("n_instances param is missing")
#     if validation_data is None:
#         raise ValueError("validation_data param is missing")
#     indices, instancias = Uncertainty(
#         learner=learner,
#         unlabeled_data=unlabeled_data,
#         n_instances=n_Uncertainty_instances,
#     )
#     query_idx, data = outlier_sampling(
#         learner=learner,
#         unlabeled_data=instancias,
#         validation_data=validation_data,
#         n_instances=n_instances,
#     )
#     return (
#         indices[
#             query_idx,
#         ],
#         data,
#     )



class RepresentativeWithClusteringSampling(Query):
    """
    Representative Sampling Cluster-based Sampling is an approach that clusters unlabeled
    images and then calculates which image is the most representative by cluster.
    """

    def __init__(self, n_instances: int) -> None:
        """
        Init RepresentativeSampling and abstract class
        """
        self.representative_sampling = RepresentativeSampling(
            n_instances=1,
        )
        super().__init__(n_instances)

    def __call__(self, classifier, X):
        """
        self-> instance of AL technique
        classifier -> ML model
        pool-> set of unlabeled images to be used in AL technique

        returns as images and corresponding indexes
        """
        (batch_size, length, height, depth) = X.shape
        X = X.reshape(len(X), -1)

        # Instantiate clustering object
        kmeans = MiniBatchKMeans(n_clusters=self.n_instances, random_state=0)

        kmeans.fit(X)

        query_idx = []

        # Iterate over each cluster
        for i in range(self.n_instances):

            # Get cluster indices
            cluster_indices = np.where(kmeans.labels_ == i)[0].tolist()
 
            images=X[cluster_indices].reshape(len(X[cluster_indices]),length, height, depth)
            # Get most representative from each cluster
            indice, _ = self.representative_sampling.__call__(
                classifier,
                images,
            )

            query_idx.append(cluster_indices[int(indice)])

        return query_idx, X[query_idx]


class HighestEntropyClusteringSampling(Query):
    """
    Sampling from the Highest Entropy Cluster
    Select n images from the cluster where the images have more entropy (average incertainty is bigger)
    """

    def __init__(self, n_instances: int) -> None:
        self.n_clusters = 10
        super().__init__(n_instances)

    def __call__(self, classifier, X):
        """
        self-> instance of AL technique
        classifier -> ML model
        pool-> set of unlabeled images to be used in AL technique

        returns as images and corresponding indexes
        """

        (batch_size, length, height, depth) = X.shape
        X = X.reshape(len(X), -1)

        # Instantiate clustering object
        kmeans = MiniBatchKMeans(n_clusters=self.n_instances, random_state=0)

        kmeans.fit(X)

        clusters_average_uncertainty = []

        # Iterate over each cluster
        for i in range(self.n_clusters):
            # Get cluster indices
            cluster_indices = np.where(kmeans.labels_ == i)[0].tolist()
            # Use the indices to compute probabilities
            
            images=X[cluster_indices].reshape(len(X[cluster_indices]),length, height, depth)
            
            #this mis deprecated, now model.predict return probabilities of prediction values

            probabilities=classifier.predict_proba(images)

            print(probabilities)
            # Compute uncertanties
            uncertanties = [abs(i[0] - i[1]) for i in probabilities]
            clusters_average_uncertainty.append(np.mean(uncertanties))

        # Get the index of the most uncertain cluster
        most_uncertain_cluster = np.argmax(clusters_average_uncertainty)
        # Get indices from must uncertain cluster
        cluster_indices = np.where(kmeans.labels_ == most_uncertain_cluster)[0].tolist()
        # Select random elements from cluster
        query_idx = np.random.choice(cluster_indices, self.n_instances, replace=True)

        X=X.reshape(len(X),length, height, depth)

        return query_idx, X[query_idx]


class UncertaintyWithRepresentativeSampling(Query):
    """
    samples unlabeled images by uncertainty and 
    then filters them using a Diversity Sampling technique called Representative Sampling. 
    """

    def __init__(self, n_instances: int) -> None:
        self.uncertainty_sampling = UncertaintySampling(n_instances=500)
        self.representative_sampling = RepresentativeSampling(n_instances=n_instances)
        super().__init__(n_instances)

    def __call__(self, classifier, X):
        """
        self-> instance of AL technique
        classifier -> ML model
        pool-> set of unlabeled images to be used in AL technique

        returns as images and corresponding indexes
        """
        # Get query idx from Uncertainty Sampling
        query_idx, instances = self.uncertainty_sampling.__call__(classifier, X)

        # Use these previous instances in Representative Sampling
        indices, instances = self.representative_sampling.__call__(
            classifier, instances
        )

        return query_idx[indices], instances


class HighestEntropyUncertaintySampling(Query):
    """
    High Uncertainty Cluster and then applies the Uncertainty Sampling 
    to the result. With this technic, the images that will be sampled are
    the most uncertainty images from the most uncertainty cluster 

    """

    def __init__(self, n_instances: int) -> None:
        self.highest_entropy_clustering_sampling = HighestEntropyClusteringSampling(
            n_instances=100
        )
        super().__init__(n_instances)

    def __call__(self, classifier, X):
        """
        self-> instance of AL technique
        classifier -> ML model
        pool-> set of unlabeled images to be used in AL technique

        returns as images and corresponding indexes
        """

        # Use highest entropy clustering first
        entropy_clustering_indices,instances=self.highest_entropy_clustering_sampling.__call__(classifier, X)
        # Get the most uncertain ones
        print(entropy_clustering_indices)
        query_idx= uncertainty_sampling(
            classifier,
            instances,
            n_instances=self.n_instances,
        )
        print(entropy_clustering_indices[query_idx])
        return (entropy_clustering_indices[query_idx], instances)





# def Model_Outliers_With_Representative_sampling(**kwargs):
#     """
#     Model-based Outliers and Representative Sampling
#     """
#     n_outliers_instances = kwargs.get("n_outliers_instances", 500)
#     learner = kwargs.get("learner")
#     unlabeled_data = kwargs.get("unlabeled_data")
#     validation_data = kwargs.get("validation_data")
#     n_instances = kwargs.get("n_instances")
#     if learner is None:
#         raise ValueError("Learner param is missing")
#     if unlabeled_data is None:
#         raise ValueError("unlabeled_data param is missing")
#     if n_instances is None:
#         raise ValueError("n_instances param is missing")
#     if validation_data is None:
#         raise ValueError("validation_data param is missing")

#     indices, instancias = outlier_sampling(
#         learner=learner,
#         unlabeled_data=unlabeled_data,
#         validation_data=validation_data,
#         n_instances=n_outliers_instances,
#     )
#     indices = np.array(indices)  # convert list to numpy array
#     query_idx, data = representative_sampling(
#         learner=learner, unlabeled_data=instancias, n_instances=n_instances
#     )
#     return indices[query_idx], data

