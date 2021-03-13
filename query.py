import numpy as np
import random
from abc import ABC, abstractmethod
import tensorflow as tf
from keras import backend as K

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances
from modAL.uncertainty import uncertainty_sampling


class Query(ABC):
    """
    Docs here
    """

    def __init__(
        self,
        n_instances: int,
    ) -> None:
        self.n_instances = n_instances

    @abstractmethod
    def __call__(self, classifier, X):
        """
        Docs
        """


class RandomSampling(Query):
    """
    Docs here
    """

    def __call__(self, classifier, X):
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
    Docs here
    """

    def __call__(self, classifier, X):
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
    Docs here
    """

    def __call__(self, classifier, X):
        """
        CLUSTERING BASED SAMPLING
        group images in n clusters and select randomly one image for each clusters
        """
        # TODO: Check this line
        unlabeled_data = X.reshape(len(X), -1)

        # Instantiate KMeans object
        # With the number of clusters, equal
        # to the number of instances
        kmeans = MiniBatchKMeans(n_clusters=self.n_instances, random_state=0)

        # Fit the data
        kmeans.fit(unlabeled_data)

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
            np.random.choice(np.where(kmeans.labels_ == i)[0].tolist(), size=1)
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
    Docs here
    """

    def __call__(self, classifier, X):
        """
        Docs here
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
        (batch_size, length, height, depth) = classifier.X.shape
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
        # TODO: Check this
        representativity = sorted(
            (
                np.mean(unlabeled_sim) - np.mean(train_sim)
                for train_sim, unlabeled_sim in zip(
                    train_similarity, unlabeled_similarity
                )
            ),
            reverse=True,
        )
        # Select first n elements (`n_instances`)
        # (Most representative ones)
        query_idx = representativity[: self.n_instances]

        return query_idx, X[query_idx]


class UncertaintyWithClusteringSampling(Query):
    """
    Docs here
    """

    def __init__(self, n_instances: int) -> None:
        """
        Docs here
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
        Least Confidence Sampling with Clustering-based Sampling

        Combining Uncertainty and Diversity sampling means applying one technique and then another.
        this allow select images in different positions of the boarder
        """
        indices, _ = self.uncertainty_sampling.__call__(
            classifier,
            X,
        )

        query_idx, _ = self.clustering_based_sampling(
            classifier,
            X[indices],
        )

        return query_idx, X[query_idx]


class RepresentativeWithClusteringSampling(Query):
    """
    Docs here
    """

    def __init__(self, n_instances: int) -> None:
        """
        Docs here
        """
        self.representative_sampling = RepresentativeSampling(
            n_instances=1,
        )
        super().__init__(n_instances)

    def __call__(self, classifier, X):
        """
        Docs here
        """
        # TODO: Confirm this
        # Instantiate clustering object
        kmeans = MiniBatchKMeans(n_clusters=self.n_instances, random_state=0)

        for x_batch in X:
            # Partially fit each batch
            kmeans.partial_fit(x_batch)

        indices = []

        # Iterate over each cluster
        for i in range(self.n_instances):

            # Get cluster indices
            cluster_indices = np.where(kmeans.labels_ == i)[0].tolist()

            # Get most representative from each cluster
            query_idx, X[query_idx] = self.representative_sampling.__call__(
                classifier,
                X[cluster_indices],
            )

            indices.append(int(cluster_indices[query_idx]))

        return query_idx, X[query_idx]


class HighestEntropyClusteringSampling(Query):
    """
    Docs here
    """

    def __init__(self, n_instances: int) -> None:
        self.representative = RepresentativeSampling(n_instances=1)
        self.n_clusters = 10
        super().__init__(n_instances)

    def __call__(self, classifier, X):
        """
        Sampling from the Highest Entropy Cluster
        Select n images from the cluster where the images have more entropy (average incertainty is bigger)
        """

        # Intantiate clustering object
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=0)

        for x_batch in X:
            # Partially fit each batch
            kmeans.partial_fit(x_batch)

        clusters_average_uncertainty = []

        # Iterate over each cluster
        for i in range(self.n_clusters):
            # Get cluster indices
            cluster_indices = np.where(kmeans.labels_ == i)[0].tolist()
            # Use the indices to compute probabilities
            probabilities = classifier.predict_proba(X[cluster_indices])

            # Compute uncertanties
            uncertanties = [abs(i[0] - i[1]) for i in probabilities]
            clusters_average_uncertainty.append(np.mean(uncertanties))

        # Get the index of the most uncertain cluster
        most_uncertain_cluster = np.argmax(clusters_average_uncertainty)
        # Get indices from must uncertain cluster
        cluster_indices = np.where(kmeans.labels_ == most_uncertain_cluster)[0].tolist()

        # Select random elements from cluster
        query_idx = np.random.choice(cluster_indices, self.n_instances, replace=True)

        return query_idx, X[query_idx]


class UncertaintyWithRepresentativeSampling(Query):
    """
    Docs here
    """

    def __init__(self, n_instances: int) -> None:
        self.uncertainty_sampling = UncertaintySampling(n_instances=500)
        self.representative_sampling = RepresentativeSampling(n_instances=n_instances)
        super().__init__(n_instances)

    def __call__(self, classifier, X):

        # Get query idx from Uncertainty Sampling
        _, instances = self.uncertainty_sampling.__call__(classifier, X)

        # Use these previous instances in Representative Sampling
        query_idx, instances = self.representative_sampling.__call__(
            classifier, instances
        )

        return query_idx, instances


class HighestEntropyUncertaintySampling(Query):
    """
    Docs here
    """

    def __init__(self, n_instances: int) -> None:
        self.highest_entropy_clustering_sampling = HighestEntropyClusteringSampling(
            n_instances=100
        )
        self.uncertainty_sampling = UncertaintySampling(n_instances)
        super().__init__(n_instances)

    def __call__(self, classifier, X):

        # Use highest entropy clustering first
        (
            entropy_clustering_indices,
            instances,
        ) = self.highest_entropy_clustering_sampling.__call__(classifier, X)

        # Get the most uncertain ones
        query_idx, instances = self.uncertainty_sampling.__call__(classifier, instances)

        return (entropy_clustering_indices[query_idx], instances)


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


# def Uncertainty_ModelOutliers_and_Clustering(**kwargs):
#     """
#     Uncertainty Sampling with Model-based Outliers and Clustering

#     the previous method might over-sample items that are very close to each other,
#     you might want to implement this strategy and then clustering to ensure diversity.
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
#     indices, instancias = Uncertainty_With_ModelOutliers_sampling(
#         learner=learner,
#         unlabeled_data=unlabeled_data,
#         validation_data=validation_data,
#         n_instances=n_Uncertainty_instances,
#     )
#     query_idx, data = cluster_based_sampling(
#         unlabeled_data=instancias, n_instances=n_instances
#     )
#     return (
#         indices[
#             query_idx,
#         ],
#         data,
#     )
