import numpy as np
import random
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances
from keras import backend as K
from modAL.uncertainty import uncertainty_sampling
from abc import ABC, abstractmethod


class Query(ABC):
    def __init__(self, unlabeled_data: np.ndarray, n_instances: int) -> None:
        self.unlabeled_data = unlabeled_data
        self.n_instances = n_instances

    @abstractmethod
    def __call__(self):
        """
        Docs
        """


class RandomQuery(Query):
    """
    Docs here
    """

    def __init__(self, unlabeled_data: np.ndarray, n_instances: int) -> None:
        """
        Docs here
        """
        super().__init__(unlabeled_data, n_instances)

    def __call__(self, *args, **kwargs):
        """
        Docs here
        """
        if self.unlabeled_data is None:
            raise ValueError("unlabeled_data param is missing")
        if self.n_instances is None:
            raise ValueError("n_instances param is missing")

        query_idx = np.random.choice(
            range(len(self.unlabeled_data)), size=self.n_instances, replace=False
        )
        return query_idx, self.unlabeled_data[query_idx]


class UncertaintyQuery(Query):
    """
    Docs here
    """

    def __init__(self, unlabeled_data: np.ndarray, n_instances: int) -> None:
        """
        Docs here
        """
        super().__init__(self,unlabeled_data, n_instances)

    def __call__(self, *args, **kwargs):
        """
        Docs here
        """
        if self is None:
            raise ValueError("Learner param is missing")
        if self.unlabled_data is None:
            raise ValueError("unlabled_data param is missing")
        if self.n_instances is None:
            raise ValueError("n_instances param is missing")
        indices, instancias = uncertainty_sampling(
            self.estimator.model, self.unlabled_data, self.n_instances
        )
        return indices, instancias

def Uncertainty(**kwargs):
    learner = kwargs.get("learner")
    unlabled_data = kwargs.get("unlabled_data")
    n_instances = kwargs.get("n_instances")
    if learner is None:
        raise ValueError("Learner param is missing")
    if unlabled_data is None:
        raise ValueError("unlabled_data param is missing")
    if n_instances is None:
        raise ValueError("n_instances param is missing")
    indices, instancias = uncertainty_sampling(
        learner.estimator.model, unlabled_data, n_instances
    )
    return indices, instancias

def Random(list):
    secure_random = random.SystemRandom()
    return secure_random.choice(list)


def GetOneIndexOfEachCluster(kmeans, n_clusters):
    indices = np.zeros(n_clusters)

    for i in range(n_clusters):
        lista = np.where(i == kmeans.labels_)[0]  # select images from one cluster/label
        valor = Random(lista)
        indices[i] = valor

    return indices.astype(int)


def cluster_based_sampling(**kwargs):
    """
    CLUSTERING BASED SAMPLING
    group images in n clusters and select randomly one image for each clusters
    """
    unlabled_data = kwargs.get("unlabled_data")
    n_instances = kwargs.get("n_instances")
    if unlabled_data is None:
        raise ValueError("unlabled_data param is missing")
    if n_instances is None:
        raise ValueError("n_instances param is missing")
    unlabled_data = unlabled_data.reshape(len(unlabled_data), -1)

    kmeans = MiniBatchKMeans(n_clusters=n_instances, random_state=0)
    kmeans.fit(unlabled_data)

    query_idx = GetOneIndexOfEachCluster(kmeans, n_instances)
    unlabled_data = unlabled_data.reshape(len(unlabled_data), 128, 128, 3)
    return query_idx, unlabled_data[query_idx]


def get_rank(value, rankings):
    """get the rank of the value in an ordered array as a percentage

    Keyword arguments:
        value -- the value for which we want to return the ranked value
        rankings -- the ordered array in which to determine the value's ranking

    returns linear distance between the indexes where value occurs, in the
    case that there is not an exact match with the ranked values
    """

    index = 0  # default: ranking = 0

    for ranked_number in rankings:
        if value < ranked_number:
            break  # NB: this O(N) loop could be optimized to O(log(N))
        index += 1

    if index >= len(rankings):
        index = len(rankings)  # maximum: ranking = 1

    elif index > 0:
        # get linear interpolation between the two closest indexes

        diff = rankings[index] - rankings[index - 1]
        perc = value - rankings[index - 1]
        linear = perc / diff
        index = float(index - 1) + linear

    absolute_ranking = index / len(rankings)

    return absolute_ranking


def get_validation_rankings(model, validation_data):
    validation_rankings = (
        []
    )  # 2D array, every neuron by ordered list of output on validation data per neuron
    v = 0
    for item in validation_data:

        item = item[np.newaxis, ...]
        # get logit of item

        # keras_function =  K.function([model.layers[0].input],[model.layers[-1].output])
        keras_function = K.function([model.get_input_at(0)], [model.layers[-1].output])
        neuron_outputs = keras_function([item, 1])

        # initialize array if we haven't yet
        if len(validation_rankings) == 0:
            for output in neuron_outputs:
                validation_rankings.append([0.0] * len(validation_data))

        n = 0
        for output in neuron_outputs:
            validation_rankings[n][v] = output
            n += 1

        v += 1

    # Rank-order the validation scores
    v = 0
    for validation in validation_rankings:
        validation.sort()
        validation_rankings[v] = validation
        v += 1

    return validation_rankings


def outlier_sampling(**kwargs):
    learner = kwargs.get("learner")
    unlabled_data = kwargs.get("unlabled_data")
    n_instances = kwargs.get("n_instances")
    validation_data = kwargs.get("validation_data")
    if learner is None:
        raise ValueError("Learner param is missing")
    if unlabled_data is None:
        raise ValueError("unlabled_data param is missing")
    if n_instances is None:
        raise ValueError("n_instances param is missing")
    if validation_data is None:
        raise ValueError("validation_data param is missing")

    model = learner.estimator.model
    # Get per-neuron scores from validation data
    validation_rankings = get_validation_rankings(model, validation_data)

    index = 0
    # outliers = {}
    outliers_rank = {}
    for item in unlabled_data:

        item = item[np.newaxis, ...]
        # get logit of item

        # keras_function =  K.function([model.layers[0].input],[model.layers[-1].output])
        keras_function = K.function([model.get_input_at(0)], [model.layers[-1].output])
        neuron_outputs = keras_function([item, 1])

        n = 0
        ranks = []
        for output in neuron_outputs:
            rank = get_rank(output, validation_rankings[n])
            ranks.append(rank)
            n += 1

        outliers_rank[index] = 1 - (sum(ranks) / len(neuron_outputs))  # average rank
        index = index + 1

    outliers_rank = sorted(outliers_rank.items(), key=lambda x: x[1], reverse=True)

    query_idx = []
    for outlier in outliers_rank[:n_instances:]:
        query_idx.append(outlier[0])

    return query_idx, unlabled_data[query_idx]


def representative_sampling(**kwargs):
    """
    REPRESENTATIVE SAMPLING
    select n images calculating the representativity of each image between unlabled and train images
    """
    learner = kwargs.get("learner")
    unlabled_data = kwargs.get("unlabled_data")
    n_instances = kwargs.get("n_instances")
    if learner is None:
        raise ValueError("Learner param is missing")
    if unlabled_data is None:
        raise ValueError("unlabled_data param is missing")
    if n_instances is None:
        raise ValueError("n_instances param is missing")
    len, length, height, depth = learner.X_training.shape
    vetor_train = learner.X_training.reshape((len, length * height * depth))

    len, length, height, depth = unlabled_data.shape
    vetor_unlabled = unlabled_data.reshape((len, length * height * depth))

    train_similarity = pairwise_distances(vetor_unlabled, vetor_train, metric="cosine")
    unlabled_similarity = pairwise_distances(
        vetor_unlabled, vetor_unlabled, metric="cosine"
    )

    representativity = {}
    index = 0
    for train_sim, unlabled_sim in zip(train_similarity, unlabled_similarity):
        representativity[index] = np.mean(unlabled_sim) - np.mean(train_sim)
        index = index + 1

    representativity = sorted(
        representativity.items(), key=lambda x: x[1], reverse=True
    )
    query_idx = []
    for r in representativity[:n_instances:]:
        query_idx.append(r[0])

    return query_idx, unlabled_data[query_idx]


def Uncertainty_With_Clustering_sampling(**kwargs):
    """
    Least Confidence Sampling with Clustering-based Sampling

    Combining Uncertainty and Diversity sampling means applying one technique and then another.
    this allow select images in different positions of the boarder
    """
    n_Uncertainty_instances = kwargs.get("n_Uncertainty_instances", 500)
    learner = kwargs.get("learner")
    unlabled_data = kwargs.get("unlabled_data")
    n_instances = kwargs.get("n_instances")
    if learner is None:
        raise ValueError("Learner param is missing")
    if unlabled_data is None:
        raise ValueError("unlabled_data param is missing")
    if n_instances is None:
        raise ValueError("n_instances param is missing")

    indices, instancias = Uncertainty(
        learner=learner,
        unlabled_data=unlabled_data,
        n_instances=n_Uncertainty_instances,
    )
    query_idx, data = cluster_based_sampling(
        unlabled_data=instancias, n_instances=n_instances
    )
    return indices[query_idx], data


# muito lento
def Representative_With_Clustering_sampling(**kwargs):
    n_cluster_instances = kwargs.get("n_cluster_instances", 10)
    learner = kwargs.get("learner")
    unlabled_data = kwargs.get("unlabled_data")
    n_instances = kwargs.get("n_instances")
    validation_data = kwargs.get("validation_data")
    if learner is None:
        raise ValueError("Learner param is missing")
    if unlabled_data is None:
        raise ValueError("unlabled_data param is missing")
    if n_instances is None:
        raise ValueError("n_instances param is missing")
    if validation_data is None:
        raise ValueError("validation_data param is missing")
    unlabled_data = unlabled_data.reshape(len(unlabled_data), -1)

    kmeans = MiniBatchKMeans(n_clusters=n_cluster_instances, random_state=0)
    kmeans.fit(unlabled_data)

    unlabled_data = unlabled_data.reshape(len(unlabled_data), 128, 128, 3)
    indices = []

    for i in range(n_instances):
        lista = np.where(i == kmeans.labels_)[0]  # select images from one cluster/label
        query_idx, unlabled_data[query_idx] = representative_sampling(
            learner=learner, unlabled_data=unlabled_data[lista], n_instances=1
        )
        indices.append(int(lista[query_idx]))

    return indices, unlabled_data[indices]


def Highest_Entropy__Clustering_sampling(**kwargs):
    """
    Sampling from the Highest Entropy Cluster
    Select n images from the cluster where the images have more entropy (average incertainty is bigger)
    """
    n_clusters = kwargs.get("n_cluster_instances", 10)
    learner = kwargs.get("learner")
    unlabled_data = kwargs.get("unlabled_data")
    n_instances = kwargs.get("n_instances")
    validation_data = kwargs.get("validation_data")
    if learner is None:
        raise ValueError("Learner param is missing")
    if unlabled_data is None:
        raise ValueError("unlabled_data param is missing")
    if n_instances is None:
        raise ValueError("n_instances param is missing")
    if validation_data is None:
        raise ValueError("validation_data param is missing")
    highest_average_Uncertainty = 1
    unlabled_data = unlabled_data.reshape(len(unlabled_data), -1)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(unlabled_data)

    unlabled_data = unlabled_data.reshape(len(unlabled_data), 128, 128, 3)

    for i in range(n_clusters):
        lista = np.where(i == kmeans.labels_)[0]  # select images from cluster i
        probabilidades = learner.predict_proba(unlabled_data[lista])
        incertezas = [abs(i[0] - i[1]) for i in probabilidades]
        average_Uncertainty = np.mean(incertezas)

        if average_Uncertainty < highest_average_Uncertainty:
            highest_average_Uncertainty = average_Uncertainty
            most_uncertain_cluster = i

    lista = np.where(most_uncertain_cluster == kmeans.labels_)[
        0
    ]  # select images from one cluster/label

    indices = np.random.choice(lista, n_instances, replace=False)

    return indices, unlabled_data[indices]


def Uncertainty_With_Representative_sampling(**kwargs):
    n_uncertainty_instances = kwargs.get("n_uncertainty_instances", 500)
    learner = kwargs.get("learner")
    unlabled_data = kwargs.get("unlabled_data")
    n_instances = kwargs.get("n_instances")
    if learner is None:
        raise ValueError("Learner param is missing")
    if unlabled_data is None:
        raise ValueError("unlabled_data param is missing")
    if n_instances is None:
        raise ValueError("n_instances param is missing")
    indices, instancias = Uncertainty(
        learner=learner,
        unlabled_data=unlabled_data,
        n_instances=n_uncertainty_instances,
    )
    query_idx, data = representative_sampling(
        learner=learner, unlabled_data=instancias, n_instances=n_instances
    )
    return indices[query_idx], data


def Highest_Entropy__Uncertainty_sampling(**kwargs):
    n_highest_entropy = kwargs.get("n_highest_entropy", 500)
    n_clusters = kwargs.get("n_clusters", 2)
    learner = kwargs.get("learner")
    unlabled_data = kwargs.get("unlabled_data")
    validation_data = kwargs.get("validation_data")
    n_instances = kwargs.get("n_instances")
    if learner is None:
        raise ValueError("Learner param is missing")
    if unlabled_data is None:
        raise ValueError("unlabled_data param is missing")
    if n_instances is None:
        raise ValueError("n_instances param is missing")
    if validation_data is None:
        raise ValueError("validation_data param is missing")
    indices, instancias = Highest_Entropy__Clustering_sampling(
        learner=learner,
        unlabled_data=unlabled_data,
        validation_data=validation_data,
        n_instances=n_highest_entropy,
        n_clusters=n_clusters,
    )
    query_idx, data = Uncertainty(
        learner=learner, unlabled_data=instancias, n_instances=n_instances
    )
    return indices[query_idx], data


def Uncertainty_With_ModelOutliers_sampling(**kwargs):
    """
    Uncertainty Sampling with Model-based Outliers

    When Combining Uncertainty Sampling with Model-based Outliers, you are maximizing your model’s current confusion.
    You are looking for items near the decision boundary and making sure that their features are relatively unknown
    to the current model.
    """
    n_Uncertainty_instances = kwargs.get("n_Uncertainty_instances", 500)
    learner = kwargs.get("learner")
    unlabled_data = kwargs.get("unlabled_data")
    validation_data = kwargs.get("validation_data")
    n_instances = kwargs.get("n_instances")
    if learner is None:
        raise ValueError("Learner param is missing")
    if unlabled_data is None:
        raise ValueError("unlabled_data param is missing")
    if n_instances is None:
        raise ValueError("n_instances param is missing")
    if validation_data is None:
        raise ValueError("validation_data param is missing")
    indices, instancias = Uncertainty(
        learner=learner,
        unlabled_data=unlabled_data,
        n_instances=n_Uncertainty_instances,
    )
    query_idx, data = outlier_sampling(
        learner=learner,
        unlabled_data=instancias,
        validation_data=validation_data,
        n_instances=n_instances,
    )
    return (
        indices[
            query_idx,
        ],
        data,
    )


def Model_Outliers_With_Representative_sampling(**kwargs):
    """
    Model-based Outliers and Representative Sampling
    """
    n_outliers_instances = kwargs.get("n_outliers_instances", 500)
    learner = kwargs.get("learner")
    unlabled_data = kwargs.get("unlabled_data")
    validation_data = kwargs.get("validation_data")
    n_instances = kwargs.get("n_instances")
    if learner is None:
        raise ValueError("Learner param is missing")
    if unlabled_data is None:
        raise ValueError("unlabled_data param is missing")
    if n_instances is None:
        raise ValueError("n_instances param is missing")
    if validation_data is None:
        raise ValueError("validation_data param is missing")

    indices, instancias = outlier_sampling(
        learner=learner,
        unlabled_data=unlabled_data,
        validation_data=validation_data,
        n_instances=n_outliers_instances,
    )
    indices = np.array(indices)  # convert list to numpy array
    query_idx, data = representative_sampling(
        learner=learner, unlabled_data=instancias, n_instances=n_instances
    )
    return indices[query_idx], data


def Uncertainty_ModelOutliers_and_Clustering(**kwargs):
    """
    Uncertainty Sampling with Model-based Outliers and Clustering

    the previous method might over-sample items that are very close to each other,
    you might want to implement this strategy and then clustering to ensure diversity.
    """
    n_Uncertainty_instances = kwargs.get("n_Uncertainty_instances", 500)
    learner = kwargs.get("learner")
    unlabled_data = kwargs.get("unlabled_data")
    validation_data = kwargs.get("validation_data")
    n_instances = kwargs.get("n_instances")
    if learner is None:
        raise ValueError("Learner param is missing")
    if unlabled_data is None:
        raise ValueError("unlabled_data param is missing")
    if n_instances is None:
        raise ValueError("n_instances param is missing")
    if validation_data is None:
        raise ValueError("validation_data param is missing")
    indices, instancias = Uncertainty_With_ModelOutliers_sampling(
        learner=learner,
        unlabled_data=unlabled_data,
        validation_data=validation_data,
        n_instances=n_Uncertainty_instances,
    )
    query_idx, data = cluster_based_sampling(
        unlabled_data=instancias, n_instances=n_instances
    )
    return (
        indices[
            query_idx,
        ],
        data,
    )
