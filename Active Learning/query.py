import numpy as np
import random
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances
from keras import backend as K
from modAL.uncertainty import uncertainty_sampling
from abc import ABC, abstractmethod
import tensorflow as tf

class Query(ABC):
    def __init__(self, unlabeled_data: np.ndarray, n_instances: int) -> None:
        self.unlabeled_data = unlabeled_data
        self.n_instances = n_instances

    @abstractmethod
    def __call__(self):
        """
        Docs
        """

class RandomSampling(Query):
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

class UncertaintySampling(Query):
    """
    Docs here
    """

    def __init__(self, unlabeled_data: np.ndarray, n_instances: int) -> None:
        """
        Docs here
        """

        super().__init__(unlabeled_data, n_instances)

    def __call__(self,learner,X_pool,*args, **kwargs):
        """
        Docs here
        """
        self.unlabeled_data = X_pool

        if learner is None:
            raise ValueError("learner_data param is missing")
        if self.unlabeled_data is None:
            raise ValueError("unlabeled_data param is missing")
        if self.n_instances is None:
            raise ValueError("n_instances param is missing")
        
        indices = uncertainty_sampling(
        learner.estimator.model, self.unlabeled_data, n_instances=self.n_instances
        )
        return indices, self.unlabeled_data[indices]

class ClusterBasedSampling(Query):
    
    """
    Docs here
    """
    def __init__(self, unlabeled_data: np.ndarray, n_instances: int) -> None:
        """
        Docs here
        """

        super().__init__(unlabeled_data, n_instances)

    def __call__(self,learner,X_pool, *args, **kwargs):
        """
        CLUSTERING BASED SAMPLING
        group images in n clusters and select randomly one image for each clusters
        """
        
        self.unlabeled_data = X_pool

        if self.unlabeled_data is None:
            raise ValueError("unlabeled_data param is missing")
        if self.n_instances is None:
            raise ValueError("n_instances param is missing")
        unlabeled_data = self.unlabeled_data.reshape(len(self.unlabeled_data), -1)

        kmeans = MiniBatchKMeans(n_clusters=self.n_instances, random_state=0)
        kmeans.fit(unlabeled_data)

        query_idx = self.GetOneIndexOfEachCluster(kmeans, self.n_instances)
        unlabeled_data = unlabeled_data.reshape(len(unlabeled_data), 128, 128, 3)
        return query_idx, unlabeled_data[query_idx]

    def Random(self,list):
        secure_random = random.SystemRandom()
        return secure_random.choice(list)


    def GetOneIndexOfEachCluster(self,kmeans, n_clusters):
        indices = np.zeros(n_clusters)

        for i in range(n_clusters):
            lista = np.where(i == kmeans.labels_)[0]  # select images from one cluster/label
            valor = self.Random(lista)
            indices[i] = valor

        return indices.astype(int)


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

    def __init__(self, unlabeled_data: np.ndarray, n_instances: int) -> None:
        """
        Docs here
        """
        super().__init__(unlabeled_data, n_instances)

    def __call__(self,learner,X_pool, *args, **kwargs):
        """
        REPRESENTATIVE SAMPLING
        select n images calculating the representativity of each image between unlabeled and train images
        """

        self.unlabeled_data=X_pool
        if self.unlabeled_data is None:
            raise ValueError("unlabeled_data param is missing")
        if self.n_instances is None:
            raise ValueError("n_instances param is missing")

        len, length, height, depth = learner.X_training.shape
        vetor_train = learner.X_training.reshape((len, length * height * depth))

        len, length, height, depth = self.unlabeled_data.shape
        vetor_unlabeled =self.unlabeled_data.reshape((len, length * height * depth))

        train_similarity = pairwise_distances(vetor_unlabeled, vetor_train, metric="cosine")
        unlabeled_similarity = pairwise_distances(
            vetor_unlabeled, vetor_unlabeled, metric="cosine"
        )

        representativity = {}
        index = 0
        for train_sim, unlabeled_sim in zip(train_similarity, unlabeled_similarity):
            representativity[index] = np.mean(unlabeled_sim) - np.mean(train_sim)
            index = index + 1

        representativity = sorted(
            representativity.items(), key=lambda x: x[1], reverse=True
        )
        query_idx = []
        for r in representativity[:self.n_instances:]:
            query_idx.append(r[0])

        return query_idx, self.unlabeled_data[query_idx]

class Uncertainty_With_Clustering_Sampling(Query):
    """
    Docs here
    """

    def __init__(self, unlabeled_data: np.ndarray, n_instances: int) -> None:
        """
        Docs here
        """
        self.uncertainty=UncertaintySampling(unlabeled_data, 500)
        self.clustering=ClusterBasedSampling(unlabeled_data, n_instances)
        super().__init__(unlabeled_data, n_instances)
       
    def __call__(self,learner,X_pool, *args, **kwargs):
        """
        Least Confidence Sampling with Clustering-based Sampling

        Combining Uncertainty and Diversity sampling means applying one technique and then another.
        this allow select images in different positions of the boarder
        """
        self.unlabeled_data = X_pool
        if learner is None:
            raise ValueError("Learner param is missing")
        if self.unlabeled_data is None:
            raise ValueError("unlabeled_data param is missing")
        if self.n_instances is None:
            raise ValueError("n_instances param is missing")

        indices, instancias = self.uncertainty.__call__(learner,self.unlabeled_data)
        
        query_idx, data = self.clustering.__call__(learner,self.unlabeled_data)
        print(query_idx)
        return query_idx, self.unlabeled_data[query_idx]

class Representative_With_Clustering_Sampling(Query):
    """
    Docs here
    """

    def __init__(self, unlabeled_data: np.ndarray, n_instances: int) -> None:
        """
        Docs here
        """
        self.representative=RepresentativeSampling(unlabeled_data, 1)
        super().__init__(unlabeled_data, n_instances)
       
    def __call__(self,learner,X_pool, *args, **kwargs):
        """
        muito lento
        """
        
        self.unlabeled_data=X_pool

        if learner is None:
            raise ValueError("Learner param is missing")
        if self.unlabeled_data is None:
            raise ValueError("unlabeled_data param is missing")
        if self.n_instances is None:
            raise ValueError("n_instances param is missing")
        
        self.unlabeled_data = self.unlabeled_data.reshape(len(self.unlabeled_data), -1)

        kmeans = MiniBatchKMeans(n_clusters=self.n_instances, random_state=0)
        kmeans.fit(self.unlabeled_data)

        unlabeled_data = self.unlabeled_data.reshape(len(self.unlabeled_data), 128, 128, 3)
        indices = []

        for i in range(self.n_instances):
            lista = np.where(i == kmeans.labels_)[0]  # select images from one cluster/label
            query_idx, unlabeled_data[query_idx] = self.representative.__call__(learner, unlabeled_data[lista])
            indices.append(int(lista[query_idx]))

        return indices, unlabeled_data[indices]

class Highest_Entropy__Clustering_Sampling(Query):
    """
    Docs here
    """

    def __init__(self, unlabeled_data: np.ndarray, n_instances: int) -> None:
        """
        Docs here
        """
        self.representative=RepresentativeSampling(unlabeled_data, 1)
        self.n_clusters=10
        super().__init__(unlabeled_data, n_instances)
       
    def __call__(self,learner,X_pool, *args, **kwargs):
        """
        Sampling from the Highest Entropy Cluster
        Select n images from the cluster where the images have more entropy (average incertainty is bigger)
        """

        self.unlabeled_data=X_pool

        if learner is None:
            raise ValueError("Learner param is missing")
        if self.unlabeled_data is None:
            raise ValueError("unlabled_data param is missing")
        if self.n_instances is None:
            raise ValueError("n_instances param is missing")
       
        highest_average_Uncertainty = 1
        self.unlabeled_data = self.unlabeled_data.reshape(len(self.unlabeled_data), -1)

        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=0)
        kmeans.fit(self.unlabeled_data)

        self.unlabeled_data = self.unlabeled_data.reshape(len(self.unlabeled_data), 128, 128, 3)

        for i in range(self.n_clusters):
            lista = np.where(i == kmeans.labels_)[0]  # select images from cluster i
            probabilidades = learner.predict_proba(self.unlabeled_data[lista])
            incertezas = [abs(i[0] - i[1]) for i in probabilidades]
            average_Uncertainty = np.mean(incertezas)

            if average_Uncertainty < highest_average_Uncertainty:
                highest_average_Uncertainty = average_Uncertainty
                most_uncertain_cluster = i

        lista = np.where(most_uncertain_cluster == kmeans.labels_)[0]  # select images from one cluster/label
        
        print(lista.shape)
        indices = np.random.choice(lista, self.n_instances, replace=False)

        return indices, self.unlabeled_data[indices]

class Uncertainty_With_Representative_sampling(Query):
    """
    Docs here
    """

    def __init__(self, unlabeled_data: np.ndarray, n_instances: int) -> None:
        """
        Docs here
        """
        self.uncertainty=UncertaintySampling(unlabeled_data,500)
        self.representative=RepresentativeSampling(unlabeled_data,  n_instances)
      
        super().__init__(unlabeled_data, n_instances)
       
    def __call__(self,learner,X_pool, *args, **kwargs):
        
        self.unlabeled_data=X_pool

        if learner is None:
            raise ValueError("Learner param is missing")
        if self.unlabeled_data is None:
            raise ValueError("unlabeled_data param is missing")
        if self.n_instances is None:
            raise ValueError("n_instances param is missing")
        
        indices, instancias = self.uncertainty.__call__(
            learner,
            self.unlabeled_data,
            self.uncertainty.n_instances,
        )
        
        query_idx, data = self.representative.__call__(
            learner, instancias, self.representative.n_instances
        )
        return indices[query_idx], data

class Highest_Entropy__Uncertainty_Sampling(Query):
    """
    Docs here
    """

    def __init__(self, unlabeled_data: np.ndarray, n_instances: int) -> None:
        """
        Docs here
        """

        self.highest_entropy_clustering=Highest_Entropy__Clustering_Sampling(unlabeled_data,100)
        self.uncertainty=UncertaintySampling(unlabeled_data,n_instances)
      
        super().__init__(unlabeled_data, n_instances)
       
    def __call__(self,learner,X_pool, *args, **kwargs):
        

        if learner is None:
            raise ValueError("Learner param is missing")
        if self.unlabeled_data is None:
            raise ValueError("unlabeled_data param is missing")
        if self.n_instances is None:
            raise ValueError("n_instances param is missing")
        
        indices, instancias = self.highest_entropy_clustering.__call__(
            learner,self.unlabeled_data,
        )
        
        print(indices.shape)
        query_idx, data = self.uncertainty.__call__(
            learner, instancias, self.n_instances
        )

        print(query_idx.shape)
        return indices[query_idx], data







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
