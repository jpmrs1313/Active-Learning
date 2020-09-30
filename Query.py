import numpy as np
import random
from modAL.uncertainty import uncertainty_sampling
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances
from keras import backend as K


#this method defines how select images
def query(learner,unlabled_data,validation_data, n_instances,diversity_strategy):
    
    if (diversity_strategy==cluster_based_sampling):   
        query_idx, unlabled_data[query_idx] = cluster_based_sampling(learner,unlabled_data, int(n_instances))     
    if (diversity_strategy==representative_sampling):   
        query_idx, unlabled_data[query_idx] = representative_sampling(learner,unlabled_data, int(n_instances))         
    if (diversity_strategy==outlier_sampling):   
        query_idx, unlabled_data[query_idx] = outlier_sampling(learner,unlabled_data,validation_data, int(n_instances))             
    if (diversity_strategy==randomly):   
        query_idx, unlabled_data[query_idx] = randomly(learner,unlabled_data,int(n_instances))                 
    
    #indices, instancias = learner.query(unlabled_data, int(n_instances/2)) 
    #query_idx=np.append(query_idx,indices)
    #unlabled_data=np.append(unlabled_data[query_idx],instancias)
        
    return query_idx, unlabled_data
    
    #return  Highest_Entropy__Uncertainty_sampling(learner,unlabled_data)


'''
RANDOMLY
select images randomly
'''

def randomly(learner, X, n_instances=1):
    query_idx = np.random.choice(range(len(X)), size=n_instances, replace=False)
    return query_idx, X[query_idx]

def Random(list):
    secure_random = random.SystemRandom()
    return secure_random.choice(list)


'''
CLUSTERING BASED SAMPLING
group images in n clusters and select randomly one image for each clusters
'''

def GetOneIndexOfEachCluster(kmeans, n_clusters):
    indices=np.zeros(n_clusters)

    for i in range(n_clusters):
        lista=np.where(i == kmeans.labels_)[0]#select images from one cluster/label
        valor=Random(lista)
        indices[i]=valor

    return indices.astype(int)


def cluster_based_sampling(learner,unlabled_data, n_instances):
     unlabled_data = unlabled_data.reshape(len(unlabled_data),-1)
     
     kmeans = MiniBatchKMeans(n_clusters=n_instances, random_state=0)
     kmeans.fit(unlabled_data)  
     
     query_idx=GetOneIndexOfEachCluster(kmeans,n_instances)
     unlabled_data = unlabled_data.reshape(len(unlabled_data),128, 128,3)
     return query_idx, unlabled_data[query_idx]



'''
OUTLIER BASED SAMPLING
selects the n most unknown images by the model

The method can generate outliers that are similar to each other and therefore lack
diversity within an Active Learning iteration.
'''

def get_rank(value, rankings):
        """ get the rank of the value in an ordered array as a percentage 
    
        Keyword arguments:
            value -- the value for which we want to return the ranked value
            rankings -- the ordered array in which to determine the value's ranking
        
        returns linear distance between the indexes where value occurs, in the
        case that there is not an exact match with the ranked values    
        """
        
        index = 0 # default: ranking = 0
        
        for ranked_number in rankings:
            if value < ranked_number:
                break #NB: this O(N) loop could be optimized to O(log(N))
            index += 1        
        
        if(index >= len(rankings)):
            index = len(rankings) # maximum: ranking = 1
            
        elif(index > 0):
            # get linear interpolation between the two closest indexes 
            
            diff = rankings[index] - rankings[index - 1]
            perc = value - rankings[index - 1]
            linear = perc / diff
            index = float(index - 1) + linear
        
        absolute_ranking = index / len(rankings)
    
        return(absolute_ranking)
   
    
def get_validation_rankings(model, validation_data):
        validation_rankings = [] # 2D array, every neuron by ordered list of output on validation data per neuron    
        v=0
        for item in validation_data:
            
            item=item[np.newaxis,...]
            #get logit of item

            keras_function = K.function([model.input], [model.get_layer('dense_2').output])
            neuron_outputs=keras_function([item, 1])

            # initialize array if we haven't yet
            if len(validation_rankings) == 0:
                for output in neuron_outputs:
                    validation_rankings.append([0.0] * len(validation_data))
                        
            n=0
            for output in neuron_outputs:
                validation_rankings[n][v] = output
                n += 1
                        
            v += 1
        
        # Rank-order the validation scores 
        v=0
        for validation in validation_rankings:
            validation.sort() 
            validation_rankings[v] = validation
            v += 1
          
        return validation_rankings


def outlier_sampling(learner,unlabled_data,validation_data, n_instances):
    model=learner.estimator.model
    # Get per-neuron scores from validation data
    validation_rankings = get_validation_rankings(model, validation_data)
    
    index=0
    #outliers = {}
    outliers_rank={}
    for item in unlabled_data:

        item=item[np.newaxis,...]
        #get logit of item
    
        keras_function = K.function([model.input], [model.get_layer('dense_2').output])
        neuron_outputs=keras_function([item, 1])

           
        n=0
        ranks = []
        for output in neuron_outputs:
            rank = get_rank(output, validation_rankings[n])
            ranks.append(rank)
            n += 1 
        
        outliers_rank[index] = 1 - (sum(ranks) / len(neuron_outputs)) # average rank
        index=index+1
            
    outliers_rank = sorted(outliers_rank.items(), key=lambda x: x[1], reverse=True) 
      
    query_idx=[]
    for outlier in outliers_rank[:n_instances:]:
        query_idx.append(outlier[0])
                            
    return query_idx, unlabled_data[query_idx]    



'''
REPRESENTATIVE SAMPLING
select n images calculating the representativity of each image between unlabled and train images
'''


def representative_sampling(learner,unlabled_data, n_instances):
   
    len, length, height, depth = learner.X_training.shape
    vetor_train=learner.X_training.reshape((len,length * height * depth))
    
    len, length, height, depth = unlabled_data.shape
    vetor_unlabled=unlabled_data.reshape((len,length * height * depth))
    
    train_similarity=pairwise_distances(vetor_unlabled, vetor_train, metric='cosine')
    unlabled_similarity=pairwise_distances(vetor_unlabled, vetor_unlabled, metric='cosine')
    
    representativity={}
    index=0
    for train_sim, unlabled_sim in zip(train_similarity, unlabled_similarity):
        representativity[index] = np.mean(unlabled_sim) - np.mean(train_sim) 
        index=index+1
   
    
    representativity = sorted(representativity.items(), key=lambda x: x[1], reverse=True)       
    query_idx=[]
    for r in representativity[:n_instances:]:
        query_idx.append(r[0])
                            
    return query_idx, unlabled_data[query_idx]    




'''
Least Confidence Sampling with Clustering-based Sampling
 
Combining Uncertainty and Diversity sampling means applying one technique and then another.
this allow select images in different positions of the boarder
'''

def Uncertainty_With_Clustering_sampling(learner,unlabled_data, n_uncertainty_instances=500, n_final_instances=20):
    indices, instancias = learner.query(unlabled_data, n_uncertainty_instances)
    query_idx, data = cluster_based_sampling(learner,instancias, n_final_instances)  
    return indices[query_idx],data



'''
Uncertainty Sampling with Model-based Outliers 

When Combining Uncertainty Sampling with Model-based Outliers, you are maximizing your model’s current confusion.
You are looking for items near the decision boundary and making sure that their features are relatively unknown
to the current model. 
'''

'''MODEL BASED OUTLIER MUITO LENTO'''
'''MODEL BASED OUTLIER MUITO LENTO'''
'''MODEL BASED OUTLIER MUITO LENTO'''
'''MODEL BASED OUTLIER MUITO LENTO'''
'''MODEL BASED OUTLIER MUITO LENTO'''

def Uncertainty_With_ModelOutliers_sampling(learner,unlabled_data,validation_data, n_uncertainty_instances=500, n_final_instances=20):
    indices, instancias = learner.query(unlabled_data, n_uncertainty_instances)
    query_idx, data = outlier_sampling(learner,instancias,validation_data, n_final_instances)             
    return indices[query_idx,], data



'''
Uncertainty Sampling with Model-based Outliers and Clustering

the previous method might over-sample items that are very close to each other,
you might want to implement this strategy and then clustering to ensure diversity. 
'''

def Uncertainty_ModelOutliers_and_Clustering(learner,unlabled_data,validation_data, n_first_instances=500, n_final_instances=10):
    indices, instancias=Uncertainty_With_ModelOutliers_sampling(learner,unlabled_data,validation_data, n_first_instances, n_final_instances=50)
    query_idx, data = cluster_based_sampling(learner,instancias, n_final_instances)  
    return indices[query_idx,], data



'''
Representative Sampling Cluster-based Sampling 
'''

def Representative_With_Clustering_sampling(learner,unlabled_data,n_final_instances=20):
  
    unlabled_data = unlabled_data.reshape(len(unlabled_data),-1)
  
    kmeans = KMeans(n_clusters=n_final_instances, random_state=0)
    kmeans.fit(unlabled_data) 
     
    unlabled_data = unlabled_data.reshape(len(unlabled_data),128, 128,3)
    indices=[]
    
    for i in range(n_final_instances):
       lista=np.where(i == kmeans.labels_)[0]#select images from one cluster/label
       query_idx, unlabled_data[query_idx] = representative_sampling(learner,unlabled_data[lista], 1)
       indices.append(int(lista[query_idx]))
            
    return indices, unlabled_data[indices]



'''
Sampling from the Highest Entropy Cluster 
'''

def Highest_Entropy__Clustering_sampling(learner,unlabled_data,n_final_instances=20,n_clusters=10):
   
    highest_average_uncertainty=1
    unlabled_data = unlabled_data.reshape(len(unlabled_data),-1)
  
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(unlabled_data) 
     
    unlabled_data = unlabled_data.reshape(len(unlabled_data),128, 128,3)
    
    for i in range(n_clusters):
        lista=np.where(i == kmeans.labels_)[0]#select images from one cluster/label
        probabilidades =learner.predict_proba(unlabled_data[lista])
        incertezas = [abs(i[0]-i[1]) for i in probabilidades]  
        average_uncertainty = np.mean(incertezas)
        
        if (average_uncertainty < highest_average_uncertainty):
            highest_average_uncertainty = average_uncertainty
            most_uncertain_cluster = i
     
    lista=np.where(most_uncertain_cluster == kmeans.labels_)[0]#select images from one cluster/label  
    indices=np.random.choice(lista,n_final_instances,replace=False)

    return indices, unlabled_data[indices]



'''
Uncertainty Sampling and Representative Sampling
'''

def Uncertainty_With_Representative_sampling(learner,unlabled_data, n_uncertainty_instances=500, n_final_instances=20):
    indices, instancias = learner.query(unlabled_data, n_uncertainty_instances)
    query_idx, data = representative_sampling(learner,instancias, n_final_instances)  
    return indices[query_idx],data



'''
Model-based Outliers and Representative Sampling
'''

def Model_Outliers_With_Representative_sampling(learner,unlabled_data,validation_data, n_outliers_instances=500, n_final_instances=20):
    indices, instancias = outlier_sampling(learner,instancias,validation_data,n_outliers_instances) 
    query_idx, data = representative_sampling(learner,instancias,n_final_instances)  
    return indices[query_idx],data 



'''
Clustering with itself for hierarchical clusters
NAO ACHO QUE FAÇA SENTIDO IMPLEMENTAR ESTE MÉTODO
'''


'''
Highest Entropy Cluster with Margin of Confidence Sampling
'''
def Highest_Entropy__Uncertainty_sampling(learner,unlabled_data, n_highest_entropy=100, n_final_instances=20,n_clusters=10):
    indices, instancias = Highest_Entropy__Clustering_sampling(learner,unlabled_data,n_highest_entropy,n_clusters) 
    query_idx, data = learner.query(instancias, n_final_instances)
    return indices[query_idx], data


'''
Combining Ensemble Methods or Dropouts with individual strategies
Nao percebi este método
'''






'''
Expected Error Reduction Sampling
Tal como diz no livro, não é recomendado para o uso de neural networks, too slow
'''










