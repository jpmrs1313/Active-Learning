import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Estimators import *
from Query import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from modAL.models import ActiveLearner


def AL(X_test,
        y_test,
        X_initial,
        y_initial,
        X_unlabled,
        X_validation,
        estimator,
        uncertainty_strategy,
        diversity_strategy):

    learner = create_active_learning_model(estimator,X_initial,y_initial,uncertainty_strategy)            
    
    accuracy_values=AL_Loop(learner,X_test, y_test,X_unlabled,X_validation,uncertainty_strategy,diversity_strategy)
    return accuracy_values


def create_active_learning_model(estimator,
                                X_initial,
                                y_initial,
                                uncertainty_strategy,
                                verbose=0,):
     
    learner = ActiveLearner(estimator=estimator,
                            X_training=X_initial,
                            y_training=y_initial,
                            query_strategy= uncertainty_strategy,
                            verbose=verbose)  
    return learner


def AL_Loop(learner,X_test, y_test,X_unlabled,X_validation,
                                            uncertainty_strategy,
                                            diversity_strategy,
                                            epochs=50,
                                            batch_size=32,
                                            accuracy_goal=0.95,
                                            n_instances=20,
                                            verbose=0):
    index=0
    model_accuracy=0
    #accuracy of model with initialize images
    model_accuracy = learner.score(X_test, y_test, verbose=0)
    print('\nAccuracy after query {n}: {acc:0.4f}'.format(n=index, acc=model_accuracy))
    
    accuracy_values = np.zeros(0)
    accuracy_values=np.append(accuracy_values,model_accuracy)
    
    while model_accuracy < accuracy_goal:
        index=index+1
        
        #select images to label 
        query_idx, query_instance= query(learner,X_unlabled,X_validation, n_instances,diversity_strategy) 
                     
        new_y=np.zeros(n_instances)
        i=0
        for x in query_idx:
            # labelling images 
            plt.imshow(X_unlabled[x])
            plt.show()
            print("Informativa = 0 / Nao Informativa =1 ?")
            new_y[i] = np.array([int(input())], dtype=int)
            i=i+1
        
        #add labeled images and teach model
        learner.teach(X_unlabled[query_idx],new_y, epochs=epochs, batch_size=batch_size, verbose=verbose) 
        learner.estimator.layers[0].trainable=True
        learner.teach(X_unlabled[query_idx],new_y, epochs=epochs, batch_size=batch_size, verbose=verbose) 
        learner.estimator.layers[0].trainable=False
        
        
        model_accuracy = learner.score(X_test, y_test, verbose=0)
        accuracy_values=np.append(accuracy_values,model_accuracy)
        X_unlabled = np.delete(X_unlabled, query_idx, axis=0)
        print('\nAccuracy after query {n}: {acc:0.4f}'.format(n=index, acc=model_accuracy))

    return accuracy_values



#MAIN CODE
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(r"C:\Users\Soares\Desktop\UTAD\Projeto de licenciatura\Mucosa\Datasets\train", target_size=(128,128), batch_size=1000, class_mode='binary')
test_generator = datagen.flow_from_directory(r"C:\Users\Soares\Desktop\UTAD\Projeto de licenciatura\Mucosa\Datasets\test_set_complete", target_size=(128,128), batch_size=2500, class_mode='binary')
validation_generator = datagen.flow_from_directory(r"C:\Users\Soares\Desktop\UTAD\Projeto de licenciatura\Mucosa\Datasets\validation" , target_size=(128,128), batch_size=1000, class_mode='binary')
unlabled_generator = datagen.flow_from_directory(r"C:\Users\Soares\Desktop\UTAD\Projeto de licenciatura\Mucosa\Datasets\unlabeled" , target_size=(128,128), batch_size=5000)


for X_initial, y_initial in train_generator:
   break
for X_test, y_test in test_generator:
   break
for X_validation, y_validation in validation_generator:
   break
for X_unlabled,y_unlabled in unlabled_generator:
   break


estimator = KerasClassifier(create_xception)        
#acc_values = AL( X_test, y_test, X_initial, y_initial,X_unlabled,X_validation,estimator,uncertainty_sampling,representative_sampling)