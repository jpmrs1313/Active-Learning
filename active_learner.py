import numpy as np
import matplotlib.pyplot as plt
from modAL.models import ActiveLearner
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.image import save_img
import os

from query import Query


class ActiveLearner(ActiveLearner):
    def __init__(
        self,
        build_fn,
        query_strategy: Query,
        X_training,
        y_training,
        **fit_kwargs,
    ) -> None:

        '''
        initiate Active learning instance with Deep Learning classifier,
        AL technic, X and Y training data
        '''
        super().__init__(
            KerasClassifier(build_fn),
            query_strategy,
            X_training,
            y_training,
            **fit_kwargs,
        )

    def loop(
        self,
        X_test,
        y_test,
        X_unlabeled,
        accuracy_goal,
    ):

        # accuracy of model with initialize images
        model_accuracy = self.score(X_test, y_test, verbose=0)
        print("\nAccuracy after query {n}: {acc:0.4f}".format(n=0, acc=model_accuracy))

        accuracy_values = [model_accuracy]

        i=1
        while (model_accuracy<accuracy_goal):

            query_idx, _ = self.query(X_unlabeled)
            new_y = [self.label(X_unlabeled, indice) for indice in query_idx]

            self.train(X_unlabeled[query_idx], new_y)

            model_accuracy = self.score(X_test, y_test, verbose=0)
            accuracy_values.append(model_accuracy)

            self.save(X_unlabeled[query_idx], new_y)

            X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
            print(
                "\nAccuracy after query {n}: {acc:0.4f}".format(
                    n=i, acc=model_accuracy
                )
            )
            i=i+1
        

        return accuracy_values

    def train(
        self,
        images,
        labels,
        epochs=10,
        batch_size=32,
    ):


        self.teach(
            images,
            labels,
            epochs=epochs,
            batch_size=batch_size,
        )

    def label(
        self,
        images,
        indice,
    ):
        # labelling images
        plt.imshow(images[indice])
        plt.show()
        print(
            "Informativa = 0 / Nao Informativa =1 ? 2 para rever os frames anteriores"
        )
        new_y = int(input())

        # verificar se pretende ver as 10 anteriores e guarda a anotação após visualizar as anteriores
        if new_y == 2:
            self.get_previous_frames(images, indice)
            new_y = self.label(images, indice)

        return new_y

    def show_images(
        self,
        images,
        cols=1,
        titles=None,
    ):
        assert (titles is None) or (len(images) == len(titles))
        n_images = len(images)
        if titles is None:
            titles = ["Image (%d)" % i for i in range(1, n_images + 1)]
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        plt.show()

    def get_previous_frames(
        self,
        X_unlabeled,
        indice,
    ):
        if indice > 10:
            imagens = []
            for imagem in zip(range(indice - 10, indice)):
                imagens.append(X_unlabeled[imagem])
            self.show_images(imagens)

        label = int(input())
        return label

    def save(self,images,labels):
        self.estimator.model.save_weights("model.h5")

        list = os.listdir('images/informativa') # dir is your directory path
        number_files = len(list)

        list = os.listdir('images/nao_informativa') # dir is your directory path
        number_files += len(list)

        if not os.path.exists('images/informativa'):
                    os.makedirs('images/informativa')

        if not os.path.exists('images/nao_informativa'):
                    os.makedirs('images/nao_informativa')
       
        for x,y in zip(images,labels):
            number_files += 1
            
            if(y==0) : file_path=('images/informativa/' + str(number_files) + '.png')
            if(y==1) : file_path=('images/nao_informativa/' + str(number_files) + '.png')

            save_img(file_path, x) 

