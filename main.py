from query import RandomSampling,UncertaintySampling, ClusterBasedSampling,RepresentativeSampling,Uncertainty_With_Clustering_Sampling,Representative_With_Clustering_Sampling,Highest_Entropy__Clustering_Sampling,Uncertainty_With_Representative_sampling,Highest_Entropy__Uncertainty_Sampling
from keras.preprocessing.image import ImageDataGenerator
from active_learner import ActiveLearner
import os
import json
from keras import layers, models, optimizers
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3

input_shape = (128, 128, 3)

def create_cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (4, 4), activation="relu", input_shape=input_shape))
    model.add(layers.Conv2D(32, (4, 4), activation="relu"))
    model.add(layers.Conv2D(32, (4, 4), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=["acc"],
    )
    return model

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# read file
with open("especificacoes.json", "r") as myfile:
    data = myfile.read()
# parse file
obj = json.loads(data)

datagen = ImageDataGenerator(rescale=1.0 / 255) 
train_generator = datagen.flow_from_directory(
    str(obj["training_path"]),
    target_size=(128, 128),
    batch_size=200,
    class_mode="binary",
    shuffle=False,
)
test_generator = datagen.flow_from_directory(
    str(obj["testing_path"]),
    target_size=(128, 128),
    batch_size=2200,
    class_mode="binary",
    shuffle=False,
)
validation_generator = datagen.flow_from_directory(
    str(obj["validation_path"]),
    target_size=(128, 128),
    batch_size=200,
    class_mode="binary",
    shuffle=False,
)
unlabeled_generator = datagen.flow_from_directory(
    str(obj["unlabeled_path"]),
    target_size=(128, 128),
    batch_size=5000,
    class_mode="binary",
    shuffle=False,
)

X_initial, y_initial = next(train_generator)
X_test, y_test = next(test_generator)
X_validation, y_validation = next(validation_generator)
X_unlabeled, y_unlabeled = next(unlabeled_generator)

learner = ActiveLearner(
    locals()[obj["build_fn"]],
    Highest_Entropy__Uncertainty_Sampling(X_unlabeled, int(obj["n_instances"])),
    X_initial,
    y_initial,
    verbose=0,
)

accuracy_values = learner.loop(
    X_test,
    y_test,
    X_unlabeled,
    float(obj["accuracy"]),
)
