from query import (
    ClusterBasedSampling,
    RandomSampling,
    UncertaintySampling,
    RepresentativeSampling,
    UncertaintyWithClusteringSampling,
    RepresentativeWithClusteringSampling,
    HighestEntropyClusteringSampling,
    UncertaintyWithRepresentativeSampling,
    HighestEntropyUncertaintySampling
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from active_learner import ActiveLearner
import os
import json
from estimators import *

input_shape = (128, 128, 3)


os.environ["TF_FORCE_GPU_A_LLOW_GROWTH"] = "true"

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
    batch_size=1000,
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
    batch_size=2200,
    class_mode="binary",
    shuffle=False,
)

X_initial, y_initial = next(train_generator)
X_test, y_test = next(test_generator)
X_validation, y_validation = next(validation_generator)
X_unlabeled, y_unlabeled = next(unlabeled_generator)

learner = ActiveLearner(
    locals()[obj["build_fn"]],
    HighestEntropyClusteringSampling(int(obj["n_instances"])),
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
