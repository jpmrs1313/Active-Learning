from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf

input_shape = (128, 128, 3)
tensor = tf.keras.Input(input_shape)


def create_cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (4, 4), activation="relu"))
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


def create_vgg16():
    conv_base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    conv_base.trainable = False

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=["acc"],
    )
    return model


def create_vgg19():
    conv_base = VGG19(weights="imagenet", include_top=False, input_shape=input_shape)

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    conv_base.trainable = False

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.RMSprop(lr=0.0001),
        metrics=["acc"],
    )
    return model


def create_resnet50():
    conv_base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    conv_base.trainable = False

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=["acc"],
    )
    return model


def create_resnet50v2():
    conv_base = ResNet50V2(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    conv_base.trainable = False

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=["acc"],
    )
    return model


def create_xception():
    conv_base = Xception(weights="imagenet", include_top=False, input_shape=input_shape)

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    conv_base.trainable = False

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=["acc"],
    )
    return model


def create_inceptionv3():
    conv_base = InceptionV3(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    conv_base.trainable = False

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=["acc"],
    )
    return model
