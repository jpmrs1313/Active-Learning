from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras import layers, models, optimizers
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3

input_shape=(128,128,3)



#[0.21477761774352103, 0.9090909361839294]
#0.9512
def create_cnn():
   model = models.Sequential()
   model.add(layers.Conv2D(32, (4, 4), activation='relu',  input_shape=input_shape))
   model.add(layers.Conv2D(32, (4, 4), activation='relu'))
   model.add(layers.Conv2D(32, (4, 4), activation='relu'))
   model.add(layers.MaxPooling2D(pool_size=(2, 2)))
   model.add(layers.Dropout(0.25))
   model.add(layers.Flatten())
   model.add(layers.Dense(128, activation='relu'))
   model.add(layers.Dropout(0.5))
   model.add(layers.Dense(1, activation='sigmoid'))
   model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
   return model



#0.336735556135962, 0.8831169009208679]
#0.9308
def create_vgg16():
   conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=input_shape)

   model = models.Sequential()
   model.add(conv_base)
   model.add(layers.Flatten())
   model.add(layers.Dropout(0.5))
   model.add(layers.Dense(256, activation='relu'))
   model.add(layers.Dense(1, activation='sigmoid'))
    
   conv_base.trainable = False
    
   model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
   return model



#[0.22633942623030057, 0.9090909361839294]
#0.9184
def create_vgg19():
   conv_base = VGG19(weights='imagenet',
                  include_top=False,
                  input_shape=input_shape)

   model = models.Sequential()
   model.add(conv_base)
   model.add(layers.Flatten())
   model.add(layers.Dropout(0.5))
   model.add(layers.Dense(256, activation='relu'))
   model.add(layers.Dense(1, activation='sigmoid'))
    
   conv_base.trainable = False
    
   model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
   return model


#1.3407960273486712, 0.5021644830703735
#0.9940
def create_resnet50():
   conv_base = ResNet50(weights='imagenet',
                  include_top=False,
                  input_shape=input_shape)

   model = models.Sequential()
   model.add(conv_base)
   model.add(layers.Flatten())
   model.add(layers.Dropout(0.5))
   model.add(layers.Dense(256, activation='relu'))
   model.add(layers.Dense(1, activation='sigmoid'))
    
   conv_base.trainable = False
    
   model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
   return model



#[3.702155284551315, 0.8484848737716675]
#0.9908
def create_resnet50v2():
   conv_base = ResNet50V2(weights='imagenet',
                  include_top=False,
                  input_shape=input_shape)

   model = models.Sequential()
   model.add(conv_base)
   model.add(layers.Flatten())
   model.add(layers.Dropout(0.5))
   model.add(layers.Dense(256, activation='relu'))
   model.add(layers.Dense(1, activation='sigmoid'))
    
   conv_base.trainable = False
    
   model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
   return model



#[3.558937101653128, 0.6320346593856812]
#0.9736
def create_xception():
   conv_base = Xception(weights='imagenet',
                  include_top=False,
                  input_shape=input_shape)

   model = models.Sequential()
   model.add(conv_base)
   model.add(layers.Flatten())
   model.add(layers.Dropout(0.5))
   model.add(layers.Dense(256, activation='relu'))
   model.add(layers.Dense(1, activation='sigmoid'))
    
   conv_base.trainable = False
    
   model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
   return model



#[1.4496736010431728, 0.7402597665786743]
#0.8640
def create_inceptionv3():
   conv_base = InceptionV3(weights='imagenet',
                  include_top=False,
                  input_shape=input_shape)

   model = models.Sequential()
   model.add(conv_base)
   model.add(layers.Flatten())
   model.add(layers.Dropout(0.5))
   model.add(layers.Dense(256, activation='relu'))
   model.add(layers.Dense(1, activation='sigmoid'))
    
   conv_base.trainable = False
    
   model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
   return model