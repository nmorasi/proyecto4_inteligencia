import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout,Flatten,Dense,Activation
from tensorflow.python.keras import backend as K 

K.clear_session()
data_entrenamiento = 'data/entrenamiento'
data_validacion = 'data/validacion'
altura = 100
longitud = 100
batch_s = 10
#tamano del filtro
#tamano del pool
#el numero de las clases 


#preprocesamiento de las imagenes 
entrenamiento_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip = True
)

validation_datagen = ImageDataGenerator(
    rescale = 1./255
)

##con estas se obtiene las imagenes 
imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size = (altura,longitud),
    batch_size = batch_s,
    class_mode = 'categorical'
)
imagen_validacion = validation_datagen.flow_from_directory(
    data_validacion,
    targe_size = (altura,longitud),
    batch_size = batch_s,
    class_mode = 'categorical')

#en donde esta el comando que le dice que tome las imagenes de los directorios 
