import tensorflow as tf
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels) , (test_images,test_labels) = fashion_mnist.load_data()
#labels are an array of integers
#images are 28x28 numPy arrays
##como son los datos
#train_images. shape
#son 60000 imagenes de 28 por 28
train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure(figsize=(10,10))
for i in range(0,25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
#el segundo tiene 128 neuronas 
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation= tf.nn.relu),
    keras.layers.Dense(10,activation = tf.nn.softmax)])
#compilar el modelo
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
##fraccion de las imagenes
##que se clasificaron correctamente

#para entrenar el modelo se hace
model.fit(train_images,train_labels,epochs = 5)
#para evaluar que tan bien se comporta se utiliza evaluate 
test_loss,test_acc = model.evaluate(test_images,test_labels) 

#ahora para para hacer predicciones se hace con la funcion
predictions = model.predict(test_images)
#es un arreglo de las etiquetas donde cada una corresponde con las
#clases 
