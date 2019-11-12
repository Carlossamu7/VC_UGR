# -*- coding: utf-8 -*-

#########################################################################
############ CARGAR LAS LIBRERÍAS NECESARIAS ############################
#########################################################################

# Importar librerı́as necesarias
import numpy as np
import keras
import matplotlib . pyplot as plt
import keras . utils as np_utils
# Importar modelos y capas que se van a usar

# Importar el optimizador a usar
from keras . optimizers import SGD
# Importar el conjunto de datos
from keras . datasets import cifar100

#########################################################################
######## FUNCIÓN PARA CARGAR Y MODIFICAR EL CONJUNTO DE DATOS ###########
#########################################################################

""" Sólo se le llama una vez. Devuelve 4 vectores conteniendo, por este orden, las imágenes de entrenamiento,
    las clases de las imagenes de entrenamiento, las imágenes del conjunto de test y las clases del conjunto de test.
"""
def cargarImagenes():
    # Cargamos Cifar100. Cada imagen tiene tamaño (32 , 32 , 3). Nos quedamos con las imágenes de 25 de las clases.
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    train_idx = np.isin(y_train, np.arange (25))
    train_idx = np.reshape(train_idx, -1)
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    test_idx = np.isin(y_test, np.arange (25))
    test_idx = np.reshape(test_idx, -1)
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]

    # Transformamos los vectores de clases en matrices. Cada componente se convierte en un vector de
    # ceros con un uno en la componente correspondiente a la clase a la que pertenece la imagen.
    y_train = np_utils.to_categorical(y_train, 25)
    y_test = np_utils.to_categorical(y_test, 25)

    return x_train, y_train, x_test, y_test

#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

""" Devuelve el accuracy de un modelo (porcentaje de etiquetas bien predichas frente al total).
- labels: vector de etiquetas verdaderas.
- preds: vector de etiquetas predichas.
"""
def calcularAccuracy (labels, preds):
    labels = np.argmax(labels, axis = 1)
    preds = np.argmax(preds, axis = 1)

    accuracy = sum (labels == preds)/ len(labels)

    return accuracy

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################


""" Pinta dos gráficas, una con la evolución de la función de pérdida en el conjunto de train y en el de validación,
    y otra con la evolución del accuracy en el conjunto de train y el de validación.
- hist: historial del entrenamiento del modelo (salida de fit () y fit_generator ()).
"""
def mostrarEvolucion(hist):
    loss = hist.history ['loss']
    val_loss = hist.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()

    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    plt.plot(acc)
    plt.plot(val_acc )
    plt.legend(['Training accuracy', 'Validation accuracy'])
    plt.show()

#########################################################################
################## DEFINICIÓN DEL MODELO BASENET ########################
#########################################################################

# Las imágenes son en color de 3 canales de 32x32 píxeles.
img_rows, img_cols = 32, 32
# Nos piden que número de clases sean 25
num_classes = 25
# Elegimos un tamaño de batch potencia de 2
batch_size = 128
# Elegimos un número de épocas aleatorio
epochs = 10

# Cargamos las imagenes
(x_train, y_train), (x_test, y_test) = cargarImagenes()
# Ponemos la dimensión
input_shape = (img_rows, img_cols, 3)
# El modelo es Sequential fuerza a que todas las capas de la red vayan una detrás
# de otra de forma secuencial, sin permitir ciclos ni saltos entre las capas.
model = Sequential()
# Añadimos una capa convolucional con:
#   - Canales de salida: 6
#   - Tamaño del kernel: 5
#   - Activacion relu
model.add(Conv2D(6, kernel_size=(5,5), activation='relu', input_shape=input_shape))
# Añadimos una capa MaxPooling con:
#   - Tamaño del kernel: 2
model.add(MaxPooling2D(pool_size=(2,2)))
# Añadimos una capa convolucional con:
#   - Canales de salida: 16
#   - Tamaño del kernel: 5
#   - Activacion relu
model.add(Conv2D(16, kernel_size=(5,5), activation='relu'))
# Añadimos una capa MaxPooling con:
#   - Tamaño del kernel: 2
model.add(MaxPooling2D(pool_size=(2,2)))
# Aplanamos la salida
model.add(Flatten())
# Definimos una capa fully connected con 50 neuronas
model.add(Dense(50, activation='relu'))
# Definimos como última capa una capa fully connected con tantas neuronas como
# clases tenga el problema (25) y una activación softmax para transformar las
# salidas de las neuronas en la probabilidad de pertenecer a cada clase.
model.add(Dense(25, activation='softmax'))
# Para ver una descripción del modelo
model.summary()

#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

# Usamos gradiente descendente estocástico (SGD) con los parámetros siguientes:
opt = SGD (lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
# Definimos la función de pérdida o función objetivo que se va a usar (la que
# se va a minimizar). Como estamos en clasificación multiclase usamos
# categorical_crossentropy también se puede especificar con el argumento metrics
# las métricas que se quieren calcular a lo largo de todas las épocas de entrenamiento.
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Una vez tenemos el modelo base, y antes de entrenar, vamos a guardar los
# pesos aleatorios con los que empieza la red, para poder reestablecerlos
# después y comparar resultados entre no usar mejoras y sí usarlas.
weights = model.get_weights()

#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################

# Entrenamos el modelo con fit que recibe las imágenes de entrenamiento directamente.
histograma = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

mostrarEvolucion(histograma)
score = model.evaluate(x_test, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Reestablecemos los pesos  antes del siguiente entrenamiento usando
model.set_weights(weights)

#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

# A completar

#########################################################################
########################## MEJORA DEL MODELO ############################
#########################################################################

# A completar. Tanto la normalización de los datos como el data
# augmentation debe hacerse con la clase ImageDataGenerator.
# Se recomienda ir entrenando con cada paso para comprobar
# en qué grado mejora cada uno de ellos.
