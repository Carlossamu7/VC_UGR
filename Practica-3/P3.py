#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Carlos Sánchez Muñoz
@date: 4 de diciembre de 2019
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

import sys
import math
import copy


""" Uso la notación Snake Case la cual es habitual en Python """

########################################
###   FUNCIONES DE OTRAS PRÁCTICAS   ###
########################################

""" Lee una imagen ya sea en grises o en color. Devuelve la imagen.
- file_name: archivo de la imagen.
- flag_color (op): modo en el que se va a leer la imagen -> grises o color. Por defecto será en color.
"""
def leer_imagen(file_name, flag_color = 1):
    if flag_color == 0:
        print("Leyendo " + file_name + " en gris")
    elif flag_color==1:
        print("Leyendo " + file_name + " en color")
    else:
        print("flag_color debe ser 0 o 1")

    img = cv2.imread(file_name, flag_color)

    if flag_color==1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #img = img.astype(np.float32)
    return img

""" Normaliza una matriz.
- image: matriz a normalizar.
- image_title (op): título de la imagen. Por defecto ' '.
"""
def normaliza(image, image_title = " "):
    # En caso de que los máximos sean 255 o las mínimos 0 no iteramos en los bucles
    if len(image.shape) == 2:
        max = np.amax(image)
        min = np.amin(image)
        if max>255 or min<0:
            print("Normalizando imagen '" + image_title + "'")
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    image[i][j] = (image[i][j]-min)/(max-min) * 255
    elif len(image.shape) == 3:
        max = [np.amax(image[:,:,0]), np.amax(image[:,:,1]), np.amax(image[:,:,2])]
        min = [np.amin(image[:,:,0]), np.amin(image[:,:,1]), np.amin(image[:,:,2])]

        if max[0]>255 or max[1]>255 or max[2]>255 or min[0]<0 or min[1]<0 or min[2]<0:
            print("Normalizando imagen '" + image_title + "'")
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    for k in range(image.shape[2]):
                        image[i][j][k] = (image[i][j][k]-min[k])/(max[k]-min[k]) * 255

    return image

""" Imprime una imagen a través de una matriz.
- image: imagen a imprimir.
- flag_color (op): bandera para indicar si la imagen es en B/N o color. Por defecto color.
- image_title(op): título de la imagen. Por defecto 'Imagen'
- window_title (op): título de la ventana. Por defecto 'Ejercicio'
"""
def pintaI(image, flag_color=1, image_title = "Imagen", window_title = "Ejercicio"):
    image = normaliza(image, image_title)               # Normalizamos la matriz
    image = image.astype(np.uint8)
    plt.figure(0).canvas.set_window_title(window_title) # Ponemos nombre a la ventana
    if flag_color == 0:
        plt.imshow(image, cmap = "gray")
    else:
        plt.imshow(image)
    plt.title(image_title)              # Ponemos nombre a la imagen
    plt.show()
    image = image.astype(np.float64)    # Devolvemos su formato

    #cv2.imshow(image_title, image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

""" Lee una lista de imágenes ya sea en grises o en color. Devuelve la lista de imágenes leída.
- image_list: lista de imágenes a concatenar.
- flag_color (op): modo en el que se van a leer las imágenes. Por defecto en color.
"""
def leer_lista_imagenes(file_name_list, flag_color = 1):
    image_list = []

    for i in file_name_list:
        img = leer_imagen(i, flag_color)
        image_list.append(img)

    return image_list

""" Muestra múltiples imágenes en una ventena Matplotlib.
- image_list: La lista de imágenes.
- image_title_list: Lista de títulos de las imágenes.
- rows: filas.
- columns: columnas.
- flag_color (op): bandera para indicar si la imagen es en B/N o color. Por defecto color.
- window_title (op): título de la ventana. Por defecto 'Imágenes con títulos'
"""
def imprimir_imagenes_titulos(image_list, image_title_list, rows, columns, flag_color = 1, window_title = 'Imágenes con títulos'):
    # Se igualan los tamaños de las Imagenes
    fig = plt.figure(0)
    fig.canvas.set_window_title(window_title)

    for i in range(len(image_list)):
        normaliza(image_list[i], image_title_list[i])
        image_list[i] = image_list[i].astype(np.uint8)

    for i in range(rows * columns):
        if i < len(image_list):
            plt.subplot(rows, columns, i+1) # El índice (3er parametro) comienza en 1 en la esquina superior izquierda y aumenta a la derecha.
            if flag_color == 0:
                plt.imshow(image_list[i], cmap = "gray")
            else:
                plt.imshow(image_list[i])
            plt.title(image_title_list[i])
            plt.xticks([])  # Se le pasa una lista de posiciones en las que se deben colocar los
            plt.yticks([])  # ticks, si pasamos una lista vacía deshabilitamos los xticks e yticks
    plt.show()

    for i in range(len(image_list)):
        image_list[i] = image_list[i].astype(np.float64)    # lo devolvemos a su formato.

""" Visualiza varias imágenes a la vez.
- image_list: Secuencia de imágenes.
- flag_color (op): bandera para indicar si la imagen es en B/N o color. Por defecto color.
- image_title (op): título de la imagen. Por defecto 'Imágenes'
- window_title (op): título de la ventana. Por defecto 'Ejercicio pirámide'
"""
def muestraMI(image_list, flag_color = 1, image_title = "Imágenes", window_title = "Ejercicio pirámide"):
  altura = max(im.shape[0] for im in image_list)

  for i,im in enumerate(image_list):
    if im.shape[0] < altura: # Redimensionar imágenes
      borde = int((altura - image_list[i].shape[0])/2)
      image_list[i] = cv2.copyMakeBorder(image_list[i], borde, borde + (altura - image_list[i].shape[0]) % 2, 0, 0, cv2.BORDER_CONSTANT, value = (0,0,0))

  im_concat = cv2.hconcat(image_list)
  pintaI(im_concat, flag_color, image_title, "Ejercicio pirámide")

""" Aplica una máscara Gaussiana 2D. Devuelve la imagen con las máscara aplicada.
- image: la imagen a tratar.
- kernel_x: kernel en las dirección X.
- kernel_y: kernel en las dirección Y.
- border_type (op): tipo de bordes. BORDER_DEFAULT.
"""
def convolution(image, kernel_x, kernel_y, border_type = cv2.BORDER_DEFAULT):
    kernel_x = np.transpose(kernel_x)
    kernel_x = cv2.flip(kernel_x, 0)
    kernel_y = cv2.flip(kernel_y, 1)
    im_conv = cv2.filter2D(image, -1, kernel_x, borderType = border_type)
    im_conv = cv2.filter2D(im_conv, -1, kernel_y, borderType = border_type)
    return im_conv

""" Aplica una máscara Gaussiana 2D. Devuelve la imagen con las máscara aplicada.
- image: la imagen a tratar.
- sigma_x: sigma en la dirección X.
- sigma_y (op): sigma en la dirección Y. Por defecto sigma_y = sigma_x
- k_size_x (op): tamaño del kernel en dirección X (positivo e impar). Por defecto es 0, se obtiene a través de sigma.
- k_size_y (op): tamaño del kernel en dirección Y (positivo e impar). Por defecto es 0, se obtiene a través de sigma.
- border_type (op): tipo de bordes. BORDER_DEFAULT.
"""
def gaussian_blur(image, sigma_x, sigma_y = 0, k_size_x = 0, k_size_y = 0, border_type = cv2.BORDER_DEFAULT):
    if sigma_y == 0:
        sigma_y = sigma_x
    if k_size_x == 0:
        k_size_x = int(6*sigma_x + 1)
    if k_size_y == 0:
        k_size_y = int(6*sigma_y + 1)

    kernel_x = cv2.getGaussianKernel(k_size_x, sigma_x)
    kernel_y = cv2.getGaussianKernel(k_size_y, sigma_y)
    return convolution(image, kernel_x, kernel_y, border_type)

""" Hace un subsampling de la imagen pasada como argumento. Devuelve la imagen recortada.
- image: imagen a recortar.
"""
def subsampling(image):
    n_fil = int(image.shape[0]/2)
    n_col = int(image.shape[1]/2)
    cp = np.copy(image)

    for a in range(0, n_fil):
        cp = np.delete(cp, a, axis = 0)
    for a in range(0, n_col):
        cp = np.delete(cp, a, axis = 1)

    return cp

""" Hace un upsampling de la imagen pasada como argumento. Devuelve la imagen agrandada.
- image: imagen a agrandar.
- n_fil: número de filas de la matriz resultante.
- n_col: número de columnas de la matriz resultante.
"""

def upsampling(image, n_fil, n_col):
    fil = False
    col = False

    if n_fil % 2 == 1:
        n_fil = n_fil-1
        fil = True

    if n_col % 2 == 1:
        n_col = n_col-1
        col = True

    if len(image.shape)==2:
        if fil and col:
            salida = np.zeros((n_fil+1, n_col+1))
        elif fil:
            salida = np.zeros((n_fil+1, n_col))
        elif col:
            salida = np.zeros((n_fil, n_col+1))
        else:
            salida = np.zeros((n_fil, n_col))

        # Relleno la matriz, en cada iteración escribo 4 elementos de la matriz de salida
        for i in range(0, n_fil, 2):
            for j in range(0, n_col, 2):
                salida[i][j] = image[int(i/2)][int(j/2)]
                salida[i+1][j] = image[int(i/2)][int(j/2)]
                salida[i][j+1] = image[int(i/2)][int(j/2)]
                salida[i+1][j+1] = image[int(i/2)][int(j/2)]

        # Si el número de filas era impar escribo la última fila la cual borré con n_fil = n_fil-1
        if fil:
            for j in range(0, n_col, 2):
                salida[n_fil][j] = image[image.shape[0]-1][int(j/2)]
                salida[n_fil][j+1] = image[image.shape[0]-1][int(j/2)]

        # Si el número de columnas era impar escribo la última columna la cual borré con n_col = n_col-1
        if col:
            for i in range(0, n_fil, 2):
                salida[i][n_col] = image[int(i/2)][image.shape[1]-1]
                salida[i+1][n_col] = image[int(i/2)][image.shape[1]-1]

            # Si se da el caso de que n_fil y n_col eran impares falta el último elemento por escribir en cada banda
            if fil and col:
                salida[n_fil][n_col] = image[image.shape[0]-1][image.shape[1]-1]

    if len(image.shape)==3:
        if fil and col:
            salida = np.zeros((n_fil+1, n_col+1, image.shape[2]))
        elif fil:
            salida = np.zeros((n_fil+1, n_col, image.shape[2]))
        elif col:
            salida = np.zeros((n_fil, n_col+1, image.shape[2]))
        else:
            salida = np.zeros((n_fil, n_col, image.shape[2]))

        # Escribo en todos los canales
        for k in range(0, image.shape[2]):
            # Relleno la matriz, en cada iteración escribo 4 elementos de la matriz de salida
            for i in range(0, n_fil, 2):
                for j in range(0, n_col, 2):
                    salida[i][j][k] = image[int(i/2)][int(j/2)][k]
                    salida[i+1][j][k] = image[int(i/2)][int(j/2)][k]
                    salida[i][j+1][k] = image[int(i/2)][int(j/2)][k]
                    salida[i+1][j+1][k] = image[int(i/2)][int(j/2)][k]

            # Si el número de filas era impar escribo la última fila la cual borré con n_fil = n_fil-1
            if fil:
                for k in range(0, image.shape[2]):
                    #salida[n_fil,:,k][::2] = image[image.shape[0]-1,:,k]
                    #salida[n_fil,:,k][1::2] = image[image.shape[0]-1,:,k]
                    for j in range(0, n_col, 2):
                        salida[n_fil][j][k] = image[image.shape[0]-1][int(j/2)][k]
                        salida[n_fil][j+1][k] = image[image.shape[0]-1][int(j/2)][k]

            # Si el número de columnas era impar escribo la última columna la cual borré con n_col = n_col-1
            if col:
                for k in range(0, image.shape[2]):
                    #salida[:,n_col,k][::2] = image[:,image.shape[1]-1,k]
                    #salida[:,n_col,k][1::2] = image[:,image.shape[1]-1,k]
                    for i in range(0, n_fil, 2):
                        salida[i][n_col,k] = image[int(i/2)][image.shape[1]-1][k]
                        salida[i+1][n_col][k] = image[int(i/2)][image.shape[1]-1][k]

                    # Si se da el caso de que n_fil y n_col eran impares falta el último elemento por escribir en cada banda
                    if fil and col:
                        salida[n_fil][n_col][k] = image[image.shape[0]-1][image.shape[1]-1][k]

    return salida

""" Genera representación de pirámide gaussiana. Devuelve la lista de imágenes que forman la pirámide gaussiana.
- image: La imagen a la que generar la pirámide gaussiana.
- levels (op): Número de niveles de la pirámide gaussiana. Por defecto 4.
- border_type (op): Tipo de borde a utilizar. Por defecto BORDER DEFAULT.
"""
def gaussian_pyramid(image, levels = 4, border_type = cv2.BORDER_DEFAULT):
    pyramid = [image]
    blur = np.copy(image)
    for n in range(levels):
        blur = gaussian_blur(blur, 1, 1, 7, 7, border_type = border_type)
        blur = subsampling(blur)
        pyramid.append(blur)
    return pyramid

""" Genera representación de pirámide laplaciana. Devuelve la lista de imágenes que forman la pirámide laplaciana.
- image: La imagen a la que generar la pirámide laplaciana.
- levels (op): Número de niveles de la pirámide laplaciana. Por defecto 4.
- border_type (op): Tipo de borde a utilizar. BORDER DEFAULT.
"""
def laplacian_pyramid(image, levels = 4, border_type = cv2.BORDER_DEFAULT):
    gau_pyr = gaussian_pyramid(image, levels+1, border_type)
    lap_pyr = []
    for n in range(levels):
        gau_n_1 = upsampling(gau_pyr[n+1], gau_pyr[n].shape[0], gau_pyr[n].shape[1])
        #gau_n_1 = 4*gaussian_blur(gau_n_1, 1, 1, 7, 7)   # Otra opción para la laplaciana: poniendo 0s.
        gau_n_1 = gaussian_blur(gau_n_1, 1, 1, 7, 7, border_type = border_type)
        lap_pyr.append(normaliza(gau_pyr[n] - gau_n_1, "Etapa {} de la pirámide gaussiana.".format(n)))
    return lap_pyr

""" Supresión de de no máximos.
- image: imagen a tratar
"""
def non_maximum_supression(image, winSize):
    res = np.zeros(image.shape)

    if len(image.shape) == 2:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                max = 0
                for p in range(-int(winSize/2), int(winSize/2)+1):
                    for q in range(-int(winSize/2), int(winSize/2)+1):
                        if i+p>=0 and j+q>=0 and (i+p)<image.shape[0] and (j+q)<image.shape[1]:
                            if max<image[i+p][j+q]:
                                max = image[i+p][j+q]

                if max<=image[i][j]:
                    res[i][j] = image[i][j]
                else:
                    res[i][j] = 0

    return res

""" Función que realiza la correlación de la máscara 'kernel' sobre la imagen 'image'. Devuelve la imagen correlada.
- image: imagen a tratar.
- kernel: kernel a pasar por la imagen.
"""
def correlation1D(image, kernel):
    mitad = int(len(kernel)/2)
    salida = np.zeros(image.shape)

    if len(image.shape) == 2:
        im = np.zeros((image.shape[0], image.shape[1] + 2*mitad))
        im[:, mitad:im.shape[1]-mitad] = image

        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                for n in range(-mitad, mitad+1):
                    salida[i][j] += im[i][j+mitad+n] * kernel[n+mitad]

    elif len(image.shape) == 3:
        im = np.zeros((image.shape[0], image.shape[1] + 2*mitad, image.shape[2]))
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                for k in range(0, image.shape[2]):
                    im[i][j+mitad][k] = image[i][j][k]

        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                for k in range(0, image.shape[2]):
                    for n in range(-mitad, mitad+1):
                        salida[i][j][k] += im[i][j+mitad+n][k] * kernel[n+mitad]

    return salida


""" Función que realiza una convolución 2D con máscaras separables. Devuelve la imagen convolucionada.
- image: imagen a tratar.
- kernel_x: kernel en las dirección X.
- kernel_y: kernel en las dirección Y.
"""
def convolution2D(image, kernel_x, kernel_y):
    salida = np.copy(image)
    kernel_x = cv2.flip(kernel_x, -1)
    kernel_y = cv2.flip(kernel_y, -1)

    salida = correlation1D(salida, kernel_x)
    salida = np.transpose(salida)
    salida = correlation1D(salida, kernel_y)
    salida = np.transpose(salida)
    salida = normaliza(salida, "Convolución 2D")

    return salida

#######################
###   EJERCICIO 1   ###
#######################

def criterioHarris(eigenVal1, eigenVal2, threshold):
    fp = np.zeros(eigenVal1.shape)

    for i in range(eigenVal1.shape[0]):
        for j in range (eigenVal1.shape[1]):
            if eigenVal1[i][j] == 0 and eigenVal2[i][j] == 0:
                fp[i][j] = 0
            else:
                fp[i][j] = eigenVal1[i][j] * eigenVal2[i][j] / (eigenVal1[i][j]+eigenVal2[i][j])
                if fp[i][j] < threshold:
                    fp[i][j] = 0
    return fp

def orientacion(u):
    ksize = 3
    kx = cv2.getDerivKernels(1, 0, ksize)
    ky = cv2.getDerivKernels(0, 1, ksize)

    u = u / sqrt(u[0]*u[0]+u[1]*u[1])
    if u[1] != 0:
        theta = math.atan(u[0]/u[1])
        if u[0]>0 and u[1]<0:
            theta = math.pi - theta
        elif u[0]<0 and u[1]<0:
            theta = math.pi + theta
        elif u[0]<0 and u[1]>0:
            theta = 2*math.pi - theta
    else:
        if u[0]>0:
            theta = math.pi/2
        elif u[0]<0:
            theta = 3/2 * math.pi

    return theta * 180 / math.pi

def get_keypoints(matrix, block_size, level):
    kp = []

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j]>0:
                kp.append( cv2.KeyPoint(j*(2**level), i*(2**level), block_size*(level+1), 1) )

    return kp;

def getHarris(img, block_size, ksize, threshold, level, winSize = 5):
    # Se calculan los autovectoresy autovalores:
    vals_vecs = cv2.cornerEigenValsAndVecs(img, block_size, ksize)

    # En cada píxel tenemos (l1, l2, x1, y1, x2, y2)
    vals_vecs = cv2.split(vals_vecs)
    eigenVal1 = vals_vecs[0]
    eigenVal2 = vals_vecs[1]
    x1 = vals_vecs[2]
    y1 = vals_vecs[3]
    x2 = vals_vecs[4]
    y2 = vals_vecs[5]
    #(eigenVal1, eigenVal2, x1, y1, x2, y2) = vals_vecs

    # Criterio de Harris para obtener la matriz con el valor asociado a cada pixel
    harris = criterioHarris(eigenVal1, eigenVal2, threshold)
    # Se suprimen los valores no máximos
    harris = non_maximum_supression(harris, winSize)
    # Obtenemos los keypoints
    return get_keypoints(harris, block_size, level)

""" Refina la posición de los keypoints sobre una imagen.
- img:
"""
def refineHarris(img):

    return true

""" Ejecución de ejemplos del ejercicio 1.
- image: Imagen a estudiar y de la que sacar los puntos Harris.
"""
def ejercicio_1(img):
    print("--- EJERCICIO 1 - PUNTOS HARRIS ---")
    levels = 4
    num_kp = 0
    pyr = gaussian_pyramid(img, levels)
    keypoints = []
    img_all_harris = np.copy(img)

    for l in range(levels):
        keypoints.append( getHarris(pyr[l], 3, 3, 0.001, l) )
        # Contabilizamos el número de keypoints
        print("{} keypoints en el nivel {}".format(len(keypoints[l]), l))
        num_kp += len(keypoints[l])
        # Los ponemos en la imagen
        img_harris = cv2.drawKeypoints(img, keypoints[l], np.array([]), color = (250,0,0), flags=4)
        img_all_harris = cv2.drawKeypoints(img_all_harris, keypoints[l], np.array([]), color = (255,0,0), flags=4)
        pintaI(img_harris, 0, "Puntos Harris", "Ejercicio 1A")
    pintaI(img_all_harris, 0, "Puntos Harris de todos los niveles", "Ejercicio 1A")
    print("El número de keypoints total es {}".format(num_kp))

    #APARTADO D
    #refineHarris(img)
    #img_refinada = cv2.circle(img, center=(y, x), radius=0.1, color=(0,250,0), thickness=1)
    #pintaI(img_refinada)
    input("Pulsa 'Enter' para continuar\n")

#######################
###   EJERCICIO 2   ###
#######################

"""
Dadas dos imágenes calcula los keypoints y descriptores para obtener los matches
usando "BruteForce+crossCheck". Devuelve la imagen compuesta.
- n: número de matches a mostrar
- flag: indica si se muestran los keypoints y los matches (0) o solo los matches (2).
"""
def getMatches_BF_CC(img1, img2, n = 100, flag = 2):
    # Inicializamos el descriptor AKAZE
    detector = cv2.AKAZE_create()
    # Se obtienen los keypoints y los descriptores de las dos imágenes
    keypoints1, descriptor1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptor2 = detector.detectAndCompute(img2, None)

    # Se crea el objeto BFMatcher activando la validación cruzada
    bf = cv2.BFMatcher(crossCheck = True)
    # Se consiguen los puntos con los que hace match
    matches1to2 = bf.match(descriptor1, descriptor2)
    # Se ordenan los matches dependiendo de la distancia entre ambos
    matches1to2 = sorted(matches1to2, key = lambda x:x.distance)[0:n]
    # Se guardan n puntos aleatorios
    #matches1to2 = random.sample(matches1to2, n)

    # Imagen con los matches
    img_match = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, None, flags = flag)
    return img_match

"""
Dadas dos imágenes calcula los keypoints y descriptores para obtener los matches
usando "Lowe-Average-2NN". Devuelve la imagen compuesta.
Si se indica el flag "improve" como True, elegirá los mejores matches.
- n: número de matches a mostrar
- flag: indica si se muestran los keypoints y los matches (0) o solo los matches (2).
"""
def getMatches_LA_2NN(img1, img2, ratio = 0.8, n = 1, flag = 2):
    # Inicializamos el descriptor AKAZE
    detector = cv2.AKAZE_create()
    # Se obtienen los keypoints y los descriptores de las dos imágenes
    keypoints1, descriptor1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptor2 = detector.detectAndCompute(img2, None)

    # Se crea el objeto BFMatcher
    bf = cv2.BFMatcher()
    # Escogemos los puntos con los que hace match indicando los vecinos más cercanos para la comprobación (2)
    matches1to2 = bf.knnMatch(descriptor1, descriptor2, 2)

    # Se mostrará el número máximo de matches
    n = int(len(matches1to2)*n)
    # Mejora de los matches -> los puntos que cumplan con un radio en concreto
    best1to2 = []
    # Se recorren todos los matches
    for p1, p2 in matches1to2:
        if p1.distance < ratio*p2.distance:
            best1to2.append([p1])

    # Se ordenan los matches dependiendo de la distancia entre ambos
    matches1to2 = sorted(best1to2, key = lambda x:x.distance)[0:n]
    # Se guardan n puntos aleatorios
    #matches1to2 = random.sample(good, n)

    # Imagen con los matches
    img_match = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches1to2, None, flags = flag)
    return img_match

""" Ejecución de ejemplos del ejercicio 2.
- image:
"""
def ejercicio_2(img1, img2):
    print("--- EJERCICIO 2 - DESCRIPTORES AKAZE CON BFMatcher Y CRITERIOS BruteForce+crossCheck y Lowe-Average-2NN ---")
    match_BF_CC = getMatches_BF_CC(img1, img2)
    pintaI(match_BF_CC)
    match_LA_2NN = getMatches_LA_2NN(img1, img2)
    input("Pulsa 'Enter' para continuar\n")


#######################
###   EJERCICIO 3   ###
#######################



""" Ejecución de ejemplos del ejercicio 3.
- image:
"""
def ejercicio_3():
    print("--- EJERCICIO 3 - AKAZE ---")

    input("Pulsa 'Enter' para continuar\n")

#######################
###   EJERCICIO 4   ###
#######################



""" Ejecución de ejemplos del ejercicio 4.
- image:
"""
def ejercicio_4():
    print("--- EJERCICIO 4 - TIT ---")

    input("Pulsa 'Enter' para continuar\n")

#######################
###     BONUS 1     ###
#######################



""" Ejecución de ejemplos del bonus 1.
- image:
"""
def bonus_1():
    print("--- BONUS 1 - TIT ---")

    input("Pulsa 'Enter' para continuar\n")

#######################
###       MAIN      ###
#######################

def main():
    gray1 = leer_imagen("imagenes/yosemite1.jpg",0)
    gray2 = leer_imagen("imagenes/yosemite2.jpg",0)

    ejercicio_1(gray1)
    ejercicio_2(gray1, gray2)
    #ejercicio_3()
    #ejercicio_4()
    #bonus_1()

if __name__ == "__main__":
	main()