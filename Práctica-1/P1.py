#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Carlos Sánchez Muñoz
@date: 1 de octubre de 2019
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

""" Uso la notación Snake Case la cual es habitual en Python """

################################################
###   Práctica 0 (Sólo lo que me interesa)   ###
################################################

""" Lee una imagen ya sea en grises o en color
- file_name: archivo de la imagen
- flag_color: modo en el que se va a leer la imagen -> grises o color
"""
def leer_imagen(file_name, flag_color = 1):
    if flag_color == 0:
        print('Leyendo ' + file_name + ' en gris')
    elif flag_color==1:
        print('Leyendo ' + file_name + ' en color')
    else:
        print('flag_color debe ser 0 o 1')

    img = cv2.imread(file_name, flag_color)
    return img

""" Normaliza una matriz
- image: matriz a normalizar
"""
def normaliza(image):
    # En caso de que los máximos sean 255 o las mínimos 0 no iteramos en los  bucles
    if len(image.shape) == 2:
        max = np.amax(image)
        min = np.amin(image)
        if max!=255 and min!=0:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    image[i][j] = (image[i][j]-min)/(max-min) * 255
    elif len(image.shape) == 3:
        max = np.amax(image, (0,1))
        min = np.amin(image, (0,1))
        if max[0]!=255 and max[1]!=255 and max[2]!=255 and min[0]!=0 and min[1]!=0 and min[2]!=0:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    for k in range(image.shape[2]):
                        image[i][j][k] = (image[i][j][k]-min[k])/(max[k]-min[k]) * 255

""" Imprime una imagen a través de una matriz
- image_title: título de la imagen
- image: imagen a imprimir
"""
def pintaI(image_title, image):
    normaliza(image)        # normalizamos la matriz
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(0).canvas.set_window_title("Ejercicio")  # Ponemos nombre a la ventana
    plt.imshow(imgrgb)
    plt.title(image_title)  # Ponemos nombre a la imagen
    plt.show()

""" Lee una lista de imágenes ya sea en grises o en color
- image_list: lista de imágenes a concatenar
- flag_color: modo en el que se van a leer las imágenes
"""
def leer_lista_imagenes(file_name_list, flag_color = 1):
    image_list = []

    for i in file_name_list:
        img = leer_imagen(i, flag_color)
        image_list.append(img)

    return image_list

""" Las imágenes tienen que ser del mismo tipo ya que si no nos da el error:
        ValueError: all the input arrays must have same number of dimensions
    No es lo mismo la matriz de una imagen a color que necesita 3 dimensiones que una en grises
- image_list: lista de imágenes a concatenar
- horizontal: modo de concatenación. 0-vertical y en otro caso horizontal
"""
def pintaMI(image_list, horizontal=1):
    if horizontal != 0: # Salvo que sea 0-vertical, ponemos por defecto 1-horizontal.
        horizontal = 1

    concatenated_img = cv2.resize(image_list[0], (300,300), interpolation = cv2.INTER_AREA)

    for i in np.arange(1,len(image_list)):
        aux = cv2.resize(image_list[i], (300,300), interpolation = cv2.INTER_AREA)
        concatenated_img = np.concatenate((concatenated_img, aux), axis=horizontal)

    pintaI("Imágenes concatenadas", concatenated_img)

""" Escala las imágenes al mismo tamaño, al de la más pequeña
- images: La lista de imágenes
"""
def escalar_imagenes(image_list):
    minRows = 999999999999
    minCols = 999999999999

    # Se obtiene el minimo ancho y alto
    for i in range(len(image_list)):
        img = image_list[i]
        if(len(img) < minRows):
            minRows = len(img)
        if(len(img[0]) < minCols):
            minCols = len(img[0])

    # Ajustamos las imágenes a la más pequeña
    for i in range(len(image_list)):
        img = image_list[i]
        image_list[i] = cv2.resize(img, (minRows, minCols))

    return image_list

""" Muestra múltiples imágenes en una ventena Matplotlib
- image_list: La lista de imágenes
- image_title_list: Lista de títulos de las imágenes
- rows: filas
- columns: columnas
"""
def imprimir_imagenes_titulos(image_list, image_title_list, rows, columns, window_title = 'Imágenes con títulos'):
    # Se igualan los tamaños de las Imagenes
    image_list = escalar_imagenes(image_list)
    fig = plt.figure(0)
    fig.canvas.set_window_title(window_title)

    for i in range(rows * columns):
        if i < len(image_list):
            plt.subplot(rows, columns, i+1) # El índice (3er parametro) comienza en 1 en la esquina superior izquierda y aumenta a la derecha.
            imgrgb = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2RGB)
            plt.imshow(imgrgb)
            plt.title(image_title_list[i])
            plt.xticks([])  # Se le pasa una lista de posiciones en las que se deben colocar los
            plt.yticks([])  # ticks, si pasamos una lista vacía deshabilitamos los xticks e yticks
    plt.show()

######################
###   Práctica 1   ###
######################

# EJERCICIO 1 #

"""Aplica una máscara Gaussiana 2D. Devuelve la imagen con las máscara aplicada.
- image: la imagen
- sigma_x: sigma en la dirección X
- sigma_y: sigma en la dirección X
- size: tamaño del kernel
- border_type: tipo de bordes

La clave es la función GaussianBlur de cv2:
dst = cv.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType=BORDER_DEFAULT]]] )
"""
def gaussian_blur(image, sigma_x, sigma_y = 0, size = (0, 0), border_type = cv2.BORDER_DEFAULT):
    return cv2.GaussianBlur(image, size, sigma_x, sigmaY = sigma_y, borderType = border_type)

"""Obtiene máscaras 1D de máscaras derivadas. Devuelve los vectores de derivada
Argumentos posicionales:
- dx: orden de derivación respecto de x
- dy: orden de derivación respecto de y
- size: tamaño del kernel
"""
def derive_convolution(dx, dy, size):
  return cv2.getDerivKernels(dx, dy, size)

"""Aplica máscara laplaciana a imagen. Devuelve la imagen con la máscara aplicada
- im: Imagen a la que aplicar la máscara
- k_size: Tamaño de la máscara
- size: tamaño del kernel
- border_type: Tipo de borde
"""
def laplacian_gaussian(image, sigma = 0, k_size = 7, size = (0, 0), border_type = cv2.BORDER_DEFAULT):
  # Reducimos ruido con alisado gaussiano
  blur = gaussian_blur(image, sigma, size = size, border_type = border_type)
  return cv2.Laplacian(blur, -1, ksize = k_size, borderType = border_type, delta = 50)

"""Ejecución de ejemplos del ejercicio 1 con diferentes σ."""
def ejercicio_1(image):
    print("--- EJERCICIO 1A - GAUSSIANA 2D Y MÁSCARAS 1D (getDerivKernels) ---")
    imprimir_imagenes_titulos([image, gaussian_blur(image, 2), gaussian_blur(image, 6)],
                              ['Original', 'σ_x = 2', 'σ_x = 6'], 1, 3, 'Gaussian')
    imprimir_imagenes_titulos([image, gaussian_blur(image, 2, sigma_y=3, size=(5,5)), gaussian_blur(image, 6, sigma_y=4, size=(7,7))],
                              ['Original', 'σ_x = 2, σ_y = 3', 'σ_x = 6, σ_y = 4'], 1, 3, 'Gaussian')

    # Derivadas y tamaños de kernel a probar
    ders = [(0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
    tams = [3, 5]

    for tam in tams:
        print("tam = {}".format(tam))
        for dx, dy in ders:
            print("  dx = {}, dy = {}".format(dx, dy), end = ": ")
            print("{}, {}".format(*map(np.transpose, derive_convolution(dx, dy, tam) )))  # Imprimo vectores fila
    input("Pulsa 'Enter' para continuar\n")

    print("--- EJERCICIO 1B -  LAPLACIANA DE GAUSSIANA ---")
    imprimir_imagenes_titulos([image, laplacian_gaussian(image, 1, border_type = cv2.BORDER_REPLICATE), laplacian_gaussian(image, 1, border_type = cv2.BORDER_CONSTANT)],
                       ['Original', '1, REPLICATE', '1, REFLECT'], 1, 3, 'Laplacian of gaussian')
    imprimir_imagenes_titulos([image, laplacian_gaussian(image, 3, border_type = cv2.BORDER_REPLICATE), laplacian_gaussian(image, 3, border_type = cv2.BORDER_REFLECT)],
                       ['Original', '3, REPLICATE', '3, REFLECT'], 1, 3, 'Laplacian of gaussian')
    input("Pulsa 'Enter' para continuar\n")

# EJERCICIO 2 #

"""Genera representación de pirámide gaussiana
- image: La imagen a la que generar la pirámide gaussiana
- border_type: Tipo de borde a utilizar
- levels: Número de niveles de la pirámide gaussiana (4 por defecto)
Devuelve: Lista de imágenes que forman la pirámide gaussiana"""
def gaussian_pyramid(image, levels = 4, border_type = cv2.BORDER_DEFAULT):
    pyramid = [image]
    for n in range(levels):
        image = gaussian_blur(image, 0, size = (7, 7), border_type = border_type)
        image = cv2.resize(image, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
        pyramid.append(image)
    return pyramid

"""Genera representación de pirámide laplaciana. Devuelve la lista de imágenes que forman la pirámide laplaciana
- image: La imagen a la que generar la pirámide laplaciana
- border_type: Tipo de borde a utilizar
- levels: Número de niveles de la pirámide laplaciana (4 por defecto)
"""
def laplacian_pyramid(image, levels = 4, border_type = cv2.BORDER_DEFAULT):
    gau_pyr = gaussian_pyramid(image, levels+1, border_type)
    lap_pyr   = []
    for n in range(levels):
        gau_n_1 = cv2.resize(gau_pyr[n+1], (gau_pyr[n].shape[1], gau_pyr[n].shape[0]), interpolation = cv2.INTER_CUBIC)
        lap_pyr.append(cv2.subtract(gau_pyr[n], gau_n_1) + 40) # Resta al nivel n el nivel n+1 y suma una constante para visualizarlo
    return lap_pyr

"""Ejecución de ejemplos del ejercicio 2."""
def ejercicio_2(image):
    print("--- EJERCICIO 2A - GAUSSIAN PYRAMID ---")
    gau_pyr = gaussian_pyramid(image, 4)
    imprimir_imagenes_titulos(gau_pyr, ['1', '2', '3', '4'], 1, 4, 'Gaussian pyramid')
    input("Pulsa 'Enter' para continuar\n")

    print("--- EJERCICIO 2B - LAPLACIAN PYRAMID ---")
    lap_pyr = laplacian_pyramid(image, 4)
    imprimir_imagenes_titulos(lap_pyr, ['1', '2', '3', '4'], 1, 4, 'Laplacian pyramid')
    input("Pulsa 'Enter' para continuar\n")

    #print("--- EJERCICIO 2C - ESPACIO DE ESCALAS LAPLACIANO ---")

    #input("Pulsa 'Enter' para continuar\n")

# EJERCICIO 3 #



"""Ejecución de ejemplos del ejercicio 3."""
def ejercicio_3(image):
    print("--- EJERCICIO 3 - NOMBRE ---")

    input("Pulsa 'Enter' para continuar\n")

#################
###   BONUS   ###
#################



################
###   MAIN   ###
################

def main():
    im_color = leer_imagen('data/plane.bmp', 1)   # Leemos la imagen en color
    #ejercicio_1(im_color)
    ejercicio_2(im_color)
    #ejercicio_3(im_color)

if __name__ == "__main__":
	main()
