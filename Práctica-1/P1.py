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
            print('Normalizando imagen')
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    image[i][j] = (image[i][j]-min)/(max-min) * 255
    elif len(image.shape) == 3:
        max = np.amax(image, (0,1))
        min = np.amin(image, (0,1))
        if max[0]!=255 and max[1]!=255 and max[2]!=255 and min[0]!=0 and min[1]!=0 and min[2]!=0:
            print('Normalizando imagen')
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

    #cv2.imshow(image_title, image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

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
- ksize: tamaño del kernel (2D, positivos e impares). Si es (0,0) se obtiene a través de sigma
- border_type: tipo de bordes

La clave es la función GaussianBlur de cv2:
dst = cv.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType=BORDER_DEFAULT]]] )
"""
def gaussian_blur(image, sigma_x, sigma_y = 0, ksize = (0,0), border_type = cv2.BORDER_DEFAULT):
    return cv2.GaussianBlur(image, ksize, sigma_x, sigmaY = sigma_y, borderType = border_type)

"""Obtiene máscaras 1D de máscaras derivadas. Devuelve los vectores de derivada
Argumentos posicionales:
- dx: orden de derivación respecto de x
- dy: orden de derivación respecto de y
- ksize: tamaño del kernel, puede ser 1, 3, 5, 7
"""
def derive_convolution(image, dx, dy, ksize):
    print('Máscara de derivada con orden ({}, {}) y tamaño del kernel {}'.format(dx, dy, ksize))
    kx, ky = cv2.getDerivKernels(dx,dy,ksize)
    image = cv2.sepFilter2D(image, -1, kx, ky)
    return image

"""Aplica máscara laplaciana a imagen. Devuelve la imagen con la máscara aplicada
- im: Imagen a la que aplicar la máscara
- k_size: Tamaño de la máscara
- size: tamaño del kernel
- border_type: Tipo de borde
"""
def laplacian_gaussian(image, sigma = 0, k_size = 7, size = (0, 0), border_type = cv2.BORDER_DEFAULT):
  # Reducimos ruido con alisado gaussiano
  blur = gaussian_blur(image, sigma, ksize = size, border_type = border_type)
  return cv2.Laplacian(blur, -1, ksize = k_size, borderType = border_type, delta = 50)

"""Ejecución de ejemplos del ejercicio 1 con diferentes σ."""
def ejercicio_1(image):
    print("--- EJERCICIO 1A - GAUSSIANA 2D Y MÁSCARAS 1D (getDerivKernels) ---")
    imprimir_imagenes_titulos([image, gaussian_blur(image, 2), gaussian_blur(image, 6)],
                              ['Original', 'σ_x = 2', 'σ_x = 6'], 1, 3, 'Gaussian')
    imprimir_imagenes_titulos([image, gaussian_blur(image, 2, sigma_y=3), gaussian_blur(image, 6, sigma_y=4)],
                              ['Original', 'σ_x = 2, σ_y = 3', 'σ_x = 6, σ_y = 4'], 1, 3, 'Gaussian')
    imprimir_imagenes_titulos([gaussian_blur(image, 1, sigma_y=1, border_type=cv2.BORDER_DEFAULT), gaussian_blur(image, 1, sigma_y=1, border_type=cv2.BORDER_REPLICATE),
                               gaussian_blur(image, 1, sigma_y=1, border_type=cv2.BORDER_REFLECT), gaussian_blur(image, 1, sigma_y=1, border_type=cv2.BORDER_CONSTANT)],
                              ['BORDER_DEFAULT', 'BORDER_REPLICATE', 'BORDER_REFLECT', 'BORDER_CONSTANT'], 2, 2, 'Gaussian with borders')

    # Máscaras de derivadas 1D
    tams = [3, 5]
    for tam in tams:
        imprimir_imagenes_titulos([derive_convolution(image, 1, 0, tam), derive_convolution(image, 0, 1, tam), derive_convolution(image, 1, 1, tam), derive_convolution(image, 2, 0, tam),
                                   derive_convolution(image, 0, 2, tam), derive_convolution(image, 2, 1, tam), derive_convolution(image, 1, 2, tam), derive_convolution(image, 2, 2, tam)],
                                  ['(1, 0)', '(0, 1)', '(1, 1)', '(2, 0)', '(0, 2)', '(2, 1)', '(1, 2)', '(2, 2)'], 4, 4, 'Máscaras de derivadas 1D')
    input("Pulsa 'Enter' para continuar\n")

    print("--- EJERCICIO 1B -  LAPLACIANA DE GAUSSIANA ---")
    imprimir_imagenes_titulos([image, laplacian_gaussian(image, 1, border_type = cv2.BORDER_REPLICATE), laplacian_gaussian(image, 1, border_type = cv2.BORDER_CONSTANT)],
                       ['Original', 'σ = 1, REPLICATE', 'σ = 1, REFLECT'], 1, 3, 'Laplacian of gaussian')
    imprimir_imagenes_titulos([image, laplacian_gaussian(image, 3, border_type = cv2.BORDER_REPLICATE), laplacian_gaussian(image, 3, border_type = cv2.BORDER_REFLECT)],
                       ['Original', 'σ = 3, REPLICATE', 'σ = 3, REFLECT'], 1, 3, 'Laplacian of gaussian')

    input("Pulsa 'Enter' para continuar\n")

# EJERCICIO 2 #
"""Visualiza varias imágenes a la vez
- image_list: Secuencia de imágenes"""
def muestraMI(image_list, image_title = "Imágenes"):
  altura = max(im.shape[0] for im in image_list)

  for i,im in enumerate(image_list):
    if im.shape[0] < altura: # Redimensionar imágenes
      borde = int((altura - image_list[i].shape[0])/2)
      image_list[i] = cv2.copyMakeBorder(
        image_list[i], borde, borde + (altura - image_list[i].shape[0]) % 2,
        0, 0, cv2.BORDER_CONSTANT, value = (0,0,0))

  im_concat = cv2.hconcat(image_list)
  pintaI(image_title, im_concat)

"""Hace un subsampling de la imagen pasada como argumento. Devuelve la imagen recortada.
- image: imagen a recortar"""
def subsampling(image):
    n_fil = int(image.shape[0]/2)
    n_col = int(image.shape[1]/2)
    cp = np.copy(image)

    for a in range(0,n_fil):
        cp = np.delete(cp,a,axis = 0)
    for a in range(0,n_col):
        cp = np.delete(cp,a,axis = 1)

    return cp

"""Genera representación de pirámide gaussiana. Devuelve la lista de imágenes que forman la pirámide gaussiana
- image: La imagen a la que generar la pirámide gaussiana
- border_type: Tipo de borde a utilizar
- levels: Número de niveles de la pirámide gaussiana (4 por defecto)"""
def gaussian_pyramid(image, levels = 4, border_type = cv2.BORDER_DEFAULT):
    pyramid = [image]
    for n in range(levels):
        image = gaussian_blur(image, 1, 1, ksize = (3, 3), border_type = border_type)
        image = subsampling(image)
        pyramid.append(image)
    return pyramid

"""Genera representación de pirámide laplaciana. Devuelve la lista de imágenes que forman la pirámide laplaciana
- image: La imagen a la que generar la pirámide laplaciana
- border_type: Tipo de borde a utilizar
- levels: Número de niveles de la pirámide laplaciana (4 por defecto)
"""
def laplacian_pyramid(image, levels = 4, border_type = cv2.BORDER_DEFAULT):
    gau_pyr = gaussian_pyramid(image, levels+1, border_type)
    lap_pyr = []
    for n in range(levels):
        gau_n_1 = cv2.resize(gau_pyr[n+1], (gau_pyr[n].shape[1], gau_pyr[n].shape[0]), interpolation = cv2.INTER_CUBIC)
        upsampling(gau_pyr[n+1], (gau_pyr[n].shape[1], gau_pyr[n].shape[0]))
        lap_pyr.append(cv2.subtract(gau_pyr[n], gau_n_1) + 64) # Resta al nivel n el nivel n+1 y suma una constante para visualizarlo
    return lap_pyr

def piramideLaplaciana(vim):
    result = []
    for i in range(len(vim)-1):
        v1 = int(vim[i].shape[0]/2)
        v2 = int(vim[i].shape[1]/2)
        upsample = np.copy(vim[i+1])
        for j in range(v1):
            upsample = np.insert(upsample,2*j+1,upsample[2*j,:],axis=0)
        for j in range(v2):
            upsample = np.insert(upsample,2*j+1,upsample[:,2*j],axis=1)
        upsample = gaussiana2D(upsample,2)
        result.append(vim[i]-upsample)
    return result

"""Ejecución de ejemplos del ejercicio 2."""
def ejercicio_2(image):
    print("--- EJERCICIO 2A - GAUSSIAN PYRAMID ---")
    gau_pyr = gaussian_pyramid(image, 4, cv2.BORDER_CONSTANT)
    muestraMI(gau_pyr, 'Pirámide gaussiana')
    input("Pulsa 'Enter' para continuar\n")

    print("--- EJERCICIO 2B - LAPLACIAN PYRAMID ---")
    lap_pyr = laplacian_pyramid(image, 4, cv2.BORDER_CONSTANT)
    muestraMI(lap_pyr, 'Pirámide laplaciana')
    input("Pulsa 'Enter' para continuar\n")

    #print("--- EJERCICIO 2C - ESPACIO DE ESCALAS LAPLACIANO ---")


    #input("Pulsa 'Enter' para continuar\n")

# EJERCICIO 3 #

"""Construye una imagen híbrida con dos imagénes pasadas como argumento con el mismo tamaño.
Devuelve un vector con la imgagen de frecuencias bajas, altas y la híbrida respectivamente.
- im1: Imagen para frecuencias bajas
- im2: Imagen para frecuencias altas
- sigma1: Parámetro sigma para la imagen de frecuencias bajas
- sigma2: Parámetro sigma para la imagen de frecuencias altas"""
def hybridize_images(im1, im2, sigma1, sigma2):
    # Sacando las frecuencias a im1 usando alisado gaussiano
    frec_bajas = gaussian_blur(im1, sigma1)
    # Sacando las frecuencias altas a im2 restando alisado gaussiano
    frec_altas = cv2.subtract(im2, gaussian_blur(im2, sigma2))
    # cv2.addWeighted calcula la suma ponderada de dos matrices (ponderaciones 0.5 para cada matriz)
    return [frec_bajas, frec_altas, cv2.addWeighted(frec_bajas, 0.5, frec_altas, 0.5, 0)]

"""Ejecución de ejemplos del ejercicio 3."""
def ejercicio_3():
    print("--- EJERCICIO 3A - FUNCIÓN 'hybridize_images' IMPLEMENTADA ---")
    print("--- EJERCICIO 3B - MOSTRANDO 3 PAREJAS DE IMÁGENES HIBRIDADAS ---")
    # Leemos las imágenes en gris
    im_a1, im_a2 = leer_imagen("data/bird.bmp", 0), leer_imagen("data/plane.bmp", 0)
    im_b1, im_b2 = leer_imagen("data/dog.bmp", 0), leer_imagen("data/cat.bmp", 0)
    im_c1, im_c2 = leer_imagen("data/bicycle.bmp", 0), leer_imagen("data/motorcycle.bmp", 0)
    #im_d1, im_d2 = leer_imagen("data/fish.bmp", 0), leer_imagen("data/submarine.bmp", 0)
    #im_e1, im_e2 = leer_imagen("data/einstein.bmp", 0), leer_imagen("data/marilyn.bmp", 0)

    # Hibridamos las imágenes
    vim_a = hybridize_images(im_a1, im_a2, 3, 5)
    vim_b = hybridize_images(im_b1, im_b2, 9, 9)
    vim_c = hybridize_images(im_c1, im_c2, 9, 5)
    #vim_d = hybridize_images(im_d1, im_d2, 7, 7)
    #vim_e = hybridize_images(im_e1, im_e2, 3, 3)

    # Mostramos las hibridaciones
    muestraMI(vim_a, "Avión - Pájaro")
    muestraMI(vim_b, "Gato - Perro")
    muestraMI(vim_c, "Bicicleta - Moto")
    #muestraMI(vim_d, "Pez - Submarino")
    #muestraMI(vim_e, "Einstein - Marilyn")

    print("--- EJERCICIO 3C - MOSTRANDO PIRÁMIDES GAUSSIANAS DE LAS IMÁGENES HIBRIDADAS ---")
    # Construimos las pirámides gaussianas
    gau_pyr_a = gaussian_pyramid(vim_a[2], 4, cv2.BORDER_CONSTANT)
    gau_pyr_b = gaussian_pyramid(vim_b[2], 4, cv2.BORDER_CONSTANT)
    gau_pyr_c = gaussian_pyramid(vim_c[2], 4, cv2.BORDER_CONSTANT)
    #gau_pyr_d = gaussian_pyramid(vim_d[2], 4, cv2.BORDER_CONSTANT)
    #gau_pyr_e = gaussian_pyramid(vim_e[2], 4, cv2.BORDER_CONSTANT)

    # Imprimimos las pirámides gaussianas
    muestraMI(gau_pyr_a, 'Pirámide gaussiana Avión - Pájaro')
    muestraMI(gau_pyr_b, 'Pirámide gaussiana Gato - Perro')
    muestraMI(gau_pyr_c, 'Pirámide gaussiana Bicicleta - Moto')
    #muestraMI(gau_pyr_d, 'Pirámide gaussiana Pez - Submarino')
    #muestraMI(gau_pyr_e, 'Pirámide gaussiana Einstein - Marilyn')

    input("Pulsa 'Enter' para continuar\n")

#################
###   BONUS   ###
#################

# Bonus 1 #

def bonus_1():
    print("--- BONUS 1 - MÁSCARAS 2D CON CÓDIGO PROPIO. CUALQUIER MÁSCARA 2D DE NÚMEROS REALES USANDO MÁSCARAS SEPARABLES ---")

# Bonus 2 #

def bonus_2():
    print("--- BONUS 2 - TODAS LAS PAREJAS DE IMÁGENES HÍBRIDAS EN FORMATO A COLOR ---")
    # Leemos las imágenes en color
    im_a1, im_a2 = leer_imagen("data/bird.bmp", 1), leer_imagen("data/plane.bmp", 1)
    im_b1, im_b2 = leer_imagen("data/dog.bmp", 1), leer_imagen("data/cat.bmp", 1)
    im_c1, im_c2 = leer_imagen("data/bicycle.bmp", 1), leer_imagen("data/motorcycle.bmp", 1)
    im_d1, im_d2 = leer_imagen("data/fish.bmp", 1), leer_imagen("data/submarine.bmp", 1)
    im_e1, im_e2 = leer_imagen("data/einstein.bmp", 1), leer_imagen("data/marilyn.bmp", 1)

    # Hibridamos las imágenes
    vim_a = hybridize_images(im_a1, im_a2, 3, 5)
    vim_b = hybridize_images(im_b1, im_b2, 9, 9)
    vim_c = hybridize_images(im_c1, im_c2, 9, 5)
    vim_d = hybridize_images(im_d1, im_d2, 7, 7)
    vim_e = hybridize_images(im_e1, im_e2, 3, 3)

    # Mostramos las hibridaciones
    muestraMI(vim_a, "Avión - Pájaro")
    muestraMI(vim_b, "Gato - Perro")
    muestraMI(vim_c, "Bicicleta - Moto")
    muestraMI(vim_d, "Pez - Submarino")
    muestraMI(vim_e, "Einstein - Marilyn")

# Bonus 3 #

def bonus_3():
    print("--- BONUS 3 - IMAGEN HÍBRIDA CON PAREJA EXTRAIDA A MI ELECCIÓN ---")



################
###   MAIN   ###
################

def main():
    #im_color = leer_imagen('data/cat.bmp', 1)   # Leemos la imagen en color
    #ejercicio_1(im_color)
    #ejercicio_2(im_color)
    #ejercicio_3()
    #bonus_1()
    bonus_2()
    #bonus_3()

if __name__ == "__main__":
	main()
