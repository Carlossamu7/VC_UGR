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

""" Lee una imagen ya sea en grises o en color. Devuelve la imagen.
- file_name: archivo de la imagen.
- flag_color (op): modo en el que se va a leer la imagen -> grises o color. Por defecto será en color.
"""
def leer_imagen(file_name, flag_color = 1):
    if flag_color == 0:
        print('Leyendo ' + file_name + ' en gris')
    elif flag_color==1:
        print('Leyendo ' + file_name + ' en color')
    else:
        print('flag_color debe ser 0 o 1')

    img = cv2.imread(file_name, flag_color)
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgrgb.astype(np.float64)
    return imgrgb

""" Normaliza una matriz.
- image: matriz a normalizar.
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

""" Imprime una imagen a través de una matriz.
- image_title: título de la imagen.
- image: imagen a imprimir.
"""
def pintaI(image_title, image):
    max = np.amax(image)
    min = np.amin(image)
    print(image)
    print(max)
    print(min)

    normaliza(image)        # normalizamos la matriz

    max = np.amax(image)
    min = np.amin(image)
    print(image)
    print(max)
    print(min)

    image.astype(np.uint8)
    plt.figure(0).canvas.set_window_title("Ejercicio")  # Ponemos nombre a la ventana
    plt.imshow(image)
    plt.title(image_title)  # Ponemos nombre a la imagen
    plt.show()

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

""" Las imágenes tienen que ser del mismo tipo ya que si no nos da el error:
        ValueError: all the input arrays must have same number of dimensions
    No es lo mismo la matriz de una imagen a color que necesita 3 dimensiones que una en grises
- image_list: lista de imágenes a concatenar.
- horizontal (op): modo de concatenación. 0-vertical y en otro caso horizontal. Por defecto horizontal.
"""
def pintaMI(image_list, horizontal=1):
    if horizontal != 0: # Salvo que sea 0-vertical, ponemos por defecto 1-horizontal.
        horizontal = 1

    concatenated_img = cv2.resize(image_list[0], (300,300), interpolation = cv2.INTER_AREA)

    for i in np.arange(1,len(image_list)):
        aux = cv2.resize(image_list[i], (300,300), interpolation = cv2.INTER_AREA)
        concatenated_img = np.concatenate((concatenated_img, aux), axis=horizontal)

    pintaI("Imágenes concatenadas", concatenated_img)

""" Muestra múltiples imágenes en una ventena Matplotlib.
- image_list: La lista de imágenes.
- image_title_list: Lista de títulos de las imágenes.
- rows: filas.
- columns: columnas.
"""
def imprimir_imagenes_titulos(image_list, image_title_list, rows, columns, window_title = 'Imágenes con títulos'):
    # Se igualan los tamaños de las Imagenes
    fig = plt.figure(0)
    fig.canvas.set_window_title(window_title)

    max = np.amax(image_list[0])
    min = np.amin(image_list[0])
    print(image_list[0])
    print(max)
    print(min)
    for i in range(len(image_list)):
        normaliza(image_list[i])
        image_list[i].astype(np.uint8)
    max = np.amax(image_list[0])
    min = np.amin(image_list[0])
    print(image_list[0])
    print(max)
    print(min)

    for i in range(rows * columns):
        if i < len(image_list):
            plt.subplot(rows, columns, i+1) # El índice (3er parametro) comienza en 1 en la esquina superior izquierda y aumenta a la derecha.
            plt.imshow(image_list[i])
            plt.title(image_title_list[i])
            plt.xticks([])  # Se le pasa una lista de posiciones en las que se deben colocar los
            plt.yticks([])  # ticks, si pasamos una lista vacía deshabilitamos los xticks e yticks
    plt.show()

######################
###   Práctica 1   ###
######################

# EJERCICIO 1 #

""" Aplica una máscara Gaussiana 2D. Devuelve la imagen con las máscara aplicada.
- image: la imagen.
- sigma_x: sigma en la dirección X.
- sigma_y (op): sigma en la dirección Y. Por defecto es 0.
- ksize (op): tamaño del kernel (2D, positivos e impares). Por defecto es (0,0), se obtiene a través de sigma.
- border_type (op): tipo de bordes. BORDER_DEFAULT.

La clave es la función GaussianBlur de cv2:
dst = cv.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType=BORDER_DEFAULT]]] )
"""
def gaussian_blur_original(image, sigma_x, sigma_y = 0, ksize = (0,0), border_type = cv2.BORDER_DEFAULT):
    return cv2.GaussianBlur(image, ksize, sigma_x, sigmaY = sigma_y, borderType = border_type)

def convolution(image, kernel_x, kernel_y, border_type = cv2.BORDER_DEFAULT):
    kernel_x = np.transpose(kernel_x)
    kernel_x = cv2.flip(kernel_x, 0)
    kernel_y = cv2.flip(kernel_y, 1)
    image = cv2.filter2D(image, -1, kernel_x, borderType = border_type)
    image = cv2.filter2D(image, -1, kernel_y, borderType = border_type)
    return image

def gaussian_blur(image, sigma_x, sigma_y, k_size_x = 0, k_size_y = 0, border_type = cv2.BORDER_DEFAULT):
    if k_size_x == 0:
        k_size_x = int(6*sigma_x + 1)
    if k_size_y == 0:
        k_size_y = int(6*sigma_y + 1)

    kernel_x = cv2.getGaussianKernel(k_size_x, sigma_x)
    kernel_y = cv2.getGaussianKernel(k_size_y, sigma_y)
    return convolution(image, kernel_x, kernel_y, border_type)

""" Obtiene máscaras 1D de máscaras derivadas. Devuelve los vectores de derivada
Argumentos posicionales:
- dx: orden de derivación respecto de x.
- dy: orden de derivación respecto de y.
- k_size: tamaño del kernel, puede ser 1, 3, 5, 7.
- border_type (op): tipo de bordes. BORDER_DEFAULT.
"""
def derive_convolution(image, dx, dy, k_size, border_type = cv2.BORDER_DEFAULT):
    print('Máscara de derivada con orden ({}, {}) y tamaño del kernel {}'.format(dx, dy, k_size))
    kx, ky = cv2.getDerivKernels(dx,dy,k_size)
    image = convolution(image, kx, ky, border_type)
    return image

""" Aplica máscara laplaciana a imagen. Devuelve la imagen con la máscara aplicada.
- im: Imagen a la que aplicar la máscara.
- k_size (op): Tamaño del kernel para Laplacian. Por defecto es 7.
- size (op): tamaño del kernel para Gaussian. Por defecto (0,0).
- border_type (op): Tipo de borde. Por defecto BORDER_DEFAULT.
"""
def laplacian_gaussian_original(image, sigma = 0, k_size = 7, border_type = cv2.BORDER_DEFAULT):
    # Reducimos ruido con alisado gaussiano
    blur = gaussian_blur(image, sigma, ksize = size, border_type = border_type)
    return cv2.Laplacian(blur, -1, ksize = k_size, borderType = border_type, delta = 50)

def laplacian_gaussian(image, sigma = 0, k_size = 5, size = (0, 0), border_type = cv2.BORDER_DEFAULT):
    k_x1, k_y1 = cv2.getDerivKernels(2, 0, k_size)
    k_x2, k_y2 = cv2.getDerivKernels(0, 2, k_size)
    im_convolution_x = convolution(image, k_x1, k_y1, border_type)
    im_convolution_y = convolution(image, k_x2, k_y2, border_type)
    return cv2.addWeighted(im_convolution_x, 1, im_convolution_y, 1, 0)

""" Ejecución de ejemplos del ejercicio 1A con diferentes σ y condiciones de contorno. """
def ejercicio_1A(image):
    print("--- EJERCICIO 1A - GAUSSIANA 2D Y MÁSCARAS 1D (getDerivKernels) ---")
    imprimir_imagenes_titulos([image, gaussian_blur(image, 2, 2, 5, 5), gaussian_blur(image, 6, 6, 7, 7)],
                              ['Original', 'σ_x = 2', 'σ_x = 6'], 1, 3, 'Gaussian')
    imprimir_imagenes_titulos([image, gaussian_blur(image, 2, 3, 5, 5), gaussian_blur(image, 6, 4, 5, 5)],
                              ['Original', 'σ_x = 2, σ_y = 3', 'σ_x = 6, σ_y = 4'], 1, 3, 'Gaussian')
    imprimir_imagenes_titulos([gaussian_blur(image, 1, 1, 5, 5, border_type=cv2.BORDER_DEFAULT), gaussian_blur(image, 1, 1, 5, 5, border_type=cv2.BORDER_REPLICATE),
                               gaussian_blur(image, 1, 1, 5, 5, border_type=cv2.BORDER_REFLECT), gaussian_blur(image, 1, 1, 5, 5, border_type=cv2.BORDER_CONSTANT)],
                              ['BORDER_DEFAULT', 'BORDER_REPLICATE', 'BORDER_REFLECT', 'BORDER_CONSTANT'], 2, 2, 'Gaussian with borders')

    # Máscaras de derivadas 1D
    tams = [3, 5]
    for tam in tams:
        imprimir_imagenes_titulos([derive_convolution(image, 1, 0, tam), derive_convolution(image, 0, 1, tam), derive_convolution(image, 1, 1, tam), derive_convolution(image, 2, 0, tam),
                                   derive_convolution(image, 0, 2, tam), derive_convolution(image, 2, 1, tam), derive_convolution(image, 1, 2, tam), derive_convolution(image, 2, 2, tam)],
                                  ['(1, 0)', '(0, 1)', '(1, 1)', '(2, 0)', '(0, 2)', '(2, 1)', '(1, 2)', '(2, 2)'], 3, 3, 'Máscaras de derivadas 1D')
    input("Pulsa 'Enter' para continuar\n")

""" Ejecución de ejemplos del ejercicio 1B con σ=1 y σ=3 y dos tipos de bordes """
def ejercicio_1B(image):
    print("--- EJERCICIO 1B -  LAPLACIANA DE GAUSSIANA ---")
    imprimir_imagenes_titulos([image, laplacian_gaussian(image, 1, border_type = cv2.BORDER_REPLICATE), laplacian_gaussian(image, 1, border_type = cv2.BORDER_CONSTANT)],
                       ['Original', 'σ = 1, REPLICATE', 'σ = 1, REFLECT'], 1, 3, 'Laplacian of gaussian')
    imprimir_imagenes_titulos([image, laplacian_gaussian(image, 3, border_type = cv2.BORDER_REPLICATE), laplacian_gaussian(image, 3, border_type = cv2.BORDER_REFLECT)],
                       ['Original', 'σ = 3, REPLICATE', 'σ = 3, REFLECT'], 1, 3, 'Laplacian of gaussian')
    input("Pulsa 'Enter' para continuar\n")

# EJERCICIO 2 #

""" Visualiza varias imágenes a la vez.
- image_list: Secuencia de imágenes.
"""
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
"""
def upsampling(image, n_fil, n_col):
    depth = image.shape[2]
    cp = np.zeros((n_fil, n_col, depth))

    for k in range(0, depth):
        for i in range(0, n_fil):
            if (i % 2) == 1:
                for j in range(0, n_col):
                    if (j % 2) == 1:
                        cp[i][j][k] = image[int(i/2), int(j/2), k]
                    else:
                        cp[i][j][k] = 0
            else:
                for j in range(0, n_col):
                    cp[i][j][k] = 0

    return cp

""" Genera representación de pirámide gaussiana. Devuelve la lista de imágenes que forman la pirámide gaussiana.
- image: La imagen a la que generar la pirámide gaussiana.
- levels (op): Número de niveles de la pirámide gaussiana. Por defecto 4.
- border_type (op): Tipo de borde a utilizar. Por defecto BORDER DEFAULT.
"""
def gaussian_pyramid(image, levels = 4, border_type = cv2.BORDER_CONSTANT):
    pyramid = [image]
    for n in range(levels):
        image = gaussian_blur(image, 1, 1, 3, 3, border_type = border_type)
        image = subsampling(image)
        pyramid.append(image)
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
        #gau_n_1 = cv2.resize(gau_pyr[n+1], (gau_pyr[n].shape[1], gau_pyr[n].shape[0]), interpolation = cv2.INTER_CUBIC)
        gau_n_1 = upsampling(gau_pyr[n+1], gau_pyr[n].shape[0], gau_pyr[n].shape[1])
        gau_n_1 = gau_n_1.astype(np.uint8)
        gau_pyr[n] = gau_pyr[n].astype(np.uint8)
        gau_n_1 = 4*gaussian_blur(gau_n_1, 5, 5)
        lap_pyr.append(cv2.subtract(gau_pyr[n], gau_n_1)+32) # Resta al nivel n el nivel n+1 y sumo una constante para visualizarlo
    return lap_pyr

""" Ejecución de ejemplos del ejercicio 2A. """
def ejercicio_2A(image):
    print("--- EJERCICIO 2A - GAUSSIAN PYRAMID ---")
    gau_pyr = gaussian_pyramid(image, 4, cv2.BORDER_CONSTANT)
    muestraMI(gau_pyr, 'Pirámide gaussiana')
    input("Pulsa 'Enter' para continuar\n")

""" Ejecución de ejemplos del ejercicio 2B. """
def ejercicio_2B(image):
    print("--- EJERCICIO 2B - LAPLACIAN PYRAMID ---")
    lap_pyr = laplacian_pyramid(image, 4, cv2.BORDER_CONSTANT)
    muestraMI(lap_pyr, 'Pirámide laplaciana')
    input("Pulsa 'Enter' para continuar\n")

""" Ejecución de ejemplos del ejercicio 2C. """
def ejercicio_2C(image):
    print("--- EJERCICIO 2C - ESPACIO DE ESCALAS LAPLACIANO ---")
    sigma = 1
    N = 4

    input("Pulsa 'Enter' para continuar\n")

# EJERCICIO 3 #

"""Construye una imagen híbrida con dos imagénes pasadas como argumento con el mismo tamaño.
Devuelve un vector con la imgagen de frecuencias bajas, altas y la híbrida respectivamente.
- im1: Imagen para frecuencias bajas.
- im2: Imagen para frecuencias altas.
- sigma1: Parámetro sigma para la imagen de frecuencias bajas.
- sigma2: Parámetro sigma para la imagen de frecuencias altas.
"""
def hybridize_images(im1, im2, sigma1, sigma2):
    # Sacando las frecuencias a im1 usando alisado gaussiano
    frec_bajas = gaussian_blur(im1, sigma1, sigma1)
    # Sacando las frecuencias altas a im2 restando alisado gaussiano
    frec_altas = cv2.subtract(im2, gaussian_blur(im2, sigma2, sigma2))
    # cv2.addWeighted calcula la suma ponderada de dos matrices (ponderaciones 0.5 para cada matriz)
    return [frec_bajas, frec_altas, cv2.addWeighted(frec_bajas, 0.5, frec_altas, 0.5, 0)]

""" Ejecución de ejemplos del ejercicio 3B. """
def ejercicio_3B(im_1, im_2, sigma1, sigma2, title = "Hibridación de imágenes"):
    vim = hybridize_images(im_1, im_2, sigma1, sigma2)  # Hibridamos las imágenes
    muestraMI(vim, title)                               # Mostramos las hibridaciones
    return vim

""" Ejecución de ejemplos del ejercicio 3C. """
def ejercicio_3C(vim, title = "Pirámide gaussiana de la hibridada", levels = 4, border_type = cv2.BORDER_CONSTANT):
    gau_pyr = gaussian_pyramid(vim[2], 4, border_type)  # Construimos las pirámides gaussianas
    muestraMI(gau_pyr, title)                           # Imprimimos las pirámides gaussianas
    return gau_pyr


#################
###   BONUS   ###
#################

# Bonus 1 #

""" Calcula correlación 1D de vector con señal. Devuelve la señal con correlación.
- mascara: vector-máscara.
- orig: Señal original.
"""
def correl(mascara, orig):
    if len(orig.shape) == 2: # si es multibanda
        NCH = orig.shape[1]
        return np.stack((bonus2(mascara, orig[::,j]) for j in range(NCH)), axis = 1)

    nueva = np.zeros(orig.shape) # Crea nueva imagen
    N, M = len(orig), (len(mascara)-1)//2
    extended = np.concatenate((orig[::-1], orig, orig[::-1]))

    for i in range(N):
        nueva[i] = np.dot(mascara, extended[i-M+N:i+M+N+1])
    return nueva

"""Calcula el vector máscara gaussiano. Devuelve el vector máscara gaussiano.
- sigma: Parámetro σ de la función de densidad de la gaussiana.
"""
def gaussian_vector(sigma):
    longitud = 1 + 2*int(3*sigma) # Calcula la longitud
    mid = int(3*sigma)

    f = lambda x: math.exp(-0.5*x*x/(sigma*sigma))
    mascara = np.zeros(longitud)

    # Rellena la máscara muestreando
    for n in range(longitud):
        x = n - mid
        mascara[n] = f(x)

    return mascara/np.sum(mascara)

"""Convolución 2D usando máscaras separables. Devuelve la imagen convolucionada.
- vX: Vector-máscara en dirección X.
- vY: Vector-máscara en dirección Y.
- im: Imagen a convolucionar.
"""
def bonus_1(vX, vY, im):
    print("--- BONUS 1 - MÁSCARAS 2D CON CÓDIGO PROPIO. CUALQUIER MÁSCARA 2D DE NÚMEROS REALES USANDO MÁSCARAS SEPARABLES ---")
    if not isBW(im): # Si tiene 3 canales
      canales  = cv.split(im)
      return cv.merge([bonus3(vX, vY, canal) for canal in canales])

    nueva = im.copy()
    N, M = im.shape
    rVX = vX[::-1]
    rVY = vY[::-1]

    for j in range(M): # Aplica convolución por columnas
        nueva[::,j] = correl(rVX, nueva[::, j])
    for i in range(N): # Aplica convolución por filas
        nueva[i,::] = correl(rVY, nueva[i, ::])

    return nueva


""" Combina ejemplos para mostrar funcionalidad en ejercicios bonus 1, 2 y 3. """
def ejemploB123(im):
  vGauss = bonus1(1)
  gauss  = bonus3(vGauss, vGauss, im)
  pintaMI((im, "Original"), (gauss, "Gaussiana propia sigma = 3"))


# Bonus 2 #

""" Ejecución del bonus 2. """
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

    input("Pulsa 'Enter' para continuar\n")

# Bonus 3 #

""" Ejecución del bonus 3. """
def bonus_3(im_1, im_2, sigma_1, sigma_2, image_title):
    print("--- BONUS 3 - IMAGEN HÍBRIDA CON ELECCIÓN DE PAREJA '" + image_title + "' ---")
    # Las dos imágenes han de tener el mismo tamaño por lo que  calculo mínimos de ancho y alto
    min_alt = min(im_1.shape[0], im_2.shape[0])
    # Hago resize a los mínimos de ambas imágenes porque una podría ser más ancha y la otra más alta.
    min_anc = min(im_1.shape[1], im_2.shape[1])
    im_1 = cv2.resize(im_1, (min_anc, min_alt), im_1, interpolation = cv2.INTER_CUBIC)
    im_2 = cv2.resize(im_2, (min_anc, min_alt), im_2, interpolation = cv2.INTER_CUBIC)
    # Hibrido y muestro las imágenes
    vim = hybridize_images(im_1, im_2, sigma_1, sigma_2)
    muestraMI(vim, image_title)

    input("Pulsa 'Enter' para continuar\n")


################
###   MAIN   ###
################

def main():
    im_cat_c = leer_imagen('data/cat.bmp', 1)   # Leemos la imagen en color

    ejercicio_1A(im_cat_c)
    ejercicio_1B(im_cat_c)

    ejercicio_2A(im_cat_c)
    ejercicio_2B(im_cat_c)
    ejercicio_2C(im_cat_c)

    print("--- EJERCICIO 3A - FUNCIÓN 'hybridize_images' IMPLEMENTADA ---")

    print("--- EJERCICIO 3B - MOSTRANDO PAREJAS DE IMÁGENES HIBRIDADAS ---")

    # Leemos las imágenes en gris
    im_bird_g, im_plane_g = leer_imagen("data/bird.bmp", 0), leer_imagen("data/plane.bmp", 0)
    im_dog_g, im_cat_g = leer_imagen("data/dog.bmp", 0), leer_imagen("data/cat.bmp", 0)
    im_bicycle_g, im_motorcycle_g = leer_imagen("data/bicycle.bmp", 0), leer_imagen("data/motorcycle.bmp", 0)
    #im_fish_g, im_submarine_g = leer_imagen("data/fish.bmp", 0), leer_imagen("data/submarine.bmp", 0)
    #im_einstein_g, im_marilyn_g = leer_imagen("data/einstein.bmp", 0), leer_imagen("data/marilyn.bmp", 0)

    # Ejecución de la hibridación y mostrado de imágenes
    vim_1 = ejercicio_3B(im_bird_g, im_plane_g, 3, 5, "Avión - Pájaro")
    vim_2 = ejercicio_3B(im_dog_g, im_cat_g, 9, 9, "Gato - Perro")
    vim_3 = ejercicio_3B(im_bicycle_g, im_motorcycle_g, 9, 5, "Bicicleta - Moto")
    #vim_4 = ejercicio_3B(im_fish_g, im_submarine_g, 7, 7, "Pez - Submarino")
    #vim_5 = ejercicio_3B(im_einstein_g, im_marilyn_g, 3, 3, "Einstein - Marilyn")
    input("Pulsa 'Enter' para continuar\n")

    print("--- EJERCICIO 3C - MOSTRANDO PIRÁMIDES GAUSSIANAS DE LAS IMÁGENES HIBRIDADAS ---")
    ejercicio_3C(vim_1, "Pirámide gaussiana Avión - Pájaro")
    ejercicio_3C(vim_2, "Pirámide gaussiana Gato - Perro")
    ejercicio_3C(vim_3, "Pirámide gaussiana Bicicleta - Moto")
    #ejercicio_3C(vim_4, "Pirámide gaussiana Pez - Submarino")
    #ejercicio_3C(vim_5, "Pirámide gaussiana Einstein - Marilyn")
    input("Pulsa 'Enter' para continuar\n")

    #bonus_1()
    bonus_2()
    im_1a, im_1b = leer_imagen("data/guitarra.png", 1), leer_imagen("data/violin.png", 1)
    im_2a, im_2b = leer_imagen("data/trompeta.jpg", 1), leer_imagen("data/saxofon.jpg", 1)
    bonus_3(im_1a, im_1b, 9, 9, "Guitarra - Violín")
    bonus_3(im_2a, im_2b, 3, 7, "Trompeta - Saxofón")

if __name__ == "__main__":
	main()
