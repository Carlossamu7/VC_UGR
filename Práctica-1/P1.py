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
        print("Leyendo " + file_name + " en gris")
    elif flag_color==1:
        print("Leyendo " + file_name + " en color")
    else:
        print("flag_color debe ser 0 o 1")

    img = cv2.imread(file_name, flag_color)

    if flag_color==1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float64)
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

######################
###   Práctica 1   ###
######################

# FUNCIONES ANTIGUAS #

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

# EJERCICIO 1

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

""" Obtiene máscaras 1D de máscaras derivadas. Devuelve los vectores de derivada
- image: la imagen a tratar.
- dx: orden de derivación respecto de x.
- dy: orden de derivación respecto de y.
- k_size: tamaño del kernel, puede ser 1, 3, 5, 7.
- border_type (op): tipo de bordes. BORDER_DEFAULT.
"""
def derive_convolution(image, dx, dy, k_size, border_type = cv2.BORDER_DEFAULT):
    print('Máscara de derivada con orden ({}, {}) y tamaño del kernel {}'.format(dx, dy, k_size))
    kx, ky = cv2.getDerivKernels(dx,dy,k_size)
    im_conv = convolution(image, kx, ky, border_type)
    return im_conv

""" Aplica máscara laplaciana a imagen. Devuelve la imagen con la máscara aplicada.
- im: Imagen a la que aplicar la máscara.
- k_size: Tamaño del kernel para Laplacian.
- border_type (op): Tipo de borde. Por defecto BORDER_DEFAULT.
"""
def laplacian_gaussian(image, k_size, border_type = cv2.BORDER_DEFAULT):
    k_x1, k_y1 = cv2.getDerivKernels(2, 0, k_size, normalize = True)
    k_x2, k_y2 = cv2.getDerivKernels(0, 2, k_size, normalize = True)
    im_convolution_x = convolution(image, k_x1, k_y1, border_type)
    im_convolution_y = convolution(image, k_x2, k_y2, border_type)
    return im_convolution_x + im_convolution_y

""" Ejecución de ejemplos del ejercicio 1A con diferentes σ y condiciones de contorno.
- image: imagen a tratar.
- flag_color (op): modo en el que se van a leer las imágenes. Por defecto en color.
"""
def ejercicio_1A(image, flag_color):
    print("--- EJERCICIO 1A - GAUSSIANA 2D Y MÁSCARAS 1D (getDerivKernels) ---")
    imprimir_imagenes_titulos([image, gaussian_blur(image, 2, 2, 5, 5), gaussian_blur(image, 4, 4, 7, 7), gaussian_blur(image, 1, 2)],
                              ['Original', 'σ_x = 2, σ_y = 2, ksize=(5,5)', 'σ_x = 4, σ_y = 4, ksize=(7,7)', 'σ_x = 1, σ_y = 2, ksize~σ'], 2, 2, flag_color, 'Gaussian with different σ')
    imprimir_imagenes_titulos([gaussian_blur(image, 1, 1, 5, 5, border_type=cv2.BORDER_DEFAULT), gaussian_blur(image, 1, 1, 5, 5, border_type=cv2.BORDER_REPLICATE),
                               gaussian_blur(image, 1, 1, 5, 5, border_type=cv2.BORDER_REFLECT), gaussian_blur(image, 1, 1, 5, 5, border_type=cv2.BORDER_CONSTANT)],
                              ['BORDER_DEFAULT', 'BORDER_REPLICATE', 'BORDER_REFLECT', 'BORDER_CONSTANT'], 2, 2, flag_color, 'Gaussian with different borders')

    # Máscaras de derivadas 1D
    tam_list = [3, 5]
    for tam in tam_list:
        pintaI(derive_convolution(image, 1, 0, tam), flag_color, '(1,0)', "Ejercicio 1A - tamaño {}".format(tam))
        pintaI(derive_convolution(image, 0, 1, tam), flag_color, '(0,1)', "Ejercicio 1A - tamaño {}".format(tam))
        pintaI(derive_convolution(image, 1, 1, tam), flag_color, '(1,1)', "Ejercicio 1A - tamaño {}".format(tam))
        pintaI(derive_convolution(image, 2, 0, tam), flag_color, '(2,0)', "Ejercicio 1A - tamaño {}".format(tam))
        pintaI(derive_convolution(image, 0, 2, tam), flag_color, '(0,2)', "Ejercicio 1A - tamaño {}".format(tam))
        pintaI(derive_convolution(image, 2, 1, tam), flag_color, '(2,1)', "Ejercicio 1A - tamaño {}".format(tam))
        pintaI(derive_convolution(image, 1, 2, tam), flag_color, '(1,2)', "Ejercicio 1A - tamaño {}".format(tam))
        pintaI(derive_convolution(image, 2, 2, tam), flag_color, '(2,2)', "Ejercicio 1A - tamaño {}".format(tam))

    input("Pulsa 'Enter' para continuar\n")

""" Ejecución de ejemplos del ejercicio 1B con σ=1 y σ=3 y dos tipos de bordes.
- image: imagen a tratar.
- flag_color (op): modo en el que se van a leer las imágenes. Por defecto en color.
"""
def ejercicio_1B(image, flag_color = 1):
    print("--- EJERCICIO 1B -  LAPLACIANA DE GAUSSIANA ---")
    # PARA NORMALIZAR MULTIPLICO POR SIGMA^2
    imprimir_imagenes_titulos([image, laplacian_gaussian(image, 7, border_type = cv2.BORDER_DEFAULT), laplacian_gaussian(image, 7, border_type = cv2.BORDER_REPLICATE), laplacian_gaussian(image, 7, border_type = cv2.BORDER_REFLECT)],
                       ['Original', 'σ = 1, DEFAULT', 'σ = 1, REPLICATE', 'σ = 1, REFLECT'], 2, 2, flag_color, 'Laplacian of gaussian with σ = 1')
    imprimir_imagenes_titulos([image, 9*laplacian_gaussian(image, 19, border_type = cv2.BORDER_DEFAULT), 9*laplacian_gaussian(image, 19, border_type = cv2.BORDER_REPLICATE), 9*laplacian_gaussian(image, 19, border_type = cv2.BORDER_REFLECT)],
                       ['Original', 'σ = 3, DEFAULT', 'σ = 3, REPLICATE', 'σ = 3, REFLECT'], 2, 2, flag_color, 'Laplacian of gaussian with σ = 3')
    input("Pulsa 'Enter' para continuar\n")

# EJERCICIO 2 #

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
def upsampling_v1(image, n_fil, n_col):
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

# Sólo vale si la matriz tiene tamaño par.
def upsampling_v2(image, n_fil, n_col):
    depth = image.shape[2]
    salida = np.zeros((n_fil, n_col, depth))

    for k in range(0, depth):
        salida[:,:,k][::2,::2] = image[:,:,k]
        salida[:,:,k][1::2,::2] = image[:,:,k]
        salida[:,:,k][::2,1::2] = image[:,:,k]
        salida[:,:,k][1::2,1::2] = image[:,:,k]

    return salida

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

""" Eleva al cuadrado cada píxel.
- image: imagen a tratar
"""
def eleva_cuadrado(image):
    salida = np.zeros(image.shape)

    if len(image.shape) == 2:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                salida[i][j] = image[i][j] * image[i][j]
    elif len(image.shape) == 3:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    salida[i][j][k] = image[i][j][k] * image[i][j][k]

    return salida

""" Supresión de de no máximos.
- image: imagen a tratar
"""
def non_maximum_supression(image):
    res = np.zeros(image.shape)

    if len(image.shape) == 2:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                max = 0
                if i>=1 and j>=1:
                    if max<image[i-1][j-1]:
                        max = image[i-1][j-1]
                if i>=1:
                    if max<image[i-1][j]:
                        max = image[i-1][j]
                if i>=1 and j<(image.shape[1]-1):
                    if max<image[i-1][j+1]:
                        max = image[i-1][j+1]
                if j>=1:
                    if max<image[i][j-1]:
                        max = image[i][j-1]
                if j<(image.shape[1]-1):
                    if max<image[i][j+1]:
                        max = image[i][j+1]
                if i<(image.shape[0]-1) and j>=1:
                    if max<image[i+1][j-1]:
                        max = image[i+1][j-1]
                if i<(image.shape[0]-1):
                    if max<image[i+1][j]:
                        max = image[i+1][j]
                if i<(image.shape[0]-1) and j<(image.shape[1]-1):
                    if max<image[i+1][j+1]:
                        max = image[i+1][j+1]

                if max<image[i][j]:
                    res[i][j] = image[i][j]
                else:
                    res[i][j] = 0

    elif len(image.shape) == 3:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    max = 0
                    if i>=1 and j>=1:
                        if max<image[i-1][j-1][k]:
                            max = image[i-1][j-1][k]
                    if i>=1:
                        if max<image[i-1][j][k]:
                            max = image[i-1][j][k]
                    if i>=1 and j<(image.shape[1]-1):
                        if max<image[i-1][j+1][k]:
                            max = image[i-1][j+1][k]
                    if j>=1:
                        if max<image[i][j-1][k]:
                            max = image[i][j-1][k]
                    if j<(image.shape[1]-1):
                        if max<image[i][j+1][k]:
                            max = image[i][j+1][k]
                    if i<(image.shape[0]-1) and j>=1:
                        if max<image[i+1][j-1][k]:
                            max = image[i+1][j-1][k]
                    if i<(image.shape[0]-1):
                        if max<image[i+1][j][k]:
                            max = image[i+1][j][k]
                    if i<(image.shape[0]-1) and j<(image.shape[1]-1):
                        if max<image[i+1][j+1][k]:
                            max = image[i+1][j+1][k]

                    if max<image[i][j][k]:
                        res[i][j][k] = image[i][j][k]
                    else:
                        res[i][j][k] = 0

    return res

""" Función que selecciona las regiones.
- image: imagen sobre la que señalar las regiones.
- scale: escala del espacio laplaciano.
- umbral: umbral para detectar la región.
- radio: radio de los círculos.
"""
def select_regions(image, scale, umbral, radio):
    res = np.copy(image)

    if len(image.shape) == 2:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if scale[i][j] > umbral:
                    cv2.circle(res, (j, i), radio, 1)

    elif len(image.shape) == 3:
        for k in range(image.shape[2]):
            res[:,:,k] = select_regions(image[:,:,k], scale[:,:,k], umbral, radio)

    return res

""" Ejecución de ejemplos del ejercicio 2A.
- image: imagen a tratar
- flag_color (op): bandera para indicar si la imagen es en B/N o color. Por defecto color.
"""
def ejercicio_2A(image, flag_color = 1):
    print("--- EJERCICIO 2A - GAUSSIAN PYRAMID ---")
    gau_pyr = gaussian_pyramid(image, 4, cv2.BORDER_DEFAULT)
    muestraMI(gau_pyr, flag_color, "Pirámide gaussiana")
    input("Pulsa 'Enter' para continuar\n")

""" Ejecución de ejemplos del ejercicio 2B.
- image: imagen a tratar
- flag_color (op): bandera para indicar si la imagen es en B/N o color. Por defecto color.
"""
def ejercicio_2B(image, flag_color = 1):
    print("--- EJERCICIO 2B - LAPLACIAN PYRAMID ---")
    lap_pyr = laplacian_pyramid(image, 4, cv2.BORDER_DEFAULT)
    muestraMI(lap_pyr, flag_color, "Pirámide laplaciana")
    input("Pulsa 'Enter' para continuar\n")

""" Ejecución de ejemplos del ejercicio 2C.
- image: imagen a tratar.
- sigma: sigma usado en la laplaciana de gaussiana.
- k: constante de incrementación de sigma.
- umbral (op): umbral para detectar la región. Por defecto 120.
- levels (op): número de escalas. Por defecto 4.
- flag_color (op): bandera para indicar si la imagen es en B/N o color. Por defecto color.
"""
def ejercicio_2C(image, sigma, k, umbral = 120, levels = 4, flag_color = 1):
    print("--- EJERCICIO 2C - ESPACIO DE ESCALAS LAPLACIANO ---")
    if len(image.shape) == 2:
        scale = np.zeros((levels + 1, image.shape[0], image.shape[1]))
        im = np.zeros((levels + 1, image.shape[0], image.shape[1]))
    elif len(image.shape) == 3:
        scale = np.zeros((levels + 1, image.shape[0], image.shape[1], image.shape[2]))
        im = np.zeros((levels + 1, image.shape[0], image.shape[1], image.shape[2]))

    scale[0] = np.copy(image)

    for i in range(1, levels + 1):
        scale[i] = sigma * sigma * laplacian_gaussian(image, 5)
        scale[i] = eleva_cuadrado(scale[i])
        scale[i] = non_maximum_supression(scale[i])
        sigma = k * sigma
        scale[i] = normaliza(scale[i], "Escala laplaciana número {}".format(i))
        im[i] = np.copy(image)
        im[i] = select_regions(im[i], scale[i], umbral, int(17*sigma))
        #pintaI(scale[i], flag_color, "Escala laplaciana número {}".format(i), "Espacio de escalas laplaciano")
        pintaI(im[i], flag_color, "Escala laplaciana número {} sobre la imagen original".format(i), "Espacio de escalas laplaciano")

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

""" Ejecución de ejemplos del ejercicio 3B.
- im1: Imagen para frecuencias bajas.
- im2: Imagen para frecuencias altas.
- sigma1: Parámetro sigma para la imagen de frecuencias bajas.
- sigma2: Parámetro sigma para la imagen de frecuencias altas.
- image_title: título de la imagen
"""
def ejercicio_3_2(im1, im2, sigma1, sigma2, title = "Hibridación de imágenes"):
    vim = hybridize_images(im1, im2, sigma1, sigma2)   # Hibridamos las imágenes
    muestraMI(vim, 0, title)                           # Mostramos las hibridaciones (B/N)
    return vim

""" Ejecución de ejemplos del ejercicio 3C.
- vim: vector de imágenes resultante de la hibridación
- title (op): título de la imagen. Por defecto "Pirámide gaussiana de la hibridada".
- levels (op): niveles de la pirámide. Por defecto 4.
- border_type (op): tipo de borde. Por defecto BORDER_CONSTANT.
"""
def ejercicio_3_3(vim, title = "Pirámide gaussiana de la hibridada", levels = 4, border_type = cv2.BORDER_CONSTANT):
    gau_pyr = gaussian_pyramid(vim[2], 4, border_type) # Construimos las pirámides gaussianas
    muestraMI(gau_pyr, 0, title)                       # Imprimimos las pirámides gaussianas (B/N)
    return gau_pyr

#################
###   BONUS   ###
#################

# Bonus 1 #

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

""" Ejecución del bonus 1.
- image: imagen a tratar.
- kernel_x: kernel en las dirección X.
- kernel_y: kernel en las dirección Y.
- flag_color (op): bandera para indicar si la imagen es en B/N o color. Por defecto color.
"""
def bonus_1(image, kernel_x, kernel_y, flag_color = 1):
    im = convolution2D(image, kernel_x, kernel_y)
    pintaI(im, flag_color, "Convolución con máscaras separables (Bonus 1)")

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

    # Mostramos las hibridaciones (COLOR)
    muestraMI(vim_a, 1, "Avión - Pájaro", "Hibridación en  color")
    muestraMI(vim_b, 1, "Gato - Perro", "Hibridación en  color")
    muestraMI(vim_c, 1, "Bicicleta - Moto", "Hibridación en  color")
    muestraMI(vim_d, 1, "Pez - Submarino", "Hibridación en  color")
    muestraMI(vim_e, 1, "Einstein - Marilyn", "Hibridación en  color")

    input("Pulsa 'Enter' para continuar\n")

# Bonus 3 #

""" Ejecución del bonus 3.
- im1: Imagen para frecuencias bajas.
- im2: Imagen para frecuencias altas.
- sigma1: Parámetro sigma para la imagen de frecuencias bajas.
- sigma2: Parámetro sigma para la imagen de frecuencias altas.
- flag_color (op): bandera para indicar si la imagen es en B/N o color. Por defecto color.
- image_title (op): título de la imagen. Por defecto "Imagen elegida por mi".
"""
def bonus_3(im_1, im_2, sigma_1, sigma_2, flag_color = 1, image_title = "Imagen elegida por mi"):
    print("--- BONUS 3 - IMAGEN HÍBRIDA CON ELECCIÓN DE PAREJA '" + image_title + "' ---")
    # Las dos imágenes han de tener el mismo tamaño por lo que  calculo mínimos de ancho y alto
    min_alt = min(im_1.shape[0], im_2.shape[0])
    # Hago resize a los mínimos de ambas imágenes porque una podría ser más ancha y la otra más alta.
    min_anc = min(im_1.shape[1], im_2.shape[1])
    im_1 = cv2.resize(im_1, (min_anc, min_alt), im_1, interpolation = cv2.INTER_CUBIC)
    im_2 = cv2.resize(im_2, (min_anc, min_alt), im_2, interpolation = cv2.INTER_CUBIC)
    # Hibrido y muestro las imágenes
    vim = hybridize_images(im_1, im_2, sigma_1, sigma_2)
    muestraMI(vim, flag_color, image_title)
    input("Pulsa 'Enter' para continuar\n")


################
###   MAIN   ###
################

def main():
    flag_color = 1
    im_cat = leer_imagen('data/cat.bmp', flag_color)    # Leemos la imagen en color
    im_cat_gray = leer_imagen('data/cat.bmp', 0)        # Leemos la imagen en color

    ejercicio_1A(im_cat, flag_color)
    ejercicio_1B(im_cat, flag_color)

    ejercicio_2A(im_cat, flag_color)
    ejercicio_2B(im_cat_gray, 0)
    #ejercicio_2B(im_cat, flag_color)
    ejercicio_2C(im_cat_gray, 1, 1.2, 140, 4, 0)
    ejercicio_2C(im_cat, 1, 1.2, 140, 4, flag_color)

    print("--- EJERCICIO 3A - FUNCIÓN 'hybridize_images' IMPLEMENTADA ---")
    print("--- EJERCICIO 3B - MOSTRANDO PAREJAS DE IMÁGENES HIBRIDADAS ---")

    # Leemos las imágenes en gris
    im_bird_g, im_plane_g = leer_imagen("data/bird.bmp", 0), leer_imagen("data/plane.bmp", 0)
    im_dog_g, im_cat_g = leer_imagen("data/dog.bmp", 0), leer_imagen("data/cat.bmp", 0)
    im_bicycle_g, im_motorcycle_g = leer_imagen("data/bicycle.bmp", 0), leer_imagen("data/motorcycle.bmp", 0)
    #im_fish_g, im_submarine_g = leer_imagen("data/fish.bmp", 0), leer_imagen("data/submarine.bmp", 0)
    #im_einstein_g, im_marilyn_g = leer_imagen("data/einstein.bmp", 0), leer_imagen("data/marilyn.bmp", 0)

    # Ejecución de la hibridación y mostrado de imágenes
    vim_1 = ejercicio_3_2(im_bird_g, im_plane_g, 3, 5, "Avión - Pájaro")
    vim_2 = ejercicio_3_2(im_dog_g, im_cat_g, 9, 9, "Gato - Perro")
    vim_3 = ejercicio_3_2(im_bicycle_g, im_motorcycle_g, 9, 5, "Bicicleta - Moto")
    #vim_4 = ejercicio_3_2(im_fish_g, im_submarine_g, 7, 7, "Pez - Submarino")
    #vim_5 = ejercicio_3_2(im_einstein_g, im_marilyn_g, 3, 3, "Einstein - Marilyn")
    input("Pulsa 'Enter' para continuar\n")

    print("--- EJERCICIO 3C - MOSTRANDO PIRÁMIDES GAUSSIANAS DE LAS IMÁGENES HIBRIDADAS ---")
    ejercicio_3_3(vim_1, "Pirámide gaussiana Avión - Pájaro")
    ejercicio_3_3(vim_2, "Pirámide gaussiana Gato - Perro")
    ejercicio_3_3(vim_3, "Pirámide gaussiana Bicicleta - Moto")
    #ejercicio_3_3(vim_4, "Pirámide gaussiana Pez - Submarino")
    #ejercicio_3_3(vim_5, "Pirámide gaussiana Einstein - Marilyn")
    input("Pulsa 'Enter' para continuar\n")

    print("--- BONUS 1 - CONVOLUCIÓN 2D CON MÁSCARAS SEPARABLES 2D DE NÚMEROS REALES ---")
    # Voy a ejecutar el bonus 1 3 veces: gaussiana a color y en B/N y getDerivKernels de orden (1,1)
    print("Versiones gris y color con máscara gaussiana con σ=1 y tam=7")
    kx = ky = cv2.getGaussianKernel(7, 1)
    bonus_1(im_cat_gray, kx, ky, 0)
    bonus_1(im_cat, kx, ky, flag_color)
    print("Versión gris con máscara de derivada de órden ({}, {}) y k_size = {}".format(1, 1, 7))
    kx, ky = cv2.getDerivKernels(1, 1, 7, normalize = True)
    bonus_1(im_cat_gray, kx, ky, 0)

    bonus_2()

    # Bonus 3. Muestro 2 ejemplos, el más intersante es el de la guitarra y el violín.
    flag_my_img = 1
    im_1a, im_1b = leer_imagen("data/guitarra.png", flag_my_img), leer_imagen("data/violin.png", flag_my_img)
    im_2a, im_2b = leer_imagen("data/trompeta.jpg", flag_my_img), leer_imagen("data/saxofon.jpg", flag_my_img)
    bonus_3(im_1a, im_1b, 9, 9, flag_my_img, "Guitarra - Violín")
    bonus_3(im_2a, im_2b, 3, 7, flag_my_img, "Trompeta - Saxofón")

if __name__ == "__main__":
	main()
