#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Carlos Sánchez Muñoz
@date: 4 de diciembre de 2019
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

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



""" Ejecución de ejemplos del ejercicio 1A.
- image:
"""
def ejercicio_1A():
    print("--- EJERCICIO 1A - TIT ---")

    input("Pulsa 'Enter' para continuar\n")

""" Ejecución de ejemplos del ejercicio 1B.
- image:
"""
def ejercicio_1B():
    print("--- EJERCICIO 1B - TIT ---")

    input("Pulsa 'Enter' para continuar\n")

""" Ejecución de ejemplos del ejercicio 1C.
- image:
"""
def ejercicio_1C():
    print("--- EJERCICIO 1C - TIT ---")

    input("Pulsa 'Enter' para continuar\n")

""" Ejecución de ejemplos del ejercicio 1D.
- image:
"""
def ejercicio_1D():
    print("--- EJERCICIO 1D - TIT ---")

    input("Pulsa 'Enter' para continuar\n")


#######################
###   EJERCICIO 2   ###
#######################

""" Ejecución de ejemplos del ejercicio 2.
- image:
"""
def ejercicio_2():
    print("--- EJERCICIO 2 - TIT ---")

    input("Pulsa 'Enter' para continuar\n")

#######################
###   EJERCICIO 3   ###
#######################



""" Ejecución de ejemplos del ejercicio 3.
- image:
"""
def ejercicio_3():
    print("--- EJERCICIO 3 - TIT ---")

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
    ejercicio_1A()
    #ejercicio_1B()
    #ejercicio_1C()
    #ejercicio_1D()
    #ejercicio_2()
    #ejercicio_3()
    #ejercicio_4()
    #bonus_1()


if __name__ == "__main__":
	main()
