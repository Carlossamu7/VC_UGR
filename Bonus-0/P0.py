#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:24:38 2019

@author: Carlos Sánchez Muñoz
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

""" Uso la notación Snake Case la cual es habitual en Python """

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
def pintaMI(image_list, horizontal):
    if horizontal != 0: # Salvo que sea 0-vertical, ponemos por defecto 1-horizontal.
        horizontal = 1

    concatenated_img = cv2.resize(image_list[0], (300,300), interpolation = cv2.INTER_AREA)

    for i in np.arange(1,len(image_list)):
        aux = cv2.resize(image_list[i], (300,300), interpolation = cv2.INTER_AREA)
        concatenated_img = np.concatenate((concatenated_img, aux), axis=horizontal)

    pintaI("Imágenes concatenadas", concatenated_img)

""" Modifica el color de cada elemento de una lista de pixels
- image: La imagen
- pixels: lista a modificar
- color_pixels: color al que vamos a modificar dichos pixels
"""
def modificar_pixels(image, pixels, color_pixels = (0,0,0)):
  if len(image.shape) == 2:    # imagen está en blanco y negro
    color = 0
  for pixel in pixels:
    image[pixel] = color_pixels

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
def imprimir_imagenes_titulos(image_list, image_title_list, rows, columns):
    # Se igualan los tamaños de las Imagenes
    image_list = escalar_imagenes(image_list)
    fig = plt.figure(0)
    fig.canvas.set_window_title('Imágenes con títulos')

    for i in range(rows * columns):
        if i < len(image_list):
            plt.subplot(rows, columns, i+1) # El índice (3er parametro) comienza en 1 en la esquina superior izquierda y aumenta a la derecha.
            imgrgb = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2RGB)
            plt.imshow(imgrgb)
            plt.title(image_title_list[i])
            plt.xticks([])  # Se le pasa una lista de posiciones en las que se deben colocar los
            plt.yticks([])  # ticks, si pasamos una lista vacía deshabilitamos los xticks e yticks
    plt.show()

def main():
    nombre_imagen = 'logoOpenCV'

    # EJERCICIO 1: Escribir una función que lea el fichero de una imagen y la muestre tanto
    # en grises como en color ( im=leeimagen(filename, flagColor))
    print("--- EJERCICIO 1 - LEYENDO IMÁGENES ---")
    imagen_gris = leer_imagen('images/'+nombre_imagen+'.jpg', 0)    # Leemos la imagen en gris
    imagen_color = leer_imagen('images/'+nombre_imagen+'.jpg', 1)   # Leemos la imagen en color


    # EJERCICIO 2: Escribir una función que visualice una matriz de números reales cualquiera ya
    # sea monobanda o tribanda (pintaI(im)). Deberá escalar y normalizar sus valores.
    print("--- EJERCICIO 2 - IMPRIMIENDO IMÁGENES ---")
    pintaI('Imagen en gris', imagen_gris)       # Mostramos la imagen en gris
    pintaI('Imagen en color', imagen_color)     # Mostramos la imagen en color
    input("Pulsa 'Enter' para continuar\n")

    # EJERCICIO 3: Escribir una función que visualice varias imágenes a la vez: pintaMI(vim).
    # (vim será una secuencia de imágenes) ¿Qué pasa si las imágenes no son todas del mismo
    # tipo: (nivel de gris, color, blanco-negro)?
    print("--- EJERCICIO 3 - CONCATENANDO IMÁGENES ---")
    vim = leer_lista_imagenes(['images/dave.jpg', 'images/logoOpenCV.jpg', 'images/messi.jpg', 'images/orapple.jpg'], 1)
    pintaMI(vim, 1) # 2do parámetro indica si horizontal o vertical
    input("Pulsa 'Enter' para continuar\n")

    # EJERCICIO 4: Escribir una función que modifique el color en la imagen de cada uno de los elementos
    # de una lista de coordenadas de píxeles. (Recordar que (fila, columna es lo contrario a (x,y). Es decir fila=y, columna=x)
    print("--- EJERCICIO 4 - MODIFICANDO PIXELS ---")
    color = (255,0,0)   #azul
    # Pongo el color anterior en los pixeles que quiera. En este caso quito la fila 50, la columna 30 y el IV cuadrante
    modificada = np.copy(leer_imagen('images/orapple.jpg', cv2.IMREAD_COLOR))
    altura, anchura, _ = modificada.shape
    modificar_pixels(modificada, [(50,y) for y in range(anchura)] + [(x,30) for x in range(altura)]
        + [(x,y) for x in range(int(altura/2),altura) for y in range(int(anchura/2),anchura)], color)
    pintaI('Quitando una fila, una columna y IV cuadrante', modificada)
    input("Pulsa 'Enter' para continuar\n")

    # EJERCICIO 5: Una función que sea capaz de representar varias imágenes con sus títulos en una misma ventana.
    # Usar las imágenes del fichero images en ficheropy.
    print("--- EJERCICIO 5 - REPRESENTAR IMÁGENES CON TÍTULOS ---")
    imprimir_imagenes_titulos([leer_imagen('images/dave.jpg', 0), leer_imagen('images/logoOpenCV.jpg', 1),
           leer_imagen('images/messi.jpg', 0), leer_imagen('images/orapple.jpg', 1)], ['Dave', 'Logo OpenCV', 'Messi', 'Orapple'], 1, 4)

if __name__ == "__main__":
	main()
