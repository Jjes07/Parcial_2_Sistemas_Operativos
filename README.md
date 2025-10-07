# Parcial 2 - Sistemas Operativos - Reto 2

## Equipo
- Juan José Escobar Saldarriaga
- Samuel Llano Madrigal

## Entorno de Desarrollo
- Visual Studio Code (Conexión a Linux con WSL:Ubuntu)
- Lenguaje de Programación: C

## Plataforma de Edición de Imágenes Concurrente en C

### Descripción:
Este programa es una plataforma interactiva de edición de imágenes desarrollada en C.  
Permite aplicar transformaciones y filtros a imágenes (PNG o JPG) utilizando procesamiento concurrente con hilos POSIX (`pthread`).

Cada operación trabaja sobre matrices de píxeles en memoria y aplica técnicas como interpolación bilineal, convolución y transformaciones geométricas.

### Cómo ejecutar la aplicación

Requisitos

- Compilador GCC (con soporte para `pthread` y `math.h`)
- Archivos de cabecera stb_image.h y stb_image_write.h
- Sistema operativo tipo Linux

Compilación
Desde la terminal, dentro de la carpeta del proyecto, compila con:

`gcc -o img img_base.c -I./include -pthread -lm`

Esto genera el ejecutable img.

Ejecuta el programa indicando la ruta de una imagen existente:

`./img /home/usuario/ruta/imagen.png`

### Funciones del Programa
La aplicación funciona de forma interactiva, guiando al usuario paso a paso por un menú de opciones.

0. **Salir**

   Finaliza la aplicación liberando toda la memoria asignada.

 1. **Cargar imagen**
   
    Permite seleccionar o reemplazar la imagen actual.
    Si el archivo es válido (PNG o JPG), la aplicación la carga en memoria y muestra sus dimensiones y número de canales (grises o RGB).

2. **Mostrar matriz de píxeles**
   
    Muestra en consola los valores RGB o de intensidad de los primeros píxeles de la imagen.
    Sirve para visualizar el contenido en forma matricial y verificar que la carga fue correcta.

3. **Guardar como PNG**
   
    Guarda la imagen procesada en un nuevo archivo.
    El programa pedirá un nombre y generará el archivo PNG en el directorio actual.
    Ideal para exportar el resultado después de aplicar filtros o transformaciones.

4. **Ajustar brillo**
   
    Permite aclarar o oscurecer la imagen.
    El usuario ingresa un valor positivo (para aumentar brillo) o negativo (para reducirlo).
    El ajuste se realiza concurrentemente, procesando distintas partes de la imagen en paralelo.

5. **Aplicar convolución (Blur Gaussiano)**
    
    Aplica un desenfoque suave tipo “blur”.
    El usuario elige el tamaño del kernel (3x3 o 5x5) y un valor de sigma (nivel de suavizado).
    Internamente, la imagen se recorre con convolución gaussiana paralelizada por filas.

6. **Detectar bordes (Sobel)**
    
    Detecta los contornos de la imagen mediante el operador Sobel.
    Produce una imagen en escala de grises que resalta los cambios de intensidad.
    El cálculo se divide entre varios hilos para mejorar la velocidad.

7. **Rotar imagen**
    
    Rota la imagen en el ángulo que el usuario indique (por ejemplo 90°, 180°, 45°, etc.).
    La rotación se realiza usando transformaciones geométricas y interpolación bilineal,
    de modo que los píxeles se suavizan al reubicarse.
    La imagen resultante puede cambiar de tamaño automáticamente según el ángulo.

8. **Escalar imagen (Resize)**
    
    Permite redimensionar la imagen a un nuevo ancho y alto.
    El usuario ingresa los valores deseados y el programa ajusta los píxeles usando interpolación bilineal.
    La operación se divide por filas entre múltiples hilos, aprovechando todos los núcleos disponibles.
