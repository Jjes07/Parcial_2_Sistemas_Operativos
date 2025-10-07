// Programa de procesamiento de imágenes en C para principiantes en Linux.
// QUÉ: Procesa imágenes PNG (escala de grises o RGB) usando matrices, con soporte
// para carga, visualización, guardado y ajuste de brillo concurrente.
// CÓMO: Usa stb_image.h para cargar PNG y stb_image_write.h para guardar PNG,
// con hilos POSIX (pthread) para el procesamiento paralelo del brillo.
// POR QUÉ: Diseñado para enseñar manejo de matrices, concurrencia y gestión de
// memoria en C, manteniendo simplicidad y robustez para principiantes.
// Dependencias: Descarga stb_image.h y stb_image_write.h desde
// https://github.com/nothings/stb
//   wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
//   wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
//
// Compilar: gcc -o img img_base.c -pthread -lm
// Ejecutar: ./img [ruta_imagen.png]

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <math.h>

// QUÉ: Incluir bibliotecas stb para cargar y guardar imágenes PNG.
// CÓMO: stb_image.h lee PNG/JPG a memoria; stb_image_write.h escribe PNG.
// POR QUÉ: Son bibliotecas de un solo archivo, simples y sin dependencias externas.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// QUÉ: Estructura para almacenar la imagen (ancho, alto, canales, píxeles).
// CÓMO: Usa matriz 3D para píxeles (alto x ancho x canales), donde canales es
// 1 (grises) o 3 (RGB). Píxeles son unsigned char (0-255).
// POR QUÉ: Permite manejar tanto grises como color, con memoria dinámica para
// flexibilidad y evitar desperdicio.
typedef struct {
    int ancho;           // Ancho de la imagen en píxeles
    int alto;            // Alto de la imagen en píxeles
    int canales;         // 1 (escala de grises) o 3 (RGB)
    unsigned char*** pixeles; // Matriz 3D: [alto][ancho][canales]
} ImagenInfo;

// QUÉ: Liberar memoria asignada para la imagen.
// CÓMO: Libera cada fila y canal de la matriz 3D, luego el arreglo de filas y
// reinicia la estructura.
// POR QUÉ: Evita fugas de memoria, esencial en C para manejar recursos manualmente.
void liberarImagen(ImagenInfo* info) {
    if (info->pixeles) {
        for (int y = 0; y < info->alto; y++) {
            for (int x = 0; x < info->ancho; x++) {
                free(info->pixeles[y][x]); // Liberar canales por píxel
            }
            free(info->pixeles[y]); // Liberar fila
        }
        free(info->pixeles); // Liberar arreglo de filas
        info->pixeles = NULL;
    }
    info->ancho = 0;
    info->alto = 0;
    info->canales = 0;
}

// QUÉ: Cargar una imagen PNG desde un archivo.
// CÓMO: Usa stbi_load para leer el archivo, detecta canales (1 o 3), y convierte
// los datos a una matriz 3D (alto x ancho x canales).
// POR QUÉ: La matriz 3D es intuitiva para principiantes y permite procesar
// píxeles y canales individualmente.
int cargarImagen(const char* ruta, ImagenInfo* info) {
    int canales;
    // QUÉ: Cargar imagen con formato original (0 canales = usar formato nativo).
    // CÓMO: stbi_load lee el archivo y llena ancho, alto y canales.
    // POR QUÉ: Respetar el formato original asegura que grises o RGB se mantengan.
    unsigned char* datos = stbi_load(ruta, &info->ancho, &info->alto, &canales, 0);
    if (!datos) {
        fprintf(stderr, "Error al cargar imagen: %s\n", ruta);
        return 0;
    }
    info->canales = (canales == 1 || canales == 3) ? canales : 1; // Forzar 1 o 3

    // QUÉ: Asignar memoria para matriz 3D.
    // CÓMO: Asignar alto filas, luego ancho columnas por fila, luego canales por píxel.
    // POR QUÉ: Estructura clara y flexible para grises (1 canal) o RGB (3 canales).
    info->pixeles = (unsigned char***)malloc(info->alto * sizeof(unsigned char**));
    if (!info->pixeles) {
        fprintf(stderr, "Error de memoria al asignar filas\n");
        stbi_image_free(datos);
        return 0;
    }
    for (int y = 0; y < info->alto; y++) {
        info->pixeles[y] = (unsigned char**)malloc(info->ancho * sizeof(unsigned char*));
        if (!info->pixeles[y]) {
            fprintf(stderr, "Error de memoria al asignar columnas\n");
            liberarImagen(info);
            stbi_image_free(datos);
            return 0;
        }
        for (int x = 0; x < info->ancho; x++) {
            info->pixeles[y][x] = (unsigned char*)malloc(info->canales * sizeof(unsigned char));
            if (!info->pixeles[y][x]) {
                fprintf(stderr, "Error de memoria al asignar canales\n");
                liberarImagen(info);
                stbi_image_free(datos);
                return 0;
            }
            // Copiar píxeles a matriz 3D
            for (int c = 0; c < info->canales; c++) {
                info->pixeles[y][x][c] = datos[(y * info->ancho + x) * info->canales + c];
            }
        }
    }

    stbi_image_free(datos); // Liberar buffer de stb
    printf("Imagen cargada: %dx%d, %d canales (%s)\n", info->ancho, info->alto,
           info->canales, info->canales == 1 ? "grises" : "RGB");
    return 1;
}

// QUÉ: Mostrar la matriz de píxeles (primeras 10 filas).
// CÓMO: Imprime los valores de los píxeles, agrupando canales por píxel (grises o RGB).
// POR QUÉ: Ayuda a visualizar la matriz para entender la estructura de datos.
void mostrarMatriz(const ImagenInfo* info) {
    if (!info->pixeles) {
        printf("No hay imagen cargada.\n");
        return;
    }
    printf("Matriz de la imagen (primeras 10 filas):\n");
    for (int y = 0; y < info->alto && y < 10; y++) {
        for (int x = 0; x < info->ancho; x++) {
            if (info->canales == 1) {
                printf("%3u ", info->pixeles[y][x][0]); // Escala de grises
            } else {
                printf("(%3u,%3u,%3u) ", info->pixeles[y][x][0], info->pixeles[y][x][1],
                       info->pixeles[y][x][2]); // RGB
            }
        }
        printf("\n");
    }
    if (info->alto > 10) {
        printf("... (más filas)\n");
    }
}

// QUÉ: Guardar la matriz como PNG (grises o RGB).
// CÓMO: Aplana la matriz 3D a 1D y usa stbi_write_png con el número de canales correcto.
// POR QUÉ: Respeta el formato original (grises o RGB) para consistencia.
int guardarPNG(const ImagenInfo* info, const char* rutaSalida) {
    if (!info->pixeles) {
        fprintf(stderr, "No hay imagen para guardar.\n");
        return 0;
    }

    // QUÉ: Aplanar matriz 3D a 1D para stb.
    // CÓMO: Copia píxeles en orden [y][x][c] a un arreglo plano.
    // POR QUÉ: stb_write_png requiere datos contiguos.
    unsigned char* datos1D = (unsigned char*)malloc(info->ancho * info->alto * info->canales);
    if (!datos1D) {
        fprintf(stderr, "Error de memoria al aplanar imagen\n");
        return 0;
    }
    for (int y = 0; y < info->alto; y++) {
        for (int x = 0; x < info->ancho; x++) {
            for (int c = 0; c < info->canales; c++) {
                datos1D[(y * info->ancho + x) * info->canales + c] = info->pixeles[y][x][c];
            }
        }
    }

    // QUÉ: Guardar como PNG.
    // CÓMO: Usa stbi_write_png con los canales de la imagen original.
    // POR QUÉ: Mantiene el formato (grises o RGB) de la entrada.
    int resultado = stbi_write_png(rutaSalida, info->ancho, info->alto, info->canales,
                                   datos1D, info->ancho * info->canales);
    free(datos1D);
    if (resultado) {
        printf("Imagen guardada en: %s (%s)\n", rutaSalida,
               info->canales == 1 ? "grises" : "RGB");
        return 1;
    } else {
        fprintf(stderr, "Error al guardar PNG: %s\n", rutaSalida);
        return 0;
    }
}

// QUÉ: Estructura para pasar datos al hilo de ajuste de brillo.
// CÓMO: Contiene matriz, rango de filas, ancho, canales y delta de brillo.
// POR QUÉ: Los hilos necesitan datos específicos para procesar en paralelo.
typedef struct {
    unsigned char*** pixeles;
    int inicio;
    int fin;
    int ancho;
    int canales;
    int delta;
} BrilloArgs;

// QUÉ: Ajustar brillo en un rango de filas (para hilos).
// CÓMO: Suma delta a cada canal de cada píxel, con clamp entre 0-255.
// POR QUÉ: Procesa píxeles en paralelo para demostrar concurrencia.
void* ajustarBrilloHilo(void* args) {
    BrilloArgs* bArgs = (BrilloArgs*)args;
    for (int y = bArgs->inicio; y < bArgs->fin; y++) {
        for (int x = 0; x < bArgs->ancho; x++) {
            for (int c = 0; c < bArgs->canales; c++) {
                int nuevoValor = bArgs->pixeles[y][x][c] + bArgs->delta;
                bArgs->pixeles[y][x][c] = (unsigned char)(nuevoValor < 0 ? 0 :
                                                          (nuevoValor > 255 ? 255 : nuevoValor));
            }
        }
    }
    return NULL;
}

// QUÉ: Ajustar brillo de la imagen usando múltiples hilos.
// CÓMO: Divide las filas entre 2 hilos, pasa argumentos y espera con join.
// POR QUÉ: Usa concurrencia para acelerar el procesamiento y enseñar hilos.
void ajustarBrilloConcurrente(ImagenInfo* info, int delta) {
    if (!info->pixeles) {
        printf("No hay imagen cargada.\n");
        return;
    }

    const int numHilos = 2; // QUÉ: Número fijo de hilos para simplicidad.
    pthread_t hilos[numHilos];
    BrilloArgs args[numHilos];
    int filasPorHilo = (int)ceil((double)info->alto / numHilos);

    // QUÉ: Configurar y lanzar hilos.
    // CÓMO: Asigna rangos de filas a cada hilo y pasa datos.
    // POR QUÉ: Divide el trabajo para procesar en paralelo.
    for (int i = 0; i < numHilos; i++) {
        args[i].pixeles = info->pixeles;
        args[i].inicio = i * filasPorHilo;
        args[i].fin = (i + 1) * filasPorHilo < info->alto ? (i + 1) * filasPorHilo : info->alto;
        args[i].ancho = info->ancho;
        args[i].canales = info->canales;
        args[i].delta = delta;
        if (pthread_create(&hilos[i], NULL, ajustarBrilloHilo, &args[i]) != 0) {
            fprintf(stderr, "Error al crear hilo %d\n", i);
            return;
        }
    }

    // QUÉ: Esperar a que los hilos terminen.
    // CÓMO: Usa pthread_join para sincronizar.
    // POR QUÉ: Garantiza que todos los píxeles se procesen antes de continuar.
    for (int i = 0; i < numHilos; i++) {
        pthread_join(hilos[i], NULL);
    }
    printf("Brillo ajustado concurrentemente con %d hilos (%s).\n", numHilos,
           info->canales == 1 ? "grises" : "RGB");
}

// ======================================== FUNCIÓN 1: Convolución (e.g., Filtro de Desenfoque Gaussiano) ================================= 

typedef struct {
    unsigned char*** entrada;   // imagen original
    unsigned char*** salida;    // imagen resultante
    int inicio;
    int fin;
    int ancho;
    int alto;
    int canales;
    float** kernel;
    int tamKernel;
} ConvolucionArgs;

float** generarKernelGaussiano(int tam, float sigma) {
    int centro = tam / 2;
    float** kernel = (float**)malloc(tam * sizeof(float*));
    for (int i = 0; i < tam; i++) {
        kernel[i] = (float*)malloc(tam * sizeof(float));
    }

    float suma = 0.0;
    for (int y = 0; y < tam; y++) {
        for (int x = 0; x < tam; x++) {
            int dx = x - centro;
            int dy = y - centro;
            float valor = exp(-(dx*dx + dy*dy) / (2 * sigma * sigma));
            kernel[y][x] = valor;
            suma += valor;
        }
    }
    // Normalizar kernel
    for (int y = 0; y < tam; y++) {
        for (int x = 0; x < tam; x++) {
            kernel[y][x] /= suma;
        }
    }
    return kernel;
}

void* convolucionHilo(void* args) {
    ConvolucionArgs* cArgs = (ConvolucionArgs*)args;
    int k = cArgs->tamKernel;
    int centro = k / 2;

    for (int y = cArgs->inicio; y < cArgs->fin; y++) {
        for (int x = 0; x < cArgs->ancho; x++) {
            for (int c = 0; c < cArgs->canales; c++) {
                float suma = 0.0;
                for (int ky = 0; ky < k; ky++) {
                    for (int kx = 0; kx < k; kx++) {
                        int ny = y + ky - centro;
                        int nx = x + kx - centro;
                        // Manejo de bordes replicando píxeles
                        if (ny < 0) ny = 0;
                        if (ny >= cArgs->alto) ny = cArgs->alto - 1;
                        if (nx < 0) nx = 0;
                        if (nx >= cArgs->ancho) nx = cArgs->ancho - 1;

                        suma += cArgs->entrada[ny][nx][c] * cArgs->kernel[ky][kx];
                    }
                }
                int valor = (int)roundf(suma);
                if (valor < 0) valor = 0;
                if (valor > 255) valor = 255;
                cArgs->salida[y][x][c] = (unsigned char)valor;
            }
        }
    }
    return NULL;
}

void aplicarConvolucionConcurrente(ImagenInfo* info, int tamKernel, float sigma) {
    if (!info->pixeles) {
        printf("No hay imagen cargada.\n");
        return;
    }

    // Validaciones y parámetros por defecto
    if (tamKernel <= 1) tamKernel = 3;
    if (tamKernel % 2 == 0) tamKernel++; // forzar impar
    if (tamKernel < 3) tamKernel = 3;
    if (sigma <= 0.0f) sigma = 1.0f;

    // Guardar dimensiones originales (las necesitaremos después de liberar)
    int ancho = info->ancho;
    int alto  = info->alto;
    int canales = info->canales;

    // Generar kernel y comprobar
    float** kernel = generarKernelGaussiano(tamKernel, sigma);
    if (!kernel) {
        fprintf(stderr, "Error: no se pudo generar kernel Gaussiano\n");
        return;
    }

    // Crear matriz de salida (igual dimensiones y canales)
    unsigned char*** salida = (unsigned char***)malloc(alto * sizeof(unsigned char**));
    if (!salida) {
        fprintf(stderr, "Error de memoria al asignar salida (filas)\n");
        // liberar kernel
        for (int i = 0; i < tamKernel; ++i) free(kernel[i]);
        free(kernel);
        return;
    }
    int ok = 1;
    for (int y = 0; y < alto; y++) {
        salida[y] = (unsigned char**)malloc(ancho * sizeof(unsigned char*));
        if (!salida[y]) { ok = 0; break; }
        for (int x = 0; x < ancho; x++) {
            salida[y][x] = (unsigned char*)malloc(canales * sizeof(unsigned char));
            if (!salida[y][x]) { ok = 0; break; }
            // inicializamos (opcional)
            for (int c = 0; c < canales; ++c) salida[y][x][c] = 0;
        }
        if (!ok) break;
    }
    if (!ok) {
        fprintf(stderr, "Error de memoria al asignar salida (columnas/canales)\n");
        // liberar lo que se haya asignado
        for (int yy = 0; yy < alto; yy++) {
            if (!salida[yy]) break;
            for (int xx = 0; xx < ancho; xx++) {
                if (salida[yy][xx]) free(salida[yy][xx]);
            }
            free(salida[yy]);
        }
        free(salida);
        // liberar kernel
        for (int i = 0; i < tamKernel; ++i) free(kernel[i]);
        free(kernel);
        return;
    }

    // Preparar hilos (número configurable; al menos 1)
    int numHilos = 4;
    if (alto < numHilos) numHilos = alto > 0 ? alto : 1;
    pthread_t *hilos = (pthread_t*)malloc(sizeof(pthread_t) * numHilos);
    ConvolucionArgs *args = (ConvolucionArgs*)malloc(sizeof(ConvolucionArgs) * numHilos);
    if (!hilos || !args) {
        fprintf(stderr, "Error de memoria para estructuras de hilos\n");
        // limpiar salida y kernel
        for (int y = 0; y < alto; y++) {
            for (int x = 0; x < ancho; x++) free(salida[y][x]);
            free(salida[y]);
        }
        free(salida);
        for (int i = 0; i < tamKernel; ++i) free(kernel[i]);
        free(kernel);
        if (hilos) free(hilos);
        if (args) free(args);
        return;
    }

    int filasPorHilo = (int)ceil((double)alto / numHilos);
    int hcreados = 0;
    for (int i = 0; i < numHilos; i++) {
        args[i].entrada = info->pixeles;
        args[i].salida = salida;
        args[i].inicio = i * filasPorHilo;
        args[i].fin = (i + 1) * filasPorHilo < alto ? (i + 1) * filasPorHilo : alto;
        args[i].ancho = ancho;
        args[i].alto = alto;
        args[i].canales = canales;
        args[i].kernel = kernel;
        args[i].tamKernel = tamKernel;

        if (pthread_create(&hilos[i], NULL, convolucionHilo, &args[i]) != 0) {
            fprintf(stderr, "Error al crear hilo %d\n", i);
            // ajustamos numHilos reales y salimos del bucle (haremos join de los creados)
            numHilos = i;
            break;
        } else {
            hcreados++;
        }
    }

    // Esperar a los hilos creados
    for (int i = 0; i < hcreados; i++) {
        pthread_join(hilos[i], NULL);
    }

    // Liberar kernel (ya no es necesario)
    for (int i = 0; i < tamKernel; ++i) free(kernel[i]);
    free(kernel);

    // Ahora reemplazamos la imagen original por la salida:
    // 1) Liberar memoria antigua (esto pone ancho/alto/canales a 0)
    liberarImagen(info);

    // 2) Asignar la nueva matriz y restaurar dimensiones
    info->pixeles = salida;
    info->ancho = ancho;
    info->alto = alto;
    info->canales = canales;

    // Liberar estructuras auxiliares de hilos
    free(hilos);
    free(args);

    printf("Convolución aplicada con kernel %dx%d y sigma=%.2f usando %d hilos.\n",
           tamKernel, tamKernel, sigma, hcreados > 0 ? hcreados : 1);
}

// ======================================== FUNCIÓN 2: Rotación  =========================================================================

typedef struct {
    unsigned char ***entrada;
    unsigned char ***salida;
    int inicio;
    int fin;
    int ancho_src;
    int alto_src;
    int ancho_dst;
    int alto_dst;
    int canales;
    double cos_t;
    double sin_t;
    double cx_src;
    double cy_src;
    double cx_dst;
    double cy_dst;
} RotacionArgs;

/* Clamp helper */
static inline int clamp_int(int v, int a, int b) {
    return v < a ? a : (v > b ? b : v);
}

/* Obtener valor de canal con replicación de bordes */
static unsigned char obtenerPixel(const unsigned char*** img, int y, int x, int c, int alto, int ancho) {
    int yy = clamp_int(y, 0, alto - 1);
    int xx = clamp_int(x, 0, ancho - 1);
    return img[yy][xx][c];
}

/* Hilo que procesa un rango de filas de la imagen destino */
static void* rotacionHilo(void* arg) {
    RotacionArgs* r = (RotacionArgs*)arg;
    for (int dy = r->inicio; dy < r->fin; dy++) {
        for (int dx = 0; dx < r->ancho_dst; dx++) {
            /* coordenadas relativas al centro de la dst */
            double rx = (double)dx - r->cx_dst;
            double ry = (double)dy - r->cy_dst;

            /* Mapeo inverso: aplicamos rotación -theta sobre las coordenadas destino
               para encontrar la coordenada en la imagen fuente */
            double sx_rel =  r->cos_t * rx + r->sin_t * ry;  // cos(+θ) y +sin para R(-θ) = [cos, sin; -sin, cos]
            double sy_rel = -r->sin_t * rx + r->cos_t * ry;

            double sx = sx_rel + r->cx_src;
            double sy = sy_rel + r->cy_src;

            /* Interpolación bilineal */
            int x0 = (int)floor(sx);
            int y0 = (int)floor(sy);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            double wx = sx - x0;
            double wy = sy - y0;

            for (int c = 0; c < r->canales; c++) {
                /* Obtener los 4 vecinos (con clamp) */
                double v00 = (double) obtenerPixel((const unsigned char***)r->entrada, y0, x0, c, r->alto_src, r->ancho_src);
                double v10 = (double) obtenerPixel((const unsigned char***)r->entrada, y0, x1, c, r->alto_src, r->ancho_src);
                double v01 = (double) obtenerPixel((const unsigned char***)r->entrada, y1, x0, c, r->alto_src, r->ancho_src);
                double v11 = (double) obtenerPixel((const unsigned char***)r->entrada, y1, x1, c, r->alto_src, r->ancho_src);

                /* Bilinear */
                double v0 = v00 * (1.0 - wx) + v10 * wx;
                double v1 = v01 * (1.0 - wx) + v11 * wx;
                double v  = v0  * (1.0 - wy) + v1  * wy;

                int vi = (int) round(v);
                if (vi < 0) vi = 0;
                if (vi > 255) vi = 255;

                r->salida[dy][dx][c] = (unsigned char)vi;
            }
        }
    }
    return NULL;
}

/* Función pública: rotar imagen 'info' en grados (ángulo positivo = rotación en sentido antihorario)
   Se crea una nueva matriz para la imagen rotada y se libera la antigua. Usa interpolación bilineal.
*/
void rotarImagen(ImagenInfo* info, float angulo_grados) {
    if (!info || !info->pixeles) {
        printf("No hay imagen cargada.\n");
        return;
    }

    /* Normalizar ángulo y calcular cos/sin de theta (en radianes).
       Para el mapeo inverso usamos R(-θ) por lo que precomputamos cosθ y sinθ. */
    double theta = (double)angulo_grados * M_PI / 180.0;
    double cos_t = cos(theta);
    double sin_t = sin(theta);

    int ancho_src = info->ancho;
    int alto_src  = info->alto;
    int canales   = info->canales;

    /* Calcular las posiciones de las 4 esquinas respecto al centro de la fuente */
    double cx_src = (ancho_src - 1) / 2.0;
    double cy_src = (alto_src  - 1) / 2.0;

    double corners_x[4], corners_y[4];
    int corners_ix[4] = {0, ancho_src - 1, ancho_src - 1, 0};
    int corners_iy[4] = {0, 0, alto_src - 1, alto_src - 1};

    double min_x = 1e9, min_y = 1e9, max_x = -1e9, max_y = -1e9;

    for (int i = 0; i < 4; i++) {
        double rx = corners_ix[i] - cx_src;
        double ry = corners_iy[i] - cy_src;
        /* rotar punto */
        double rrx = cos_t * rx - sin_t * ry;
        double rry = sin_t * rx + cos_t * ry;
        double fx = rrx + cx_src;
        double fy = rry + cy_src;
        if (fx < min_x) min_x = fx;
        if (fx > max_x) max_x = fx;
        if (fy < min_y) min_y = fy;
        if (fy > max_y) max_y = fy;
    }

    /* dimensiones destino (redondear y asegurar al menos 1) */
    int ancho_dst = (int)ceil(max_x - min_x + 1.0);
    int alto_dst  = (int)ceil(max_y - min_y + 1.0);
    if (ancho_dst < 1) ancho_dst = 1;
    if (alto_dst  < 1) alto_dst = 1;

    /* centros de src y dst */
    double cx_dst = (ancho_dst - 1) / 2.0;
    double cy_dst = (alto_dst  - 1) / 2.0;

    /* Asignar matriz de salida [alto_dst][ancho_dst][canales] */
    unsigned char*** salida = (unsigned char***)malloc(alto_dst * sizeof(unsigned char**));
    if (!salida) {
        fprintf(stderr, "Error de memoria al asignar salida (filas) en rotación\n");
        return;
    }
    int ok = 1;
    for (int y = 0; y < alto_dst; y++) {
        salida[y] = (unsigned char**)malloc(ancho_dst * sizeof(unsigned char*));
        if (!salida[y]) { ok = 0; break; }
        for (int x = 0; x < ancho_dst; x++) {
            salida[y][x] = (unsigned char*)malloc(canales * sizeof(unsigned char));
            if (!salida[y][x]) { ok = 0; break; }
            /* Opcional: inicializar a 0 (fondo negro) */
            for (int c = 0; c < canales; c++) salida[y][x][c] = 0;
        }
        if (!ok) break;
    }
    if (!ok) {
        fprintf(stderr, "Error de memoria al asignar salida en rotación (columnas/canales)\n");
        /* liberar lo asignado */
        for (int yy = 0; yy < alto_dst; yy++) {
            if (!salida[yy]) break;
            for (int xx = 0; xx < ancho_dst; xx++) {
                if (salida[yy][xx]) free(salida[yy][xx]);
            }
            free(salida[yy]);
        }
        free(salida);
        return;
    }

    /* Preparar concurrencia: dividir por filas */
    int numHilos = 4;
    if (alto_dst < numHilos) numHilos = alto_dst > 0 ? alto_dst : 1;
    pthread_t *hilos = (pthread_t*)malloc(sizeof(pthread_t) * numHilos);
    RotacionArgs *args = (RotacionArgs*)malloc(sizeof(RotacionArgs) * numHilos);
    if (!hilos || !args) {
        fprintf(stderr, "Error de memoria para estructuras de hilos en rotación\n");
        for (int y = 0; y < alto_dst; y++) {
            for (int x = 0; x < ancho_dst; x++) free(salida[y][x]);
            free(salida[y]);
        }
        free(salida);
        if (hilos) free(hilos);
        if (args) free(args);
        return;
    }

    int filasPorHilo = (int)ceil((double)alto_dst / numHilos);
    int hcreados = 0;
    for (int i = 0; i < numHilos; i++) {
        args[i].entrada = info->pixeles;
        args[i].salida = salida;
        args[i].inicio = i * filasPorHilo;
        args[i].fin = (i + 1) * filasPorHilo < alto_dst ? (i + 1) * filasPorHilo : alto_dst;
        args[i].ancho_src = ancho_src;
        args[i].alto_src  = alto_src;
        args[i].ancho_dst = ancho_dst;
        args[i].alto_dst  = alto_dst;
        args[i].canales   = canales;
        args[i].cos_t     = cos_t;
        args[i].sin_t     = sin_t;
        args[i].cx_src    = cx_src;
        args[i].cy_src    = cy_src;
        args[i].cx_dst    = cx_dst;
        args[i].cy_dst    = cy_dst;

        if (pthread_create(&hilos[i], NULL, rotacionHilo, &args[i]) != 0) {
            fprintf(stderr, "Error al crear hilo %d en rotación\n", i);
            numHilos = i;
            break;
        } else {
            hcreados++;
        }
    }

    for (int i = 0; i < hcreados; i++) pthread_join(hilos[i], NULL);

    /* Reemplazar imagen original por la salida: liberar la antigua y setear la nueva */
    liberarImagen(info);

    info->pixeles = salida;
    info->ancho   = ancho_dst;
    info->alto    = alto_dst;
    info->canales = canales;

    free(hilos);
    free(args);

    printf("Rotación aplicada: %.2f grados — nueva dimensión: %dx%d — hilos usados: %d\n",
           angulo_grados, ancho_dst, alto_dst, hcreados > 0 ? hcreados : 1);
}


// ======================================== FUNCIÓN 3: Detección de Bordes (e.g., Operador Sobel) ========================================

typedef struct {
    unsigned char ***entrada;
    unsigned char ***salida;
    int inicio;
    int fin;
    int ancho;
    int alto;
    int canales;
} SobelArgs;

static const int GX[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

static const int GY[3][3] = {
    {-1,-2,-1},
    { 0, 0, 0},
    { 1, 2, 1}
};

static void* sobelHilo(void* arg) {
    SobelArgs* s = (SobelArgs*)arg;

    for (int y = s->inicio; y < s->fin; y++) {
        if (y == 0 || y == s->alto - 1) continue;
        for (int x = 1; x < s->ancho - 1; x++) {
            int pixelGray[3][3];

            // Ventana 3x3 convertida a gris
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;

                    if (s->canales == 1) {
                        pixelGray[ky + 1][kx + 1] = s->entrada[ny][nx][0];
                    } else {
                        int r = s->entrada[ny][nx][0];
                        int g = s->entrada[ny][nx][1];
                        int b = s->entrada[ny][nx][2];
                        pixelGray[ky + 1][kx + 1] = (r + g + b) / 3;
                    }
                }
            }

            int gx = 0, gy = 0;
            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    gx += GX[ky][kx] * pixelGray[ky][kx];
                    gy += GY[ky][kx] * pixelGray[ky][kx];
                }
            }

            int mag = (int)round(sqrt((double)(gx*gx + gy*gy)));
            if (mag > 255) mag = 255;

            s->salida[y][x][0] = (unsigned char)mag;
        }
    }
    return NULL;
}

void aplicarSobelConcurrente(ImagenInfo* info) {
    if (!info || !info->pixeles) {
        printf("No hay imagen cargada.\n");
        return;
    }

    // Reservar matriz 3D de salida: [alto][ancho][1]
    unsigned char*** salida = malloc(info->alto * sizeof(unsigned char**));
    for (int y = 0; y < info->alto; y++) {
        salida[y] = malloc(info->ancho * sizeof(unsigned char*));
        for (int x = 0; x < info->ancho; x++) {
            salida[y][x] = calloc(1, sizeof(unsigned char)); // inicializa en 0
        }
    }

    const int numHilos = 4;
    pthread_t hilos[numHilos];
    SobelArgs args[numHilos];

    int filasPorHilo = (info->alto + numHilos - 1) / numHilos;

    for (int i = 0; i < numHilos; i++) {
        args[i].entrada = info->pixeles;
        args[i].salida  = salida;
        args[i].inicio  = i * filasPorHilo;
        args[i].fin     = (i + 1) * filasPorHilo;
        if (args[i].fin > info->alto) args[i].fin = info->alto;
        args[i].ancho   = info->ancho;
        args[i].alto    = info->alto;
        args[i].canales = info->canales;

        pthread_create(&hilos[i], NULL, sobelHilo, &args[i]);
    }

    for (int i = 0; i < numHilos; i++) {
        pthread_join(hilos[i], NULL);
    }

    // Sustituir imagen por la salida en grises
    int ancho = info->ancho;
    int alto  = info->alto;

    liberarImagen(info);  // deja ancho/alto/canales en 0

    info->pixeles = salida;
    info->ancho   = ancho;   // ← restaurar
    info->alto    = alto;    // ← restaurar
    info->canales = 1;       // imagen en escala de grises

    printf("Detección de bordes (Sobel) finalizada con %d hilos.\n", numHilos);

}

// ======================================== FUNCIÓN 4: Escalado de Imagen (Resize) =======================================================
typedef struct {
    unsigned char ***entrada;
    unsigned char ***salida;
    int inicio;
    int fin;
    int ancho_src;
    int alto_src;
    int ancho_dst;
    int alto_dst;
    int canales;
    double escalaX;
    double escalaY;
} EscaladoArgs;

/* Hilo que procesa un rango de filas en la imagen destino */
static void* escaladoHilo(void* arg) {
    EscaladoArgs* e = (EscaladoArgs*)arg;

    for (int y_dst = e->inicio; y_dst < e->fin; y_dst++) {
        double sy = (y_dst + 0.5) * e->escalaY - 0.5;
        int y0 = (int)floor(sy);
        int y1 = y0 + 1;
        double wy = sy - y0;

        for (int x_dst = 0; x_dst < e->ancho_dst; x_dst++) {
            double sx = (x_dst + 0.5) * e->escalaX - 0.5;
            int x0 = (int)floor(sx);
            int x1 = x0 + 1;
            double wx = sx - x0;

            for (int c = 0; c < e->canales; c++) {
                unsigned char p00 = obtenerPixel((const unsigned char***)e->entrada, y0, x0, c, e->alto_src, e->ancho_src);
                unsigned char p10 = obtenerPixel((const unsigned char***)e->entrada, y0, x1, c, e->alto_src, e->ancho_src);
                unsigned char p01 = obtenerPixel((const unsigned char***)e->entrada, y1, x0, c, e->alto_src, e->ancho_src);
                unsigned char p11 = obtenerPixel((const unsigned char***)e->entrada, y1, x1, c, e->alto_src, e->ancho_src);

                double v0 = p00 * (1.0 - wx) + p10 * wx;
                double v1 = p01 * (1.0 - wx) + p11 * wx;
                double valor = v0 * (1.0 - wy) + v1 * wy;

                int vi = (int)round(valor);
                if (vi < 0) vi = 0;
                if (vi > 255) vi = 255;
                e->salida[y_dst][x_dst][c] = (unsigned char)vi;
            }
        }
    }
    return NULL;
}

/* Escalar imagen con interpolación bilineal y concurrencia */
void escalarImagen(ImagenInfo* info, int nuevoAncho, int nuevoAlto) {
    if (!info || !info->pixeles) {
        printf("No hay imagen cargada.\n");
        return;
    }
    if (nuevoAncho <= 0 || nuevoAlto <= 0) {
        printf("Dimensiones inválidas.\n");
        return;
    }

    int ancho_src = info->ancho;
    int alto_src = info->alto;
    int canales = info->canales;

    double escalaX = (double)ancho_src / nuevoAncho;
    double escalaY = (double)alto_src / nuevoAlto;

    // Asignar nueva matriz destino
    unsigned char*** salida = (unsigned char***)malloc(nuevoAlto * sizeof(unsigned char**));
    if (!salida) {
        fprintf(stderr, "Error de memoria al asignar filas (resize)\n");
        return;
    }
    int ok = 1;
    for (int y = 0; y < nuevoAlto; y++) {
        salida[y] = (unsigned char**)malloc(nuevoAncho * sizeof(unsigned char*));
        if (!salida[y]) { ok = 0; break; }
        for (int x = 0; x < nuevoAncho; x++) {
            salida[y][x] = (unsigned char*)malloc(canales * sizeof(unsigned char));
            if (!salida[y][x]) { ok = 0; break; }
        }
        if (!ok) break;
    }
    if (!ok) {
        fprintf(stderr, "Error de memoria al asignar columnas/canales (resize)\n");
        for (int y = 0; y < nuevoAlto; y++) {
            if (!salida[y]) break;
            for (int x = 0; x < nuevoAncho; x++) free(salida[y][x]);
            free(salida[y]);
        }
        free(salida);
        return;
    }

    // Configurar concurrencia
    int numHilos = 4;
    if (nuevoAlto < numHilos) numHilos = nuevoAlto > 0 ? nuevoAlto : 1;

    pthread_t *hilos = (pthread_t*)malloc(numHilos * sizeof(pthread_t));
    EscaladoArgs *args = (EscaladoArgs*)malloc(numHilos * sizeof(EscaladoArgs));

    int filasPorHilo = (int)ceil((double)nuevoAlto / numHilos);
    int hcreados = 0;

    for (int i = 0; i < numHilos; i++) {
        args[i].entrada = info->pixeles;
        args[i].salida = salida;
        args[i].inicio = i * filasPorHilo;
        args[i].fin = (i + 1) * filasPorHilo < nuevoAlto ? (i + 1) * filasPorHilo : nuevoAlto;
        args[i].ancho_src = ancho_src;
        args[i].alto_src = alto_src;
        args[i].ancho_dst = nuevoAncho;
        args[i].alto_dst = nuevoAlto;
        args[i].canales = canales;
        args[i].escalaX = escalaX;
        args[i].escalaY = escalaY;

        if (pthread_create(&hilos[i], NULL, escaladoHilo, &args[i]) != 0) {
            fprintf(stderr, "Error al crear hilo %d (resize)\n", i);
            numHilos = i;
            break;
        } else {
            hcreados++;
        }
    }

    for (int i = 0; i < hcreados; i++) pthread_join(hilos[i], NULL);

    liberarImagen(info);
    info->pixeles = salida;
    info->ancho = nuevoAncho;
    info->alto = nuevoAlto;
    info->canales = canales;

    free(hilos);
    free(args);

    printf("Escalado aplicado: %dx%d → %dx%d (%.2fx, %.2fy) con %d hilos.\n",
           ancho_src, alto_src, nuevoAncho, nuevoAlto, 1.0 / escalaX, 1.0 / escalaY,
           hcreados > 0 ? hcreados : 1);
}


// QUÉ: Mostrar el menú interactivo.
// CÓMO: Imprime opciones y espera entrada del usuario.
// POR QUÉ: Proporciona una interfaz simple para interactuar con el programa.
void mostrarMenu() {
    printf("\n--- Plataforma de Edición de Imágenes ---\n");
    printf("1. Cargar imagen PNG\n");
    printf("2. Mostrar matriz de píxeles\n");
    printf("3. Guardar como PNG\n");
    printf("4. Ajustar brillo (+/- valor) concurrentemente\n");
    printf("5. Aplicar convolución (blur gaussiano)\n");
    printf("6. Detectar bordes (Sobel)\n");
    printf("7. Rotar imagen (grados)\n");
    printf("8. Escalar imagen (resize)\n");
    printf("0. Salir\n");
    printf("Opción: ");
}

// QUÉ: Función principal que controla el flujo del programa.
// CÓMO: Maneja entrada CLI, ejecuta el menú en bucle y llama funciones según opción.
// POR QUÉ: Centraliza la lógica y asegura limpieza al salir.
int main(int argc, char* argv[]) {
    ImagenInfo imagen = {0, 0, 0, NULL}; // Inicializar estructura
    char ruta[256] = {0}; // Buffer para ruta de archivo

    // QUÉ: Cargar imagen desde CLI si se pasa.
    // CÓMO: Copia argv[1] y llama cargarImagen.
    // POR QUÉ: Permite ejecución directa con ./img imagen.png.
    if (argc > 1) {
        strncpy(ruta, argv[1], sizeof(ruta) - 1);
        if (!cargarImagen(ruta, &imagen)) {
            return EXIT_FAILURE;
        }
    }

    int opcion;
    while (1) {
        mostrarMenu();
        // QUÉ: Leer opción del usuario.
        // CÓMO: Usa scanf y limpia el buffer para evitar bucles infinitos.
        // POR QUÉ: Manejo robusto de entrada evita errores comunes.
        if (scanf("%d", &opcion) != 1) {
            while (getchar() != '\n');
            printf("Entrada inválida.\n");
            continue;
        }
        while (getchar() != '\n'); // Limpiar buffer

        switch (opcion) {
            case 1: { // Cargar imagen
                printf("Ingresa la ruta del archivo PNG: ");
                if (fgets(ruta, sizeof(ruta), stdin) == NULL) {
                    printf("Error al leer ruta.\n");
                    continue;
                }
                ruta[strcspn(ruta, "\n")] = 0; // Eliminar salto de línea
                liberarImagen(&imagen); // Liberar imagen previa
                if (!cargarImagen(ruta, &imagen)) {
                    continue;
                }
                break;
            }
            case 2: // Mostrar matriz
                mostrarMatriz(&imagen);
                break;
            case 3: { // Guardar PNG
                char salida[256];
                printf("Nombre del archivo PNG de salida: ");
                if (fgets(salida, sizeof(salida), stdin) == NULL) {
                    printf("Error al leer ruta.\n");
                    continue;
                }
                salida[strcspn(salida, "\n")] = 0;
                guardarPNG(&imagen, salida);
                break;
            }
            case 4: { // Ajustar brillo
                int delta;
                printf("Valor de ajuste de brillo (+ para más claro, - para más oscuro): ");
                if (scanf("%d", &delta) != 1) {
                    while (getchar() != '\n');
                    printf("Entrada inválida.\n");
                    continue;
                }
                while (getchar() != '\n');
                ajustarBrilloConcurrente(&imagen, delta);
                break;
            }
            case 5: { // Convolución
                int tamKernel;
                float sigma;
                printf("Tamaño del kernel (3 o 5): ");
                scanf("%d", &tamKernel);
                while (getchar() != '\n');
                printf("Valor de sigma (ej: 1.0): ");
                scanf("%f", &sigma);
                while (getchar() != '\n');
                aplicarConvolucionConcurrente(&imagen, tamKernel, sigma);
                break;
            }
            case 6:
                aplicarSobelConcurrente(&imagen);
                break;
            case 7: {
                float ang;
                printf("Ángulo de rotación en grados (positivo = antihorario): ");
                if (scanf("%f", &ang) != 1) {
                    while (getchar() != '\n');
                    printf("Entrada inválida.\n");
                    break;
                }
                while (getchar() != '\n');
                rotarImagen(&imagen, ang);
                break;
            }
            case 8: {
                int nuevoAncho, nuevoAlto;
                printf("Nuevo ancho: ");
                if (scanf("%d", &nuevoAncho) != 1) {
                    while (getchar() != '\n');
                    printf("Entrada inválida.\n");
                    break;
                }
                printf("Nuevo alto: ");
                if (scanf("%d", &nuevoAlto) != 1) {
                    while (getchar() != '\n');
                    printf("Entrada inválida.\n");
                    break;
                }
                while (getchar() != '\n');
                escalarImagen(&imagen, nuevoAncho, nuevoAlto);
                break;
            }
            case 0: // Salir
                liberarImagen(&imagen);
                printf("¡Adiós!\n");
                return EXIT_SUCCESS;
            default:
                printf("Opción inválida.\n");
        }
    }
    liberarImagen(&imagen);
    return EXIT_SUCCESS;
}