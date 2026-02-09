# Laboratorio CNN: Cats vs Dogs (Mini Dataset)

## Descripcion del problema
Este laboratorio analiza el impacto de decisiones arquitectonicas en redes convolucionales para clasificacion de imagenes. Se compara un baseline sin convoluciones contra una CNN disenada desde cero y se realiza un experimento controlado sobre el **tamano de kernel**.

## Dataset
- Fuente: Cats and Dogs Mini Dataset (Kaggle)
- Ruta local: `archive/cats_set` y `archive/dogs_set`
- Clases: `cat` y `dog`
- Total de imagenes: **1000** (500 por clase)

Justificacion: es un dataset de imagenes a color con dos clases y un tamano adecuado para entrenar en laptop sin requerir hardware especializado. La estructura espacial de los datos favorece el uso de convoluciones.

## EDA (Exploracion de datos)
En el notebook se realiza una EDA minima:
- Conteo por clase
- Dimensiones y modo de color de imagenes ejemplo
- Visualizacion de muestras por clase
- Preprocesamiento: resize a 64x64 y normalizacion a [0, 1]

## Baseline (Sin Convoluciones)
**Arquitectura:**
- `Input(64x64x3)`
- `Flatten`
- `Dense(128, relu)`
- `Dense(2, softmax)`

**Limitaciones esperadas:**
- Muchos parametros
- No aprovecha estructura espacial
- Riesgo de sobreajuste con dataset pequeno

## CNN Disenada
**Arquitectura:**
- `Conv2D(32, 3x3) + MaxPool`
- `Conv2D(64, 3x3) + MaxPool`
- `Conv2D(128, 3x3) + MaxPool`
- `Flatten`
- `Dense(128, relu)`
- `Dense(2, softmax)`

**Justificacion:**
- Kernels 3x3 capturan patrones locales
- MaxPooling reduce dimensionalidad y aporta invariancia
- La profundidad permite composicion de features

## Experimento Controlado
**Variable controlada:** Tamano de kernel (3x3 vs 5x5)

Se entrenan dos CNN identicas en todo excepto el kernel.

## Resultados (completar al ejecutar)
| Modelo | Accuracy Test | Observaciones |
|---|---|---|
| Baseline | 0.62 |  Bajo rendimiento, pierde estructura espacial |
| CNN 3x3 | 0.69 | Mejor generalizacion; kernels pequenos capturan patrones locales |
| CNN 5x5 | 0.59 | Peor que 3x3; mas parametros, posible sobreajuste |

## Archivos
- Notebook principal: `cats_dogs_cnn_lab.ipynb`
