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

## Resultados
| Modelo | Accuracy Test | Observaciones |
|---|---|---|
| Baseline | 0.62 |  Bajo rendimiento, pierde estructura espacial |
| CNN 3x3 | 0.69 | Mejor generalizacion; kernels pequenos capturan patrones locales |
| CNN 5x5 | 0.59 | Peor que 3x3; mas parametros, posible sobreajuste |

## Interpretacion y razonamiento arquitectonico
- La CNN supero al baseline porque conserva estructura espacial, reutiliza filtros (comparticion de pesos) y reduce parametros efectivos frente a `Flatten + Dense`.
- El baseline pierde relaciones locales al aplanar la imagen, por eso generaliza peor en clasificacion visual.
- El sesgo inductivo de convolucion es localidad + equivarianza a traslacion: asume que patrones utiles (bordes, texturas) pueden aparecer en distintas posiciones.
- Con pooling se agrega cierta invariancia a pequeñas traslaciones y ruido local.
- Convolucion no es ideal en datos tabulares sin estructura espacial o en problemas donde la relacion entre variables no es local en una grilla.

## Arquitectura (diagrama simple)
Baseline:
- `Input(64,64,3) -> Flatten -> Dense(128, relu) -> Dense(2, softmax)`

CNN 3x3:
- `Input(64,64,3) -> Conv(32,3x3) -> MaxPool -> Conv(64,3x3) -> MaxPool -> Conv(128,3x3) -> MaxPool -> Flatten -> Dense(128,relu) -> Dense(2,softmax)`

CNN 5x5:
- Igual que la CNN 3x3, cambiando kernels `3x3` por `5x5`.

## Como ejecutar
Asegurate de que el dataset este en `archive/` con estructura de carpetas por clase.
Abre `cats_dogs_cnn_lab.ipynb`.
Ejecuta todas las celdas de arriba hacia abajo.

## SageMaker (entrenamiento y despliegue)
Pasos realizados:
1. Subir dataset a S3 con estructura `archive/cats_set` y `archive/dogs_set`.
2. Lanzar entrenamiento con `TensorFlow Estimator` usando `entry_point='train.py'`.
3. Empaquetar/usar artefactos del modelo y desplegar endpoint.
4. Ejecutar una inferencia de prueba contra el endpoint.

## Archivos
- Notebook principal: `cats_dogs_cnn_lab.ipynb`
- Script de entrenamiento: `train.py`
- Artefactos de inferencia: `model/`
