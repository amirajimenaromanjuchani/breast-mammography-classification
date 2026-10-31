# Clasificación de mamografías digitales 2D con ResNet-50, DenseNet-121 y EfficientNet-B4

**Tesis de grado** · Amira Jimena Roman Juchani · Universidad Privada del Valle
Carrera de Ingeniería en Ciencia de Datos e Inteligencia de Negocios

> **Estado:** en desarrollo. Este repositorio contiene **código y documentación**.
> **No contiene imágenes, DICOM ni datos de pacientes** (ver «Datos y ética»).

## Objetivo general

Comparar el desempeño de ResNet-50, DenseNet-121 y EfficientNet-B4 en la clasificación de mamografías digitales 2D en **benigna (BI-RADS 2)** y **alta sospecha (BI-RADS 4 y 5)** en el Hospital Univalle Zona Norte de Cochabamba, Bolivia.

## Objetivos específicos

1. Construir la muestra (BI-RADS 2, 4 y 5; se excluyen 0, 1, 3 y 6), anonimizar y preprocesar los estudios DICOM, y documentar el flujo de pacientes según la guía CLAIM 2024.
2. Preparar el conjunto de mamografías 2D «For Presentation» mediante conversión a PNG (1 canal, 8 bits, 1024 × 768 px), normalización con la ventana VOI del DICOM, recorte de la región mamaria y estandarización de la orientación; y definir el aumento de datos, aplicado únicamente al conjunto de entrenamiento.
3. Entrenar ResNet-50, DenseNet-121 y EfficientNet-B4 mediante transferencia de aprendizaje, con hiperparámetros fijados antes de evaluar en los pliegues de prueba, para la clasificación binaria.
4. Comparar el rendimiento de los modelos a nivel de paciente con AUC-ROC (métrica principal), sensibilidad, especificidad y F1, con intervalos de confianza bootstrap por paciente y pruebas de significancia estadística para comparaciones pareadas.
5. Determinar la capacidad diagnóstica del modelo con mejor desempeño, discutir su potencial de integración e identificar las limitaciones del estudio.

## Datos y ética

- 371 estudios (pacientes) y 1484 imágenes: 311 benignas (BI-RADS 2) y 60 de alta sospecha (45 de BI-RADS 4 y 15 de BI-RADS 5). Cada estudio aporta las 4 vistas estándar (R-CC, L-CC, R-MLO, L-MLO).
- Estudio retrospectivo. El hospital autorizó el uso de los datos bajo la condición de anonimización total. Las pacientes se identifican con códigos neutros (`P0001`…) y la tabla de equivalencias se conserva fuera de este repositorio.
- **Los datos no se publican.** Sí se publican las particiones (`splits/splits_publico.csv`: ID neutro, etiqueta y pliegues), de modo que el esquema de validación se puede reproducir y auditar.
- Etiqueta de referencia: categoría BI-RADS del informe radiológico (no confirmación histopatológica) a nivel de estudio.

## Diseño experimental

| Elemento | Decisión |
|---|---|
| Unidad de partición y evaluación | La **paciente**: sus 4 vistas siempre en el mismo pliegue; la probabilidad de la paciente es el promedio de sus 4 vistas |
| Validación | Validación cruzada de 5 pliegues, estratificada por BI-RADS (2, 4, 5). Sin conjunto de prueba fijo (la muestra es pequeña) |
| Cada ronda *k* | Prueba = pliegue *k*; validación (parada temprana) = pliegue (*k*+1) mod 5; entrenamiento = los otros tres |
| Semillas | 42, 43 y 44 (fijadas antes de entrenar); se inicia con 42 |
| Hiperparámetros | Fijados antes de evaluar; el pliegue de prueba no interviene en ninguna decisión de modelado |

## Preprocesamiento (scripts `01`–`03`)

1. Lectura de metadatos desde el XML del estudio y selección de imágenes 2D «For Presentation» (SOP Class `1.2.840.10008.5.1.4.1.1.1.2`); se descartan «For Processing» y tomosíntesis.
2. Si una vista se repitió, se conserva la toma más reciente (orden por `SeriesNumber` y luego `InstanceNumber`).
3. Rescale (pendiente/intercepto) y ventana VOI del DICOM respetando `VOILUTFunction` (los equipos declaran SIGMOID); intensidades en [0, 1].
4. Recorte de la región mamaria (mayor componente conectado, con margen) con imágenes de control para revisión visual.
5. Orientación: pared torácica a la izquierda (según `PatientOrientation`; lateralidad si falta).
6. Redimensionado sin deformar con relleno negro a 1024 × 768; PNG de 1 canal y 8 bits. Los 3 canales se generan al cargar en la red.

## Aumento de datos y desbalance de clases

- Se aplica **solo al entrenamiento** y **al vuelo** (no se guardan imágenes aumentadas).
- Transformaciones leves: rotación pequeña, traslación y escala reducidas, y variaciones de brillo/contraste. Los valores están en `src/config.py` y se congelan antes del entrenamiento.
- **No se usa** volteo horizontal (se estandarizó la orientación), SMOTE ni CutMix/mixup.
- Desbalance (5.2 : 1): sobremuestreo por clase solo en los pliegues de entrenamiento; evaluación con AUC-ROC y métricas por clase.

## Estructura del repositorio (prevista)

```
README.md  LICENSE  requirements.txt  .gitignore
src/        configuración y código de datos, modelos, entrenamiento y evaluación
scripts/    01_contar_muestra, 02_extraer_vistas, 03_convertir_2d, 04_crear_pliegues
splits/     splits_publico.csv
docs/       verificacion_pliegues.txt, CLAIM_2024.md (mapeo ítem por ítem)
notebooks/  cuaderno de Colab (clona el repositorio y monta Drive)
```

## Uso en Google Colab

1. Los datos (PNG y `dataset_final_v2.csv`) están en Google Drive y **no** en este repositorio.
2. En Colab, monta Drive y clona el repositorio. Si es privado, guarda el token de GitHub en los **Secretos de Colab**; nunca lo escribas en el cuaderno.
3. Ajusta `DRIVE_ROOT` en `src/config.py` a tu carpeta.

## Reproducibilidad y CLAIM 2024

Se registran las semillas, los parámetros y las versiones de software (`entorno.txt`). El mapeo de los ítems de la lista CLAIM 2024 a las secciones de la tesis y a este repositorio estará en `docs/CLAIM_2024.md`.

## Limitaciones

Estudio de un solo centro y un solo fabricante de equipos; etiqueta por estudio basada en el informe radiológico (una mama sin hallazgos puede heredar la categoría de la otra); muestra pequeña (60 pacientes de alta sospecha), por lo que los intervalos de confianza serán amplios; sin validación externa.

## Licencia y cita

Ver `LICENSE`. Si usas este código, cita la tesis (referencia pendiente de publicación).

## Referencias

- Varoquaux, G. (2018). Cross-validation failure: small sample sizes lead to large error bars. *NeuroImage, 180*, 68–77.
- Vabalas, A. et al. (2019). Machine learning algorithm validation with a limited sample size. *PLOS ONE, 14*(11), e0224365.
- Garcea, F. et al. (2023). Data augmentation for medical imaging: a systematic literature review. *Computers in Biology and Medicine, 152*, 106391.
- Buda, M., Maki, A., & Mazurowski, M. A. (2018). A systematic study of the class imbalance problem in convolutional neural networks. *Neural Networks, 106*, 249–259.
- Chawla, N. V. et al. (2002). SMOTE: synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357.
- Wu, N. et al. (2020). Deep neural networks improve radiologists' performance in breast cancer screening. *IEEE Transactions on Medical Imaging, 39*(4), 1184–1194.
- Tejani, A. S. et al. (2024). Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 update. *Radiology: Artificial Intelligence*.
