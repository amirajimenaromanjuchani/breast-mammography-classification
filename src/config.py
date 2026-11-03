"""
Configuración central del proyecto (tesis: comparación de CNN en mamografía 2D).

REGLA: los hiperparámetros y los parámetros de aumento de datos se FIJAN ANTES de
evaluar en los pliegues de prueba. Los valores marcados «provisional» se confirman
con un piloto en el pliegue de validación y luego se congelan (no se cambian al ver
resultados de prueba).
"""
from pathlib import Path

# Carpeta del proyecto en Google Drive (AJUSTAR a tu ruta real)
DRIVE_ROOT = Path("/content/drive/MyDrive/tesis-cnn/mammo-tesis")

CONFIG = {
    "paths": {
        "png_root": DRIVE_ROOT / "png_2d",
        "dataset_csv": DRIVE_ROOT / "png_2d" / "dataset_final_v1.csv",  # PRIVADO (no va a GitHub)
        "splits_publico": "splits/splits_publico.csv",                  # público (en el repositorio)
        "checkpoints": DRIVE_ROOT / "resultados" / "checkpoints",
        "tablas": DRIVE_ROOT / "resultados" / "tablas",
        "figuras": DRIVE_ROOT / "resultados" / "figuras",
    },
    "clases": {"benigna": 0, "alta_sospecha": 1},
    "birads": {"benigna": [2], "alta_sospecha": [4, 5], "excluidos": [0, 1, 3, 6]},
    "imagen": {
        "alto": 1024, "ancho": 768,
        "bits_en_disco": 8, "canales_en_disco": 1, "canales_modelo": 3,   # se replica al cargar
        "normalizacion": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},  # ImageNet (fija)
    },
    "validacion_cruzada": {
        "n_pliegues": 5,
        "semillas": [42, 43, 44],
        "semillas_iniciales": [42],
        "columna_pliegue": "fold_seed{semilla}",
        # ronda k: prueba = k; validación = (k + 1) % 5; entrenamiento = el resto
        "unidad": "paciente",
    },
    "modelos": ["resnet50", "densenet121", "efficientnet_b4"],
    "entrenamiento": {                       # provisional
        "epocas_max": 100, "paciencia": 10,
        "optimizador": "adam", "lr": 1e-4, "weight_decay": 1e-5,
        "batch_size": 8,                     # provisional: depende de la memoria de la T4 con 1024x768
    },
    "aumento_datos": {                       # SOLO entrenamiento, al vuelo (propuesta; congelar antes de entrenar)
        "rotacion_grados": 10,
        "traslacion_max_frac": 0.05,
        "escala": [0.9, 1.1],
        "brillo_contraste_frac": 0.10,
        "volteo_horizontal": False,          # se estandarizó la orientación (pared torácica a la izquierda)
        "volteo_vertical": False,
        "smote": False, "cutmix": False, "mixup": False,
    },
    "desbalance": {"estrategia": "sobremuestreo_por_clase", "solo_entrenamiento": True},
    "evaluacion": {
        "nivel": "paciente",                 # promedio de las probabilidades de sus 4 vistas
        "metricas": ["auc_roc", "sensibilidad", "especificidad", "f1"],
        "bootstrap_n": 1000, "bootstrap_unidad": "paciente",
        "umbral": "se elige en el pliegue de validación, nunca en el de prueba",
    },
}
