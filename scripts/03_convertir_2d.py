"""
03_convertir_2d.py
Convierte los DICOM 2D «For Presentation» (salida del script 02) a PNG listos
para ResNet-50 / DenseNet-121 / EfficientNet-B4, siguiendo la guía:

  1. Lee el DICOM y extrae la matriz de píxeles (pydicom)
  2. Rescale (pendiente/intercepto) + VOI LUT / ventana del propio DICOM
     (respetando VOILUTFunction: LINEAR, LINEAR_EXACT o SIGMOID; tu equipo GE usa SIGMOID)
  3. Corrección fotométrica (MONOCHROME1 -> fondo negro, tejido brillante)
  4. Normalización a [0,1] con la ventana FIJA del DICOM (NO min-max por imagen)
     y escalado a 8 bits (o 16 con --bits 16)
  5. Recorte del fondo negro (caja del tejido mamario, con margen)
  6. Volteo horizontal: todas las mamas con la pared torácica a la IZQUIERDA
     (se decide con PatientOrientation (0020,0020); si falta, con la lateralidad)
  7. Redimensionado SIN deformar (conserva proporción) + relleno negro a ALTO x ANCHO
  8. 1 canal -> 3 canales (solo si usas --rgb; ver nota abajo)

Uso:
    pip install pydicom numpy opencv-python
    python 03_convertir_2d.py "C:\\imagenes_2d" "C:\\png_2d"
    python 03_convertir_2d.py "C:\\imagenes_2d" "C:\\png_2d" --alto 1024 --ancho 768 --rgb

Nota 1->3 canales: guardar 3 canales idénticos solo triplica el espacio. Es más
barato guardar 1 canal y replicar al cargar (en PyTorch: Grayscale(3) o
.repeat(3,1,1)); el resultado para la red es exactamente el mismo.
"""
import argparse
import csv
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# 1) Lectura DICOM
# ----------------------------------------------------------------------------
def _primero(v, defecto=None):
    """Primer valor numérico de una etiqueta (puede traer 3 ventanas)."""
    if v is None or v == "":
        return defecto
    try:
        if v.__class__.__name__ == "MultiValue" or isinstance(v, (list, tuple)):
            v = v[0]
        return float(v)
    except (TypeError, ValueError, IndexError):
        return defecto


def _texto_multi(v):
    """['P','L'] -> 'P\\L'  (formato DICOM con barra invertida)."""
    if v is None or v == "":
        return ""
    if v.__class__.__name__ == "MultiValue" or isinstance(v, (list, tuple)):
        return "\\".join(str(x) for x in v)
    return str(v)


def leer_dicom(ruta):
    """Devuelve (matriz_de_pixeles, metadatos)."""
    import pydicom
    ds = pydicom.dcmread(str(ruta))
    arr = ds.pixel_array
    meta = {
        "bits_stored": int(ds.get("BitsStored", 16)),
        "photometric": str(ds.get("PhotometricInterpretation", "MONOCHROME2")),
        "slope": _primero(ds.get("RescaleSlope"), 1.0),
        "intercept": _primero(ds.get("RescaleIntercept"), 0.0),
        "wc": _primero(ds.get("WindowCenter")),
        "ww": _primero(ds.get("WindowWidth")),
        "func": str(ds.get("VOILUTFunction", "LINEAR") or "LINEAR").upper(),
        "orientacion": _texto_multi(ds.get("PatientOrientation")),
        "lado": str(ds.get("ImageLaterality", "") or ds.get("Laterality", "")).upper(),
        "vista": str(ds.get("ViewPosition", "")).upper(),
    }
    return arr, meta


# ----------------------------------------------------------------------------
# 2-4) Ventana VOI (DICOM PS3.3 C.11.2.1.2 y C.11.2.1.3) -> valores en [0,1]
# ----------------------------------------------------------------------------
def aplicar_ventana(x, c, w, funcion="LINEAR"):
    if funcion == "SIGMOID":                      # y = 1 / (1 + exp(-4 (x-c) / w))
        return 1.0 / (1.0 + np.exp(np.clip(-4.0 * (x - c) / w, -60, 60)))
    if funcion == "LINEAR_EXACT":
        return np.clip((x - c) / w + 0.5, 0.0, 1.0)
    # LINEAR (por defecto)
    w1 = max(w - 1.0, 1e-6)
    return np.clip((x - (c - 0.5)) / w1 + 0.5, 0.0, 1.0)


def a_unidad(arr, meta):
    """Pasos 2-4: matriz cruda -> (y en [0,1] con tejido brillante, lin, info)."""
    x = arr.astype(np.float64) * meta["slope"] + meta["intercept"]
    maximo = float(2 ** meta["bits_stored"] - 1)
    if meta["wc"] is not None and meta["ww"] is not None and meta["ww"] >= 1:
        y = aplicar_ventana(x, meta["wc"], meta["ww"], meta["func"])
        origen = f"DICOM ({meta['func']})"
    else:                                          # sin ventana: todo el rango de bits
        y = np.clip((x - (x.min() if x.min() < 0 else 0)) / max(maximo, 1.0), 0, 1)
        origen = "rango completo (el DICOM no trae ventana)"
    lin = x.copy()
    if meta["photometric"] == "MONOCHROME1":       # tumores/tejido denso salen oscuros
        y = 1.0 - y
        lin = maximo - lin
    return y, lin, origen


# ----------------------------------------------------------------------------
# 5) Recorte del fondo negro
# ----------------------------------------------------------------------------
def caja_mama(lin, margen=0.05, frac=0.03):
    """Caja (x0,y0,x1,y1) del mayor componente conectado por encima del fondo.
    El fondo se estima en el borde de la imagen y el umbral queda justo encima."""
    h, w = lin.shape
    b = max(2, int(0.01 * min(h, w)))
    borde = np.concatenate([lin[:b].ravel(), lin[-b:].ravel(), lin[:, :b].ravel(), lin[:, -b:].ravel()])
    fondo = float(np.median(borde))
    alto = float(np.percentile(lin, 99.5))
    umbral = fondo + frac * max(alto - fondo, 1.0)
    mascara = (cv2.GaussianBlur(lin.astype(np.float32), (0, 0), 3) > umbral).astype(np.uint8)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    if n <= 1:
        return 0, 0, w, h
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x0, y0 = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
    bw, bh = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
    mx, my = int(margen * bw), int(margen * bh)
    return max(0, x0 - mx), max(0, y0 - my), min(w, x0 + bw + mx), min(h, y0 + bh + my)


# ----------------------------------------------------------------------------
# 6) Volteo: pared torácica siempre a la IZQUIERDA
# ----------------------------------------------------------------------------
def debe_voltear(meta):
    """PatientOrientation (0020,0020): el 1er valor es hacia dónde avanzan las
    columnas. 'P' = hacia posterior => pared torácica a la DERECHA => voltear.
    'A' => pared torácica a la izquierda => no voltear. Sin etiqueta: lateralidad."""
    o = meta.get("orientacion", "").upper().split("\\")[0][:1]
    if o == "P":
        return True, "orientacion"
    if o == "A":
        return False, "orientacion"
    if meta.get("lado") == "R":
        return True, "lateralidad"
    if meta.get("lado") == "L":
        return False, "lateralidad"
    return False, "sin_dato"


# ----------------------------------------------------------------------------
# 7) Redimensionar sin deformar + relleno negro
# ----------------------------------------------------------------------------
def ajustar_tamano(img, alto, ancho):
    h, w = img.shape
    esc = min(alto / h, ancho / w)
    nh, nw = max(1, round(h * esc)), max(1, round(w * esc))
    interp = cv2.INTER_AREA if esc < 1 else cv2.INTER_LINEAR
    r = cv2.resize(img, (nw, nh), interpolation=interp)      # cv2: (ancho, alto)
    lienzo = np.zeros((alto, ancho), dtype=img.dtype)
    lienzo[:nh, :nw] = r                                      # anclado arriba-izquierda (pared torácica)
    return lienzo, esc


# ----------------------------------------------------------------------------
# Pipeline completo sobre una matriz ya leída
# ----------------------------------------------------------------------------
def preprocesar(arr, meta, alto=1024, ancho=768, bits=8, recortar=True, rgb=False, control=False):
    y, lin, origen = a_unidad(arr, meta)
    h0, w0 = y.shape
    x0, y0, x1, y1 = caja_mama(lin) if recortar else (0, 0, w0, h0)
    ctrl = None
    if control:                                           # imagen completa con la caja verde del recorte
        e = 800.0 / max(h0, w0)
        peq = cv2.resize(np.rint(y * 255).astype(np.uint8), None, fx=e, fy=e, interpolation=cv2.INTER_AREA)
        ctrl = cv2.cvtColor(peq, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(ctrl, (int(x0 * e), int(y0 * e)), (int(x1 * e) - 1, int(y1 * e) - 1), (0, 255, 0), 2)
    y = y[y0:y1, x0:x1]
    volteo, por = debe_voltear(meta)
    if volteo:
        y = y[:, ::-1]
    # controles de calidad
    qc = []
    if np.median(np.concatenate([y[0], y[-1], y[:, -1]])) > 0.5:
        qc.append("fondo_claro")                          # intensidad invertida
    cols = y.sum(axis=0)
    if cols.sum() > 0 and (np.arange(len(cols)) * cols).sum() / cols.sum() / len(cols) > 0.55:
        qc.append("pared_toracica_dudosa")               # tras voltear debería quedar a la izquierda
    y = np.ascontiguousarray(y)
    maxv = 255 if bits == 8 else 65535
    img = np.rint(y * maxv).astype(np.uint8 if bits == 8 else np.uint16)
    img, esc = ajustar_tamano(img, alto, ancho)
    if rgb:                                               # paso 8: 1 canal -> 3 canales
        img = np.stack([img, img, img], axis=-1)
    info = {"filas_orig": h0, "cols_orig": w0, "caja": f"{x0},{y0},{x1},{y1}",
            "volteo": int(volteo), "volteo_segun": por, "ventana_origen": origen,
            "voi_function": meta["func"], "wc": meta["wc"], "ww": meta["ww"],
            "escala": round(esc, 4), "qc": ";".join(qc), "_control": ctrl}
    return img, info


def lado_vista_nombre(stem):
    m = re.search(r"_(R|L)_(CC|MLO)(?:_rep\d+)?$", stem)
    return (m.group(1), m.group(2)) if m else ("", "")


def asignar_particiones(pacientes_por_clase, semilla, k=5):
    """Reparte POR PACIENTE (nunca por imagen) y estratificado por clase:
    particion = train 70% / val 15% / test 15%;  fold = 0..k-1 para validación cruzada."""
    rng = random.Random(semilla)
    part, fold = {}, {}
    for clase in sorted(pacientes_por_clase):
        pacs = sorted(pacientes_por_clase[clase])
        rng.shuffle(pacs)
        n_tr, n_va = round(0.70 * len(pacs)), round(0.15 * len(pacs))
        for i, p in enumerate(pacs):
            part[p] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
            fold[p] = i % k
    return part, fold


def construir_dataset(filas, salida, semilla):
    """dataset_final.csv listo para PyTorch + resumen de clases y desbalance."""
    ok = [f for f in filas if not f.get("error")]
    por_pac = {}
    for f in ok:
        por_pac.setdefault(f["paciente"], []).append(f)
    completos = {p: v for p, v in por_pac.items() if len(v) == 4}
    descartados = len(por_pac) - len(completos)
    if descartados:
        print(f"AVISO: {descartados} pacientes salen del dataset por tener menos de 4 PNG convertidas.")
    pac_clase = {p: v[0]["clase"] for p, v in completos.items()}
    por_clase = {}
    for p, c in pac_clase.items():
        por_clase.setdefault(c, []).append(p)
    if len(por_clase) < 2:
        print("AVISO: solo hay una clase; no se genera dataset_final.csv"); return
    part, fold = asignar_particiones(por_clase, semilla)
    filas_ds = []
    for p, v in sorted(completos.items()):
        for f in sorted(v, key=lambda r: (r["lado"], r["vista"])):
            filas_ds.append({"paciente": p, "birads_carpeta": f["birads_carpeta"], "clase": f["clase"],
                             "etiqueta": 1 if f["clase"] == "alta_sospecha" else 0,
                             "lado": f["lado"], "vista": f["vista"], "ruta_png": f["png"],
                             "particion": part[p], "fold": fold[p], "qc": f.get("qc", "")})
    with open(salida / "dataset_final.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas_ds[0].keys()))
        w.writeheader()
        w.writerows(filas_ds)

    print("\nDATASET FINAL (etiqueta 0 = benigna, 1 = alta sospecha)")
    print(f"{'Partición':<11}{'Pac. benigna':>13}{'Pac. alta':>11}{'Imágenes':>10}")
    for pt in ("train", "val", "test"):
        b = sum(1 for p, c in pac_clase.items() if part[p] == pt and c == "benigna")
        a = sum(1 for p, c in pac_clase.items() if part[p] == pt and c == "alta_sospecha")
        print(f"{pt:<11}{b:>13}{a:>11}{(a + b) * 4:>10}")
    nb, na = len(por_clase.get("benigna", [])), len(por_clase.get("alta_sospecha", []))
    mayor, menor = max(nb, na), min(nb, na)
    print(f"Total: {nb} benigna / {na} alta sospecha  (proporción {mayor / max(menor, 1):.2f} : 1)")
    if menor and mayor / menor > 1.5:
        pos_weight = nb / na
        print(f"DESBALANCE: la clase minoritaria es {'alta sospecha' if na < nb else 'benigna'}.")
        print(f"  - Usa pesos de clase (BCEWithLogitsLoss pos_weight ≈ {pos_weight:.2f}) o WeightedRandomSampler")
        print("  - NO uses accuracy como métrica principal: reporta sensibilidad, especificidad, AUC y AUPRC")
        print("  - Hazlo solo en train; val y test se dejan con su proporción real")
    if min(nb, na) < 30:
        print("AVISO: menos de 30 pacientes en una clase; usa validación cruzada con la columna 'fold'.")


def registrar_entorno(salida, args):
    """CLAIM 2024 ítems 9 y 23: versiones de software y parámetros usados (reproducibilidad)."""
    def ver(nombre):
        try:
            return __import__(nombre).__version__
        except Exception:  # noqa: BLE001
            return "no instalado"
    lineas = [f"python {sys.version.split()[0]}", f"pydicom {ver('pydicom')}",
              f"numpy {ver('numpy')}", f"opencv {cv2.__version__}", "", "Parámetros:"]
    lineas += [f"  {k} = {v}" for k, v in sorted(vars(args).items())]
    (salida / "entorno.txt").write_text("\n".join(lineas), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("entrada", help="carpeta de salida del script 02 (BIRADS_x/paciente/*.dcm)")
    ap.add_argument("salida")
    ap.add_argument("--alto", type=int, default=1024)
    ap.add_argument("--ancho", type=int, default=768)
    ap.add_argument("--bits", type=int, choices=(8, 16), default=8)
    ap.add_argument("--rgb", action="store_true", help="guardar 3 canales idénticos")
    ap.add_argument("--sin-recorte", action="store_true")
    ap.add_argument("--control", type=int, default=50,
                    help="guarda N imágenes al azar con la caja del recorte dibujada (0 = ninguna)")
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()

    entrada, salida = Path(args.entrada), Path(args.salida)
    archivos = sorted(entrada.glob("BIRADS_*/*/*.dcm"))
    if not archivos:
        print("No hay DICOM en", entrada, "(se esperaba BIRADS_x/paciente/*.dcm)")
        sys.exit(1)
    salida.mkdir(parents=True, exist_ok=True)
    elegidas = set(random.Random(args.semilla).sample(range(len(archivos)), min(args.control, len(archivos))))
    (salida / "_control_recorte").mkdir(exist_ok=True) if elegidas else None

    filas, errores, avisos = [], 0, 0
    for i, ruta in enumerate(archivos, 1):
        paciente, carpeta_b = ruta.parent.name, ruta.parent.parent.name
        n = re.match(r"BIRADS_(\d)", carpeta_b)
        clase = "benigna" if n and n.group(1) == "2" else "alta_sospecha"
        fila = {"paciente": paciente, "birads_carpeta": carpeta_b, "clase": clase,
                "archivo_dicom": ruta.name}
        try:
            arr, meta = leer_dicom(ruta)
            img, info = preprocesar(arr, meta, args.alto, args.ancho, args.bits,
                                    not args.sin_recorte, args.rgb, control=(i - 1) in elegidas)
            ctrl = info.pop("_control")
            destino = salida / carpeta_b / paciente
            destino.mkdir(parents=True, exist_ok=True)
            png = destino / (ruta.stem + ".png")
            if not cv2.imwrite(str(png), img):
                raise OSError("cv2.imwrite no pudo guardar " + str(png))
            if ctrl is not None:
                cv2.imwrite(str(salida / "_control_recorte" / (ruta.stem + "_control.jpg")), ctrl)
            lado, vista = meta["lado"], meta["vista"]
            if not lado or not vista:
                lado, vista = lado_vista_nombre(ruta.stem)
            fila.update({"lado": lado, "vista": vista,
                         "png": f"{carpeta_b}/{paciente}/{png.name}", **info, "error": ""})
            avisos += int(bool(info["qc"]))
        except Exception as e:  # noqa: BLE001
            errores += 1
            fila.update({"error": f"{type(e).__name__}: {e}"})
        filas.append(fila)
        if i % 100 == 0 or i == len(archivos):
            print(f"{i}/{len(archivos)} convertidas | errores: {errores} | con aviso QC: {avisos}")

    campos = list({k: None for f in filas for k in f})
    with open(salida / "indice_conversion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    construir_dataset(filas, salida, args.semilla)
    registrar_entorno(salida, args)
    print(f"\nListo. PNG en {salida}")
    print(f"Salida: {args.alto}x{args.ancho}, {args.bits} bits, {'3 canales' if args.rgb else '1 canal'}")
    if elegidas:
        print(f"Revisa a ojo la carpeta _control_recorte (caja verde = recorte) antes de entrenar.")
    print("Revisa en indice_conversion.csv las filas con error o con qc (fondo_claro, pared_toracica_dudosa).")


if __name__ == "__main__":
    main()
