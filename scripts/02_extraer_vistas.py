"""
02_extraer_vistas.py
Extrae SOLO lo que importa y dice con cuántas imágenes se queda:
  - estudios BI-RADS 2, 4 (4a/4b/4c) y 5      (se excluyen 0, 1, 3, 6)
  - únicamente imágenes 2D «For Presentation» de las 4 vistas (R-CC, L-CC, R-MLO, L-MLO)
  - estudios con las 4 vistas completas
  - si una vista se repitió, se queda la toma MÁS RECIENTE
  - un solo estudio por código de paciente

Los DICOM originales NUNCA se modifican: solo se copian.

Uso:
    python 02_extraer_vistas.py "C:\\MedSoft\\filestore" "C:\\imagenes_2d" --solo-indice
    python 02_extraer_vistas.py "C:\\MedSoft\\filestore" "C:\\imagenes_2d"
    python 02_extraer_vistas.py "C:\\MedSoft\\filestore" "C:\\imagenes_2d" --tambien "E:/Mamografias"

Resultado:
    SALIDA/BIRADS_2/B2-0282/B2-0282_R_CC.dcm  (y L_CC, R_MLO, L_MLO)
    SALIDA/BIRADS_4/..., BIRADS_5/...
    SALIDA/indice_imagenes_2d.csv   (todas las 2D candidatas; usar=1 las conservadas)
    SALIDA/resumen_estudios.csv     (una fila por estudio y motivo de exclusión)
"""
import argparse
import csv
import shutil
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from mamo_comun import (CLASE_ALTA, CLASE_BENIGNA, COMBINACIONES, a_entero,
                        clasificar_birads, codigo_paciente, encontrar_estudios,
                        es_2d_presentacion, lado_y_vista, orden_cronologico)


@lru_cache(maxsize=None)
def indice_dcm(carpeta):
    """{nombre_sin_extension: ruta} de todos los .dcm bajo la carpeta del estudio."""
    return {p.stem: p for p in Path(carpeta).rglob("*.dcm")}


def alerta_anonimizacion(d):
    alertas = []
    nombre = d.get("00100010", "").upper()
    if nombre and "ANONYMOUS" not in nombre:
        alertas.append("nombre")
    for tag, txt in (("00100030", "nacimiento"), ("00101010", "edad"),
                     ("00080020", "fecha_estudio"), ("00080080", "institucion"),
                     ("00080090", "medico")):
        if d.get(tag, ""):
            alertas.append(txt)
    return ";".join(alertas)


def evaluar_estudio(carpeta, uid, inst):
    """Decide si el estudio entra y qué imágenes conserva (sin copiar nada)."""
    cod = codigo_paciente(inst)
    birads, clase, carpeta_birads = clasificar_birads(cod)
    r = {"paciente": cod, "uid_estudio": uid, "birads": birads, "clase": clase,
         "carpeta_birads": carpeta_birads, "carpeta_origen": carpeta,
         "motivo": "", "imagenes": [], "n_2d": 0, "n_repetidas": 0}
    if clase not in (CLASE_BENIGNA, CLASE_ALTA):
        r["motivo"] = f"birads_excluido ({birads or 'desconocido'})"
        return r
    candidatas = [d for d in inst if es_2d_presentacion(d)]
    por_vista = defaultdict(list)
    for d in candidatas:
        lv = lado_y_vista(d)
        if lv:
            por_vista[lv].append(d)
    r["n_2d"] = sum(len(v) for v in por_vista.values())
    if not por_vista:
        r["motivo"] = "sin_2d_for_presentation"
        return r
    faltan = [f"{l}-{v}" for l, v in COMBINACIONES if (l, v) not in por_vista]
    for (lado, vista), grupo in por_vista.items():
        grupo.sort(key=orden_cronologico)
        for pos, d in enumerate(grupo):
            es_ultima = pos == len(grupo) - 1
            r["imagenes"].append({"lado": lado, "vista": vista, "d": d,
                                  "usar": int(es_ultima), "rep": pos})
            r["n_repetidas"] += int(not es_ultima)
    if faltan:
        r["motivo"] = "vistas_incompletas (faltan " + ",".join(faltan) + ")"
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("entrada")
    ap.add_argument("salida")
    ap.add_argument("--tambien", action="append", default=[])
    ap.add_argument("--solo-indice", action="store_true",
                    help="no copia nada; solo genera tablas y conteos")
    args = ap.parse_args()

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    estudios, ilegibles = encontrar_estudios([Path(args.entrada)] + [Path(p) for p in args.tambien])
    if not estudios:
        print("No se encontró ningún estudio.")
        return

    res = [evaluar_estudio(*e) for e in estudios]

    # un solo estudio por código de paciente (los XML no traen fecha: está anonimizada)
    por_codigo = defaultdict(list)
    for r in res:
        if not r["motivo"]:
            por_codigo[r["paciente"]].append(r)
    for cod, lista in por_codigo.items():
        if len(lista) > 1:
            lista.sort(key=lambda r: max(orden_cronologico(i["d"]) for i in r["imagenes"]), reverse=True)
            print(f"AVISO: {cod} tiene {len(lista)} estudios; se conserva el de mayor "
                  f"SeriesNumber. REVISA a mano cuál es el más reciente.")
            for r in lista[1:]:
                r["motivo"] = "duplicado_codigo_paciente"

    # un estudio con algún DICOM que no se encuentra en disco NO entra (evita datos incompletos)
    for r in res:
        r["n_faltantes"] = 0
        if not r["motivo"]:
            idx = indice_dcm(str(r["carpeta_origen"]))
            r["n_faltantes"] = sum(1 for im in r["imagenes"]
                                   if im["usar"] and idx.get(im["d"]["_uid"]) is None)
            if r["n_faltantes"]:
                r["motivo"] = f"archivo_no_encontrado ({r['n_faltantes']} de 4)"

    # ---- copia de archivos y construcción de índices
    filas, resumen = [], []
    for r in res:
        entra = not r["motivo"]
        n_usar = 0
        for im in r["imagenes"]:
            d, usar = im["d"], im["usar"] and entra
            base = f"{r['paciente']}_{im['lado']}_{im['vista']}"
            nombre = base if im["usar"] else f"{base}_rep{im['rep'] + 1}"
            origen = indice_dcm(str(r["carpeta_origen"])).get(d["_uid"])
            alerta = alerta_anonimizacion(d)
            if usar:
                if not args.solo_indice:
                    destino = salida / r["carpeta_birads"] / r["paciente"]
                    destino.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(origen, destino / f"{nombre}.dcm")
                n_usar += 1
            filas.append({
                "paciente": r["paciente"], "birads": r["birads"], "clase": r["clase"],
                "lado": im["lado"], "vista": im["vista"], "archivo": f"{nombre}.dcm",
                "ruta": f"{r['carpeta_birads']}/{r['paciente']}/{nombre}.dcm",
                "usar": int(usar), "series_number": d.get("00200011", ""),
                "instance_number": d.get("00200013", ""),
                "patient_orientation": d.get("00200020", ""),
                "filas": d.get("00280010", ""), "columnas": d.get("00280011", ""),
                "bits_stored": d.get("00280101", ""), "photometric": d.get("00280004", ""),
                "voi_function": d.get("00281056", ""),
                "window_center": d.get("00281050", ""), "window_width": d.get("00281051", ""),
                "archivo_origen": d["_uid"] + ".dcm", "motivo_estudio": r["motivo"],
                "alerta": alerta})
        resumen.append({"paciente": r["paciente"], "uid_estudio": r["uid_estudio"],
                        "birads": r["birads"], "clase": r["clase"],
                        "incluido": int(entra), "motivo_exclusion": r["motivo"],
                        "imagenes_2d_candidatas": r["n_2d"], "imagenes_conservadas": n_usar,
                        "repeticiones_descartadas": r["n_repetidas"] if entra else 0})

    for nombre, datos in (("indice_imagenes_2d.csv", filas), ("resumen_estudios.csv", resumen)):
        if datos:
            with open(salida / nombre, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(datos[0].keys()))
                w.writeheader()
                w.writerows(datos)

    # ---- TABLAS
    inc = [x for x in resumen if x["incluido"]]
    print(f"\nEstudios leídos: {len(res)}   |   estudios que se quedan: {len(inc)}")

    print("\nTABLA 1. Muestra final por clase")
    print(f"{'Clase':<16}{'BI-RADS':<9}{'Estudios':>9}{'Imágenes':>10}{'Repet. descartadas':>20}")
    tot_e = tot_i = tot_r = 0
    esperados = {CLASE_BENIGNA: ["2"], CLASE_ALTA: ["4", "5"]}
    for clase in (CLASE_BENIGNA, CLASE_ALTA):
        sub = [x for x in inc if x["clase"] == clase]
        for b in esperados[clase]:
            s_ = [x for x in sub if x["birads"][:1] == b]
            e, i, rp = len(s_), sum(x["imagenes_conservadas"] for x in s_), sum(x["repeticiones_descartadas"] for x in s_)
            print(f"{clase:<16}{b:<9}{e:>9}{i:>10}{rp:>20}")
        e, i, rp = len(sub), sum(x["imagenes_conservadas"] for x in sub), sum(x["repeticiones_descartadas"] for x in sub)
        print(f"{'  subtotal':<25}{e:>9}{i:>10}{rp:>20}")
        tot_e, tot_i, tot_r = tot_e + e, tot_i + i, tot_r + rp
    print(f"{'TOTAL':<25}{tot_e:>9}{tot_i:>10}{tot_r:>20}")

    print("\nTABLA 2. Estudios descartados y por qué")
    motivos = Counter(x["motivo_exclusion"].split(" (")[0] for x in resumen if not x["incluido"])
    for m, n in motivos.most_common():
        print(f"  {m:<32}{n:>8}")
    if not motivos:
        print("  (ninguno)")
    # ---- CLAIM 2024, ítem 13: protocolo de adquisición de las imágenes conservadas
    prot = Counter()
    for r in res:
        if r["motivo"]:
            continue
        for im in r["imagenes"]:
            if im["usar"]:
                d = im["d"]
                prot[(d.get("00080070", ""), d.get("00081090", ""), d.get("00180060", ""),
                      d.get("00181164", ""), d.get("00181191", ""), d.get("00187050", ""),
                      d.get("00280010", ""), d.get("00280011", ""), d.get("00280101", ""))] += 1
    campos_p = ["fabricante", "equipo_software", "kVp", "pixel_spacing_mm", "anodo", "filtro",
                "filas", "columnas", "bits_stored", "n_imagenes"]
    if prot:
        with open(salida / "protocolo_adquisicion.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(campos_p)
            for k, n in prot.most_common():
                w.writerow(list(k) + [n])
        print("\nPROTOCOLO DE ADQUISICIÓN (CLAIM ítem 13) -> protocolo_adquisicion.csv")
        for k, n in prot.most_common(5):
            print(f"  {n:>6} img | {k[0]} | {k[1]} | {k[2]} kVp | {k[3]} mm | {k[4]}/{k[5]} | {k[6]}x{k[7]} | {k[8]} bits")

    faltantes = sum(r["n_faltantes"] for r in res)
    est_falt = sum(1 for r in res if r["n_faltantes"])
    print(f"\nArchivos .dcm NO encontrados en disco: {faltantes} (en {est_falt} estudios, ya excluidos)")
    if faltantes:
        print("  -> Revisa cómo se llaman los .dcm: se esperan como <SOPInstanceUID>.dcm")
    print(f"Imágenes conservadas con alerta de anonimización: "
          f"{sum(1 for f in filas if f['usar'] and f['alerta'])}")
    if ilegibles:
        print(f"AVISO: {len(ilegibles)} XML ilegibles (ejemplo: {ilegibles[0]})")
    if args.solo_indice:
        print("(modo --solo-indice: no se copió ningún archivo)")


if __name__ == "__main__":
    main()
