"""
01_contar_muestra.py
Cuenta CUÁNTOS ESTUDIOS (mamografías) hay por clase BI-RADS en total, antes de
aplicar ningún filtro de vistas. Sirve para conocer el tamaño de la muestra.
Solo lee los XML (segundos), no abre ni copia DICOM.

Uso:
    python 01_contar_muestra.py "C:\\MedSoft\\filestore"
    python 01_contar_muestra.py "C:\\MedSoft\\filestore" --tambien "E:/Mamografias"
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from mamo_comun import (CLASE_ALTA, CLASE_BENIGNA, clasificar_birads,
                        codigo_paciente, encontrar_estudios)

ETIQUETA = {CLASE_BENIGNA: "Benigna (BI-RADS 2)",
            CLASE_ALTA: "Alta sospecha (BI-RADS 4+5)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("entrada")
    ap.add_argument("--tambien", action="append", default=[],
                    help="otra carpeta de entrada (se puede repetir)")
    ap.add_argument("--csv", default="muestra_total_por_clase.csv")
    args = ap.parse_args()

    estudios, ilegibles = encontrar_estudios([Path(args.entrada)] + [Path(p) for p in args.tambien])
    if not estudios:
        print("No se encontró ningún estudio.")
        return

    por_birads = Counter()                 # estudios por BI-RADS exacto (2, 4a, 4, 5, ...)
    por_clase = Counter()                  # estudios por clase del proyecto
    codigos = defaultdict(set)             # códigos distintos por BI-RADS
    studies_por_codigo = Counter()
    for _, uid, inst in estudios:
        cod = codigo_paciente(inst)
        birads, clase, _ = clasificar_birads(cod)
        por_birads[birads or "?"] += 1
        por_clase[clase] += 1
        codigos[birads or "?"].add(cod)
        studies_por_codigo[cod] += 1

    # ---- Tabla 1: por BI-RADS exacto
    print("\nMUESTRA TOTAL (todos los estudios leídos, sin filtrar vistas)")
    print(f"{'BI-RADS':<10}{'Estudios':>10}{'Códigos distintos':>20}  Destino")
    destino = {"2": "BENIGNA", "4": "ALTA SOSPECHA", "5": "ALTA SOSPECHA"}
    filas = []
    for b in sorted(por_birads):
        base = b[:1]
        dest = destino.get(base, "EXCLUIDO")
        print(f"{b:<10}{por_birads[b]:>10}{len(codigos[b]):>20}  {dest}")
        filas.append({"birads": b, "estudios": por_birads[b],
                      "codigos_distintos": len(codigos[b]), "destino": dest})
    print(f"{'TOTAL':<10}{sum(por_birads.values()):>10}")

    # ---- Tabla 2: por clase del proyecto
    print("\nPOR CLASE DEL PROYECTO")
    for clase in (CLASE_BENIGNA, CLASE_ALTA):
        print(f"  {ETIQUETA[clase]:<32}{por_clase[clase]:>8} estudios")
    excl = sum(v for k, v in por_clase.items() if k not in (CLASE_BENIGNA, CLASE_ALTA))
    print(f"  {'Excluidos (BI-RADS 0,1,3,6 u otros)':<32}{excl:>8} estudios")

    repetidos = {c: n for c, n in studies_por_codigo.items() if n > 1}
    if repetidos:
        print(f"\nAVISO: {len(repetidos)} códigos de paciente aparecen en más de un estudio "
              f"(ejemplos: {list(repetidos)[:5]}). Revísalos con el script 02.")
    if ilegibles:
        print(f"AVISO: {len(ilegibles)} XML no se pudieron leer (ejemplo: {ilegibles[0]})")

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["birads", "estudios", "codigos_distintos", "destino"])
        w.writeheader()
        w.writerows(filas)
    print(f"\nTabla guardada en {args.csv}")


if __name__ == "__main__":
    main()
