"""
04_crear_pliegues.py
Rehace el CSV del dataset con:
  - IDs NEUTROS (P0001...) asignados al azar: el ID ya no delata la clase ni el BI-RADS
  - validación cruzada ESTRATIFICADA DE 5 PLIEGUES A NIVEL DE PACIENTE, estratificada por
    categoría BI-RADS (2, 4, 5) y no solo por clase
  - varias semillas (por defecto 42, 43, 44) para poder repetir la validación cruzada
  - sin la columna 'particion' (70/15/15): con esta muestra no es fiable

Cada paciente es UNA unidad (sus 4 vistas siempre van juntas), por lo que estratificar
por paciente equivale a un StratifiedGroupKFold con un grupo por paciente.

Uso de los pliegues en cada ronda k (0..4):
    prueba     = pliegue k
    validación = pliegue (k+1) % 5      (parada temprana; nunca se usa la prueba)
    entrenamiento = los otros tres

Uso:
    python 04_crear_pliegues.py dataset_final.csv SALIDA
Salida (en la carpeta SALIDA):
    dataset_final_v2.csv      PRIVADO (Drive): IDs neutros + código del hospital + rutas + pliegues
    clave_identificacion.csv  PRIVADO (nunca a GitHub): ID neutro <-> código del hospital
    splits_publico.csv        PÚBLICO (puede ir a GitHub): ID neutro, etiqueta y pliegues
    verificacion_pliegues.txt resumen de controles
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

N_PLIEGUES = 5
SEMILLA_IDS = 2026


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("csv_entrada")
    ap.add_argument("salida")
    ap.add_argument("--semillas", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()

    d = pd.read_csv(args.csv_entrada)
    necesarias = {"paciente", "birads_carpeta", "clase", "etiqueta", "lado", "vista", "ruta_png"}
    faltan = necesarias - set(d.columns)
    if faltan:
        sys.exit(f"Faltan columnas en el CSV: {sorted(faltan)}")

    # ---- una fila por paciente (la unidad de reparto)
    pac = d.drop_duplicates("paciente")[["paciente", "birads_carpeta", "clase", "etiqueta"]].copy()
    assert pac.paciente.is_unique
    assert (d.groupby("paciente").size() == 4).all(), "hay pacientes sin exactamente 4 imágenes"
    assert (d.groupby("paciente")[["etiqueta", "birads_carpeta"]].nunique() == 1).all().all(), \
        "etiqueta o BI-RADS inconsistente dentro de un paciente"
    pac = pac.sort_values("paciente").reset_index(drop=True)

    # ---- IDs neutros: permutación aleatoria (si fueran correlativos por clase, el número delataría la etiqueta)
    rng = np.random.default_rng(SEMILLA_IDS)
    orden = rng.permutation(len(pac))
    pac["paciente_id"] = [f"P{i + 1:04d}" for i in orden]

    # ---- pliegues: estratificados por BI-RADS (2/4/5), uno por semilla
    estrato = pac["birads_carpeta"].str.extract(r"(\d)")[0]
    for s in args.semillas:
        col = f"fold_seed{s}"
        pac[col] = -1
        skf = StratifiedKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=s)
        for k, (_, idx_test) in enumerate(skf.split(pac, estrato)):
            pac.loc[idx_test, col] = k
        assert (pac[col] >= 0).all()
    cols_fold = [f"fold_seed{s}" for s in args.semillas]

    # ---- CSV a nivel de imagen
    base = d.rename(columns={"paciente": "codigo_hospital"})
    cols_base = [c for c in ["codigo_hospital", "birads_carpeta", "clase", "etiqueta", "lado", "vista",
                             "ruta_png", "qc"] if c in base.columns]
    v2 = base[cols_base].merge(
        pac[["paciente", "paciente_id"] + cols_fold].rename(columns={"paciente": "codigo_hospital"}),
        on="codigo_hospital", how="left")
    v2 = v2.rename(columns={"birads_carpeta": "birads"})
    orden_cols = ["paciente_id", "codigo_hospital", "birads", "clase", "etiqueta", "lado", "vista",
                  "ruta_png"] + cols_fold + (["qc"] if "qc" in v2.columns else [])
    v2 = v2[orden_cols].sort_values(["paciente_id", "lado", "vista"]).reset_index(drop=True)

    out = Path(args.salida)
    out.mkdir(parents=True, exist_ok=True)
    v2.to_csv(out / "dataset_final_v2.csv", index=False)
    pac[["paciente_id", "paciente"]].rename(columns={"paciente": "codigo_hospital"}) \
        .sort_values("paciente_id").to_csv(out / "clave_identificacion.csv", index=False)
    pac[["paciente_id", "etiqueta"] + cols_fold].sort_values("paciente_id") \
        .to_csv(out / "splits_publico.csv", index=False)

    # ---- VERIFICACIONES (si algo falla, el script se detiene)
    lineas = [f"Pacientes: {len(pac)} | imágenes: {len(v2)} | semillas: {args.semillas}", ""]
    ok = True
    for c in cols_fold:
        t = pac.groupby(c).agg(pacientes=("paciente_id", "size"), alta=("etiqueta", "sum"),
                               birads4=("birads_carpeta", lambda x: (x == "BIRADS_4").sum()),
                               birads5=("birads_carpeta", lambda x: (x == "BIRADS_5").sum()))
        lineas += [c, t.to_string(), ""]
        ok &= t.alta.max() - t.alta.min() <= 1 and t.birads5.max() - t.birads5.min() <= 1
        # cada paciente en un único pliegue por semilla y sus 4 imágenes iguales
        assert (v2.groupby("paciente_id")[c].nunique() == 1).all()
        for k in range(N_PLIEGUES):                        # rondas: test/val/train disjuntos
            te, va = v2[c] == k, v2[c] == (k + 1) % N_PLIEGUES
            tr = ~(te | va)
            assert not (set(v2[te].paciente_id) & set(v2[va].paciente_id))
            assert not (set(v2[te].paciente_id) & set(v2[tr].paciente_id))
            assert not (set(v2[va].paciente_id) & set(v2[tr].paciente_id))
            assert v2[tr].etiqueta.nunique() == 2 and v2[va].etiqueta.nunique() == 2 and v2[te].etiqueta.nunique() == 2
    # el ID neutro no debe estar relacionado con la clase
    num = pac.paciente_id.str[1:].astype(int)
    corr = np.corrcoef(num, pac.etiqueta)[0, 1]
    lineas.append(f"Correlación entre número del ID y etiqueta: {corr:+.3f} (debe ser ~0)")
    ok &= abs(corr) < 0.15
    # semillas distintas dan pliegues distintos
    if len(cols_fold) > 1:
        iguales = (pac[cols_fold[0]] == pac[cols_fold[1]]).mean()
        lineas.append(f"Coincidencia de pliegues entre semillas {args.semillas[:2]}: {iguales:.0%} (esperado ~20%)")
    lineas.append("VERIFICACIÓN: " + ("CORRECTA" if ok else "REVISAR"))
    (out / "verificacion_pliegues.txt").write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas))
    print(f"\nArchivos escritos en {out}")
    print("RECUERDA: clave_identificacion.csv y dataset_final_v2.csv NO se suben a GitHub.")


if __name__ == "__main__":
    main()
