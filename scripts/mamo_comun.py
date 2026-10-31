"""
Funciones compartidas por los scripts 01, 02 y 03.
Deja este archivo en la MISMA carpeta que los demás scripts.

Lee el XML de cada estudio (formato ClearCanvasStudyXml), así que no hace falta
abrir los DICOM pesados para contar o filtrar.
"""
import gzip
import re
import xml.etree.ElementTree as ET
from pathlib import Path

# DICOM PS3.6: 1.2.840.10008.5.1.4.1.1.1.2 = Digital Mammography X-Ray Image
# Storage - For Presentation.  (OJO: ...1.1.2 es CT Image Storage, no mamografía)
SOP_FOR_PRESENTATION = "1.2.840.10008.5.1.4.1.1.1.2"
COMBINACIONES = [("R", "CC"), ("L", "CC"), ("R", "MLO"), ("L", "MLO")]

# Clases del estudio (según la metodología 2.1.6)
CLASE_BENIGNA = "benigna"
CLASE_ALTA = "alta_sospecha"


def atributos(elemento):
    """Etiquetas simples (sin secuencias) de un elemento del XML."""
    return {a.attrib["Tag"]: (a.text or "").strip()
            for a in elemento.findall("Attribute") if len(a) == 0}


def leer_estudio_xml(ruta_xml):
    """Devuelve (uid_estudio, [instancias]) o None si el XML no sirve.
    Cada instancia es un dict con todas sus etiquetas (las comunes de la serie
    ya mezcladas) más: _uid, _sop, _serie, _orden (posición en el XML)."""
    try:
        if ruta_xml.name.lower().endswith(".gz"):
            with gzip.open(ruta_xml, "rb") as f:
                raiz = ET.parse(f).getroot()
        else:
            raiz = ET.parse(ruta_xml).getroot()
    except (ET.ParseError, OSError):
        return None
    if raiz.tag != "ClearCanvasStudyXml":
        return None
    estudio = raiz.find("Study")
    if estudio is None:
        return None
    instancias = []
    for serie in estudio.findall("Series"):
        comunes = {}
        base = serie.find("BaseInstance")
        if base is not None:
            for x in base.findall("Instance"):
                comunes.update(atributos(x))
            comunes.update(atributos(base))
        for inst in serie.findall("Instance"):
            d = dict(comunes)
            d.update(atributos(inst))
            d["_uid"] = inst.attrib.get("UID", "")
            d["_sop"] = inst.attrib.get("SopClassUID", "")
            d["_serie"] = serie.attrib.get("UID", "")
            d["_orden"] = len(instancias)
            instancias.append(d)
    return estudio.attrib.get("UID", ""), instancias


def encontrar_estudios(entradas):
    """Lista de (carpeta, uid_estudio, instancias) sin duplicar el mismo
    StudyInstanceUID aunque esté en varias carpetas (disco + PC)."""
    estudios, vistos, ilegibles = [], set(), []
    for entrada in entradas:
        entrada = Path(entrada)
        if not entrada.exists():
            print(f"AVISO: no existe la ruta {entrada}")
            continue
        rutas = sorted(list(entrada.rglob("*.xml")) + list(entrada.rglob("*.xml.gz")))
        for ruta in rutas:
            leido = leer_estudio_xml(ruta)
            if not leido or not leido[1]:
                ilegibles.append(ruta)
                continue
            if leido[0] in vistos:
                continue
            vistos.add(leido[0])
            estudios.append((ruta.parent, leido[0], leido[1]))
    return estudios, ilegibles


def clasificar_birads(codigo):
    """
    Lee la categoría del código del paciente (B2-0282, B4a-0001, B4-0006...).
    Devuelve (birads, clase, carpeta):
      BI-RADS 2        -> benigna
      BI-RADS 4 y 5    -> alta_sospecha   (4a/4b/4c se agrupan en 4)
      BI-RADS 0,1,3,6  -> excluido_<n>    (según la metodología 2.1.6.1)
    """
    m = re.match(r"^B(\d)([a-cA-C])?-", codigo or "")
    if not m:
        return "", "desconocido", "BIRADS_otros"
    n = int(m.group(1))
    sub = (m.group(2) or "").lower()
    birads = f"{n}{sub}"
    if n == 2:
        return birads, CLASE_BENIGNA, "BIRADS_2"
    if n in (4, 5):
        return birads, CLASE_ALTA, f"BIRADS_{n}"
    return birads, f"excluido_{n}", f"BIRADS_{n}" if n in (0, 1, 3, 6) else "BIRADS_otros"


def codigo_paciente(instancias):
    """PatientID (0010,0020) tal como está en el DICOM, no el nombre de carpeta."""
    for d in instancias:
        v = d.get("00100020", "")
        if v:
            return re.sub(r"[^A-Za-z0-9_-]", "_", v)
    return "SIN_ID"


def a_entero(valor, defecto=0):
    try:
        return int(float(str(valor).strip()))
    except (TypeError, ValueError):
        return defecto


def es_2d_presentacion(d):
    """2D «For Presentation» de mamografía; descarta tomosíntesis y FOR PROCESSING."""
    if d.get("_sop") != SOP_FOR_PRESENTATION:
        return False
    tipo = d.get("00080008", "").upper()
    if "PROJECTION" in tipo or "VOLUME" in tipo:
        return False
    return d.get("00080068", "").upper() in ("", "FOR PRESENTATION")


def lado_y_vista(d):
    """('R'|'L', 'CC'|'MLO') o None si no es una de las 4 vistas estándar."""
    lado = (d.get("00200062", "") or d.get("00200060", "")).upper()
    vista = d.get("00185101", "").upper()
    if lado in ("R", "L") and vista in ("CC", "MLO"):
        return lado, vista
    return None


def orden_cronologico(d):
    """Clave para saber cuál es la toma MÁS RECIENTE de una vista repetida.
    Los InstanceNumber se reinician en cada serie, así que NO sirven solos:
    primero va el SeriesNumber (aumenta con cada adquisición), luego el
    InstanceNumber y por último el orden en el XML."""
    return (a_entero(d.get("00200011")), a_entero(d.get("00200013")), d["_orden"])
