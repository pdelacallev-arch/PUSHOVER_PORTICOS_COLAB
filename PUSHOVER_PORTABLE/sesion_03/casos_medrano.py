"""Preparación de C1, C2, V1 y V2 del edificio de 6 pisos de Medrano.

El módulo adapta los datos de la tesis al núcleo de fibras de la sesión 03.
Unidades: kgf, cm, kgf/cm2, kgf-cm y rad. Compresión axial positiva.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from sesion_02.mander_concreto import (
    deformacion_ultima,
    factor_efectividad_confinamiento,
    presion_lateral,
    relacion_volumetrica,
    resistencia_confinada,
)
from sesion_03.modelo_rotula import criterios_columna_asce41_17, propiedades_rotula
from sesion_03.modelo_seccion import crear_fibras, curva_momento_curvatura, resumir_curva


RUTA_ENTRADA = (
    Path(__file__).resolve().parent
    / "datos"
    / "entrada_secciones_medrano_p6.json"
)
PULGADA_CM = 2.54


def cargar_entrada(ruta=RUTA_ENTRADA):
    """Lee una entrada de secciones sin imponer nombres propios del caso."""
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    secciones = datos.get("secciones")
    if not isinstance(secciones, dict) or not secciones:
        raise ValueError("La entrada debe contener un catálogo no vacío de 'secciones'.")
    for sid, seccion in secciones.items():
        if not isinstance(seccion, dict) or seccion.get("tipo") not in {"columna", "viga"}:
            raise ValueError(f"Sección {sid}: 'tipo' debe ser 'columna' o 'viga'.")
    return datos


def _barra(x, y, db):
    return {"x_cm": float(x), "y_cm": float(y), "diametro_cm": float(db)}


def _barras_columna(sec, db_e):
    """12 barras perimetrales: 4 arriba, 4 abajo y 2 en cada lateral."""
    b, h = float(sec["b_cm"]), float(sec["h_cm"])
    rec = float(sec["recubrimiento_cm"])
    db = float(sec["barras"]["diametro_in"]) * PULGADA_CM
    x_max = b / 2 - rec - db_e - db / 2
    y_max = h / 2 - rec - db_e - db / 2
    coordenadas = sec["barras"].get("coordenadas_cm")
    if coordenadas is not None:
        barras = [
            _barra(p["x_cm"], p["y_cm"], p.get("diametro_cm", db))
            for p in coordenadas
        ]
        if len(barras) != int(sec["barras"]["n"]):
            raise ValueError("El número de coordenadas no coincide con barras.n.")
        return barras

    n_total = int(sec["barras"]["n"])
    disposicion = sec["barras"].get("disposicion", {})
    n_horizontal = int(disposicion.get("n_cara_horizontal", 4))
    restantes = n_total - 2 * n_horizontal
    if n_horizontal < 2 or restantes < 0 or restantes % 2:
        raise ValueError(
            "Defina barras.coordenadas_cm o una disposición perimetral compatible."
        )
    n_interior_vertical = restantes // 2
    barras = []
    for y in (y_max, -y_max):
        barras.extend(
            _barra(x, y, db) for x in np.linspace(-x_max, x_max, n_horizontal)
        )
    for x in (-x_max, x_max):
        barras.extend(
            _barra(x, y, db)
            for y in np.linspace(-y_max, y_max, n_interior_vertical + 2)[1:-1]
        )
    if len(barras) != n_total:
        raise ValueError("No se pudo construir la disposición perimetral indicada.")
    return barras


def _capa(b, h, rec, db_e, n, db, superior):
    x_max = b / 2 - rec - db_e - db / 2
    y = h / 2 - rec - db_e - db / 2
    y = y if superior else -y
    return [_barra(x, y, db) for x in np.linspace(-x_max, x_max, int(n))]


def _barras_viga(sec, db_e):
    b, h = float(sec["b_cm"]), float(sec["h_cm"])
    rec = float(sec["recubrimiento_cm"])
    sup, inf = sec["barras"]["superior"], sec["barras"]["inferior"]
    db_sup = float(sup["diametro_in"]) * PULGADA_CM
    db_inf = float(inf["diametro_in"]) * PULGADA_CM
    return (
        _capa(b, h, rec, db_e, sup["n"], db_sup, True)
        + _capa(b, h, rec, db_e, inf["n"], db_inf, False)
    )


def _gaps_cara(cara, coordenada):
    orden = sorted(cara, key=lambda barra: barra[coordenada])
    return [
        max(
            b[coordenada] - a[coordenada]
            - 0.5 * (a["diametro_cm"] + b["diametro_cm"]),
            0.0,
        )
        for a, b in zip(orden, orden[1:])
    ]


def _distancias_libres(barras):
    """Distancias libres consecutivas en las cuatro caras del perímetro."""
    xs = np.array([b["x_cm"] for b in barras])
    ys = np.array([b["y_cm"] for b in barras])
    tol = 1e-8
    caras = (
        ([b for b in barras if abs(b["y_cm"] - ys.max()) < tol], "x_cm"),
        ([b for b in barras if abs(b["y_cm"] - ys.min()) < tol], "x_cm"),
        ([b for b in barras if abs(b["x_cm"] - xs.max()) < tol], "y_cm"),
        ([b for b in barras if abs(b["x_cm"] - xs.min()) < tol], "y_cm"),
    )
    return [gap for cara, eje in caras for gap in _gaps_cara(cara, eje)]


def preparar_seccion(nombre, ruta=RUTA_ENTRADA):
    """Devuelve datos de fibras, confinamiento y metadatos de una sección."""
    entrada = cargar_entrada(ruta)
    nombre = str(nombre).upper()
    if nombre not in entrada["secciones"]:
        raise KeyError(f"Sección desconocida: {nombre}")
    sec = deepcopy(entrada["secciones"][nombre])
    b, h = float(sec["b_cm"]), float(sec["h_cm"])
    rec = float(sec["recubrimiento_cm"])
    fc, fy, fyh = (float(sec[k]) for k in ("fc_kgf_cm2", "fy_kgf_cm2", "fyh_kgf_cm2"))
    est = sec["estribos"]
    db_e = float(est["diametro_in"]) * PULGADA_CM
    s = float(est["separacion_cm"])
    n_ramas = int(est["ramas_direccion"])
    barras = _barras_columna(sec, db_e) if sec["tipo"] == "columna" else _barras_viga(sec, db_e)

    ae = rec + db_e / 2
    bc, hc = b - 2 * ae, h - 2 * ae
    As_total = sum(np.pi * barra["diametro_cm"] ** 2 / 4 for barra in barras)
    area_rama = np.pi * db_e**2 / 4
    Asx = Asy = n_ramas * area_rama
    flx, fly = presion_lateral(Asx, Asy, s, bc, hc, fyh)
    ke = factor_efectividad_confinamiento(
        bc, hc, s, db_e, _distancias_libres(barras), As_total
    )
    fl_ef = ke * (flx + fly) / 2
    rho_s = relacion_volumetrica(Asx, Asy, s, bc, hc)
    fcc, epscc = resistencia_confinada(fc, fl_ef, 0.002)
    eps_cu = deformacion_ultima(fcc, rho_s, fyh, 0.10)
    datos = {
        "b": b, "h": h, "Ag": b * h,
        "recubrimiento_libre": rec, "db_estribo": db_e, "s_estribo": s,
        "fco": fc, "epsco": 0.002, "fy": fy, "fyh": fyh,
        "Es": 2_100_000.0, "Esh": 21_000.0, "eps_sh": 0.010,
        "eps_su": 0.10, "n_fibras_y": 80, "bc": bc, "hc": hc,
        "As_total": As_total, "rho_l": As_total / (b * h), "barras": barras,
    }
    return {
        "nombre": nombre, "tipo": sec["tipo"],
        "fuente": str(sec.get("fuente", entrada.get("fuente", "entrada del usuario"))),
        "datos_seccion": datos, "Lp_cm": float(sec["Lp_cm"]),
        "fl_ef": float(fl_ef), "rho_s": float(rho_s), "ke": float(ke),
        "fcc": float(fcc), "epscc": float(epscc), "eps_cu": float(eps_cu),
        "n_ramas": n_ramas,
    }


def criterios_viga_asumidos(nombre, ruta=RUTA_ENTRADA):
    """Límites editables adoptados para poder ejecutar el ejemplo completo."""
    nombre = str(nombre).upper()
    entrada = cargar_entrada(ruta)
    if nombre not in entrada["secciones"] or entrada["secciones"][nombre]["tipo"] != "viga":
        raise ValueError(f"La sección {nombre!r} no es una viga definida en la entrada.")
    datos = entrada["secciones"][nombre].get("criterios", {})
    limites = {k: datos.get(k) for k in ("IO", "LS", "CP")}
    if any(v is None for v in limites.values()):
        raise ValueError(f"La viga {nombre} debe definir criterios IO, LS y CP.")
    return {
        "referencia": str(datos.get("referencia", "DEFINIDO POR EL USUARIO")),
        "componente": f"viga de concreto armado {nombre}",
        "theta_p_rad": {k: float(v) for k, v in limites.items()},
        "parametros": {"origen": "archivo de entrada de secciones"},
        "supuestos_provisionales": [
            "valores editables; no atribuidos a una tabla ASCE 41-17 verificada",
            "pendientes las comprobaciones de cortante, anclaje y empalmes",
        ],
    }


def analizar_seccion(nombre, P_kgf=0.0, sentido="positivo", n_pasos=180):
    """Calcula M-phi y M-theta para una sección, carga axial y sentido."""
    cfg = preparar_seccion(nombre)
    datos = cfg["datos_seccion"]
    fibras = crear_fibras(datos)
    if sentido not in {"positivo", "negativo"}:
        raise ValueError("sentido debe ser 'positivo' o 'negativo'.")
    if sentido == "negativo":
        fibras = deepcopy(fibras)
        fibras["xs"] = -np.asarray(fibras["xs"])
        fibras["ys"] = -np.asarray(fibras["ys"])
    registros = curva_momento_curvatura(
        float(P_kgf), fibras, datos, cfg["fl_ef"], cfg["rho_s"], cfg["eps_cu"],
        phi_max=0.004, n_pasos=int(n_pasos),
    )
    resumen = resumir_curva(registros, float(P_kgf))
    criterios = None
    if cfg["tipo"] == "columna":
        criterios = criterios_columna_asce41_17(
            float(P_kgf), datos, ratio_cortante=0.20, n_ramas=cfg["n_ramas"]
        )
    else:
        criterios = criterios_viga_asumidos(nombre)
    rotula = propiedades_rotula(
        registros, resumen, datos, Lp=cfg["Lp_cm"], criterios=criterios
    )
    return {"configuracion": cfg, "fibras": fibras, "registros": registros,
            "resumen": resumen, "rotula": rotula, "sentido": sentido}


def cargas_axiales_columna(nombre, razones=None, ruta=RUTA_ENTRADA):
    """Crea una malla P = nu Ag f'c compatible con el ejemplo ASCE de sesión 03."""
    entrada = cargar_entrada(ruta)
    cfg = preparar_seccion(nombre, ruta)
    if cfg["tipo"] != "columna":
        return [0.0]
    seccion = entrada["secciones"][str(nombre).upper()]
    valores = seccion.get("P_kgf")
    if valores is not None:
        return [float(x) for x in valores]
    if razones is None:
        razones = entrada.get(
            "biblioteca_rotulas", {}
        ).get("razones_axiales_columnas", (-0.10, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50))
    d = cfg["datos_seccion"]
    return [float(nu * d["Ag"] * d["fco"]) for nu in razones]


__all__ = ["RUTA_ENTRADA", "cargar_entrada", "preparar_seccion",
           "criterios_viga_asumidos", "analizar_seccion",
           "cargas_axiales_columna"]
