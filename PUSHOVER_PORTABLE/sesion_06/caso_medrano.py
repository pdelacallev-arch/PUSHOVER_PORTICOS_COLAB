"""Caso gravitacional trazable para el modelo Medrano de seis pisos."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sesion_06.cargas_gravedad import crear_caso_gravitacional_por_portico
from sesion_06.rotulas_axiales import asignar_secciones_por_reglas


RUTA_CARGAS = Path(__file__).with_name("datos") / "entrada_cargas_medrano_p6.json"
RUTA_CARGAS_2PISOS = Path(__file__).with_name("datos") / "entrada_cargas_2pisos.json"

# Rutas de los archivos de secciones (schema 1.0) compartidos con la sesión 03.
RUTA_SECCIONES_MEDRANO = (
    Path(__file__).resolve().parents[1] / "sesion_03" / "datos" / "entrada_secciones_medrano_p6.json"
)
RUTA_SECCIONES_2PISOS = (
    Path(__file__).resolve().parents[1] / "sesion_03" / "datos" / "entrada_secciones_2pisos.json"
)


def cargar_datos_cargas(ruta=RUTA_CARGAS):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def cargar_reglas_asignacion(datos):
    """Extrae y valida las reglas de asignación del JSON de entrada (schema 1.0).

    Las reglas usan el mismo formato que consume ``asignar_secciones_por_reglas``:
    ``{"tipo": ..., "niveles": [...], "seccion_id": ...}`` con filtros opcionales
    (``portico``, ``ids_elementos``). La lista vacía o una regla sin ``seccion_id``
    se rechazan aquí, para que el error ocurra al leer la entrada y no a mitad
    del análisis. Los filtros omitidos actúan como comodines.
    """
    asignacion = datos.get("asignacion")
    if not isinstance(asignacion, dict):
        raise ValueError("El JSON de entrada debe contener un bloque 'asignacion'.")
    reglas = asignacion.get("reglas")
    if not isinstance(reglas, list) or not reglas:
        raise ValueError("'asignacion.reglas' debe ser una lista no vacía.")
    for i, regla in enumerate(reglas):
        if not isinstance(regla, dict) or "seccion_id" not in regla:
            raise ValueError(f"Regla {i}: cada regla necesita 'seccion_id'.")
        if "niveles" in regla:
            try:
                [int(x) for x in regla["niveles"]]
            except (TypeError, ValueError):
                raise ValueError(f"Regla {i}: 'niveles' debe ser una lista de enteros.")
    return reglas


def reglas_secciones_medrano():
    """C1/C2 por intervalo de pisos y V1/V2 por nivel.

    Leído del JSON de entrada (schema 1.0); es el caso particular del edificio
    de 6 pisos de Medrano. El test de equivalencia garantiza que las reglas
    coinciden con la versión hardcodeada previa.
    """
    return cargar_reglas_asignacion(
        json.loads(Path(RUTA_SECCIONES_MEDRANO).read_text(encoding="utf-8"))
    )


def reglas_secciones_2pisos():
    """Reglas del edificio ejemplo del curso (2 pisos, 1 vano, P1-P2).

    C1/V1 en el nivel 1 y C2/V2 en el nivel 2, leídas del JSON de entrada.
    Demuestra el patrón para un edificio nuevo: solo se cambian los valores
    del JSON (schema 1.0), nunca las claves ni el código.
    """
    return cargar_reglas_asignacion(
        json.loads(Path(RUTA_SECCIONES_2PISOS).read_text(encoding="utf-8"))
    )


def crear_caso_gravitacional_desde_entradas(
    contrato,
    ruta_cargas,
    ruta_secciones,
):
    """Construye la gravedad para cualquier geometría compatible con el contrato.

    Las cargas, anchos tributarios, combinación y secciones se leen de JSON.
    No se fijan ids de pórticos, número de niveles ni nombres de sección.
    """
    entrada = cargar_datos_cargas(ruta_cargas)
    datos_secciones = json.loads(Path(ruta_secciones).read_text(encoding="utf-8"))
    return crear_caso_gravitacional_desde_datos(contrato, entrada, datos_secciones)


def crear_caso_gravitacional_desde_datos(contrato, entrada, datos_secciones):
    """Versión en memoria usada por el ejecutor de la entrada maestra."""
    ids = tuple(str(x) for x in contrato["geometria"]["ids_porticos"])
    ids_entrada = tuple(
        str(x) for x in entrada.get("edificio", {}).get(
            "porticos_direccion_analisis", ids
        )
    )
    if ids_entrada != ids:
        raise ValueError(
            "Los pórticos del archivo de cargas no coinciden con el contrato: "
            f"cargas={list(ids_entrada)}, contrato={list(ids)}."
        )

    anchos_entrada = entrada.get("anchos_tributarios_m", {})
    faltantes = set(ids) - set(anchos_entrada)
    if faltantes:
        raise ValueError(f"Faltan anchos tributarios para {sorted(faltantes)}.")
    anchos = {pid: 100.0 * float(anchos_entrada[pid]) for pid in ids}

    D = float(entrada["carga_muerta_superficial"]["total_D_kgf_m2"]) / 10_000.0
    vivo = entrada["carga_viva_kgf_m2"]
    L_tipico = vivo.get("pisos_tipicos", vivo.get("pisos_1_a_5"))
    L_techo = vivo.get("techo", vivo.get("techo_nivel_6"))
    if L_tipico is None or L_techo is None:
        raise ValueError(
            "'carga_viva_kgf_m2' debe definir pisos_tipicos y techo."
        )
    L_tipico, L_techo = float(L_tipico) / 10_000.0, float(L_techo) / 10_000.0

    reglas = cargar_reglas_asignacion(datos_secciones)
    asignacion = asignar_secciones_por_reglas(contrato, reglas)
    secciones = datos_secciones.get("secciones", {})
    usadas = set(asignacion.values())
    if not usadas <= set(secciones):
        raise ValueError(
            f"Faltan definiciones de sección para {sorted(usadas - set(secciones))}."
        )

    nodos = {int(k): np.asarray(v, dtype=float) for k, v in contrato["nodos"].items()}
    elevaciones = np.asarray(sorted({float(xy[1]) for xy in nodos.values()}))
    n_pisos = len(elevaciones) - 1
    gamma = float(entrada["peso_propio"]["gamma_concreto_kgf_m3"])
    t_losa = float(entrada["peso_propio"]["espesor_losa_m"])

    cargas_area_elemento = {}
    cargas_lineales_elemento = {}
    cargas_nodales = {}
    for elemento in contrato["elementos"]:
        eid = int(elemento["id"])
        i, j = int(elemento["i"]), int(elemento["j"])
        sid = asignacion[eid]
        seccion = secciones[sid]
        tipo = str(elemento["tipo"])
        if tipo == "viga":
            nivel = int(np.argmin(np.abs(elevaciones - max(nodos[i][1], nodos[j][1]))))
            L_ejes = float(np.linalg.norm(nodos[j] - nodos[i]))
            L_flexible = (
                L_ejes - float(elemento.get("brazo_i_cm", 0.0))
                - float(elemento.get("brazo_j_cm", 0.0))
            )
            if L_flexible <= 0:
                raise ValueError(f"Elemento {eid}: longitud flexible no positiva.")
            factor_area = L_ejes / L_flexible
            cargas_area_elemento[eid] = {
                "D": D * factor_area,
                "L": (L_techo if nivel == n_pisos else L_tipico) * factor_area,
            }
            b_m = float(seccion["b_cm"]) / 100.0
            h_m = float(seccion["h_cm"]) / 100.0
            peso_neto_kgf_m = b_m * max(h_m - t_losa, 0.0) * gamma
            cargas_lineales_elemento[eid] = {
                "D": peso_neto_kgf_m / 100.0,
                "L": 0.0,
            }
        elif tipo == "columna":
            longitud_m = float(np.linalg.norm(nodos[j] - nodos[i])) / 100.0
            peso = (
                float(seccion["b_cm"]) / 100.0
                * float(seccion["h_cm"]) / 100.0
                * longitud_m * gamma
            )
            nodo_superior = i if nodos[i][1] > nodos[j][1] else j
            registro = cargas_nodales.setdefault(nodo_superior, {"D": 0.0, "L": 0.0})
            registro["D"] += peso

    factores = entrada["combinacion_gravitacional"]
    return crear_caso_gravitacional_por_portico(
        contrato,
        cargas_area_kgf_cm2={},
        anchos_tributarios_cm=anchos,
        factores_combinacion={"D": factores["D"], "L": factores["L"]},
        cargas_area_por_elemento_kgf_cm2=cargas_area_elemento,
        cargas_lineales_por_elemento_kgf_cm=cargas_lineales_elemento,
        cargas_nodales=cargas_nodales,
        descripcion=str(entrada.get("descripcion", "caso gravitacional desde JSON")),
        supuestos=(
            "cargas y factores leídos del archivo de entrada",
            "carga superficial escalada por L_ejes/L_flexible",
            "peso de viga neto bajo la losa",
            "peso de cada tramo de columna concentrado en su nodo superior",
        ),
    )


def resumen_cargas(contrato, caso):
    """Resume un caso gravitacional sin depender del nombre del edificio."""
    por_portico_nivel = {}
    for registro in caso.cargas_distribuidas.values():
        clave = (registro["portico"], int(registro["nivel"]))
        por_portico_nivel[clave] = por_portico_nivel.get(clave, 0.0) + (
            registro["q_total_kgf_cm"] * registro["L_cargada_cm"]
        )
    peso_vigas = sum(
        r["q_total_kgf_cm"] * r["L_cargada_cm"]
        for r in caso.cargas_distribuidas.values()
    )
    peso_nodal = -float(np.asarray(caso.vector_nodal)[1::3].sum())
    return {
        "por_portico_nivel_kgf": por_portico_nivel,
        "carga_distribuida_total_kgf": peso_vigas,
        "carga_nodal_columnas_kgf": peso_nodal,
        "carga_gravitacional_total_kgf": peso_vigas + peso_nodal,
    }


def crear_caso_gravitacional_medrano(contrato, ruta=RUTA_CARGAS):
    """Compatibilidad: caso Medrano construido por el lector genérico."""
    return crear_caso_gravitacional_desde_entradas(
        contrato, ruta, RUTA_SECCIONES_MEDRANO
    )


def resumen_cargas_medrano(contrato, caso):
    """Compatibilidad con el nombre histórico del resumen."""
    return resumen_cargas(contrato, caso)


__all__ = ["RUTA_CARGAS", "RUTA_CARGAS_2PISOS", "RUTA_SECCIONES_MEDRANO", "RUTA_SECCIONES_2PISOS",
           "cargar_datos_cargas", "cargar_reglas_asignacion",
           "reglas_secciones_medrano", "reglas_secciones_2pisos",
           "crear_caso_gravitacional_desde_entradas", "crear_caso_gravitacional_desde_datos",
           "resumen_cargas",
           "crear_caso_gravitacional_medrano", "resumen_cargas_medrano"]
