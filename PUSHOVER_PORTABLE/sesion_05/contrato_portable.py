"""Construcción autónoma del contrato estructural de la Sesión 5."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from sesion_05.modelo_portico import (
    crear_modelo_edificio_porticos,
    normalizar_geometria_edificio,
)
from sesion_05.rotulas import LeyRotulaPlastica, asignar_rotulas_extremos


def _jsonificable(valor):
    if isinstance(valor, np.ndarray):
        return [_jsonificable(x) for x in valor.tolist()]
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating,)):
        return float(valor)
    if isinstance(valor, Mapping):
        return {str(k): _jsonificable(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_jsonificable(x) for x in valor]
    return valor


def _asignar_secciones(elementos, nodos, reglas):
    elevaciones = np.asarray(sorted({float(xy[1]) for xy in nodos.values()}))
    salida = {}
    for elemento in elementos:
        eid = int(elemento["id"])
        nivel = int(np.argmin(np.abs(
            elevaciones - max(nodos[int(elemento["i"])][1], nodos[int(elemento["j"])][1])
        )))
        coincidencias = []
        for regla in reglas:
            if "portico" in regla and str(regla["portico"]) != str(elemento["id_portico"]):
                continue
            if "tipo" in regla and str(regla["tipo"]) != str(elemento["tipo"]):
                continue
            if "niveles" in regla and nivel not in {int(x) for x in regla["niveles"]}:
                continue
            if "ids_elementos" in regla and eid not in {int(x) for x in regla["ids_elementos"]}:
                continue
            coincidencias.append(str(regla["seccion_id"]))
        if len(set(coincidencias)) != 1:
            raise ValueError(
                f"Elemento {eid}: debe coincidir con una única sección; obtuvo {coincidencias}."
            )
        salida[eid] = coincidencias[0]
    return salida


def _brazos_en_caras(elementos, nodos, asignacion, secciones):
    conectados = {}
    for elemento in elementos:
        for nodo in (int(elemento["i"]), int(elemento["j"])):
            conectados.setdefault(nodo, []).append(elemento)

    def brazo(nodo, tipo_actual):
        if np.isclose(float(nodos[nodo][1]), 0.0):
            return 0.0
        transversales = [
            e for e in conectados[nodo] if str(e["tipo"]) != tipo_actual
        ]
        if not transversales:
            return 0.0
        dimensiones = []
        for otro in transversales:
            sec = secciones[asignacion[int(otro["id"])]]
            dimensiones.append(
                float(sec["h_cm"] if tipo_actual == "columna" else sec["b_cm"]) / 2.0
            )
        return max(dimensiones)

    return {
        int(e["id"]): (brazo(int(e["i"]), str(e["tipo"])),
                        brazo(int(e["j"]), str(e["tipo"])))
        for e in elementos
    }


def construir_contrato_sesion05(
    entrada: Mapping,
    biblioteca_rotulas: Mapping,
) -> dict:
    """Crea geometría, propiedades y rótulas solo a partir de la entrada maestra."""
    if "edificio" not in entrada or "secciones" not in entrada:
        raise ValueError("La entrada necesita 'edificio' y 'secciones'.")
    reglas = entrada.get("asignacion", {}).get("reglas")
    if not isinstance(reglas, list) or not reglas:
        raise ValueError("La entrada necesita 'asignacion.reglas'.")
    modelo_cfg = entrada.get("modelo", {})
    for campo in ("EA_diafragma_kgf", "Ec_coef_sqrt_fc", "P_referencia_inicial_kgf"):
        if campo not in modelo_cfg:
            raise ValueError(f"Falta modelo.{campo} en la entrada maestra.")

    geometria = normalizar_geometria_edificio(entrada)
    ids_seccion = list(entrada["secciones"])
    if not ids_seccion or not set(ids_seccion) <= set(biblioteca_rotulas):
        raise ValueError("La biblioteca no cubre todas las secciones de la entrada.")

    from sesion_06.rotulas_axiales import interpolar_ley_axial

    P0 = float(modelo_cfg["P_referencia_inicial_kgf"])
    leyes = {}
    for sid in ids_seccion:
        familia = biblioteca_rotulas[sid]["positivo"]
        seleccion = interpolar_ley_axial(familia, P0, fuera_de_rango="limitar")
        leyes[sid] = seleccion

    primera = entrada["secciones"][ids_seccion[0]]
    Ec0 = float(modelo_cfg["Ec_coef_sqrt_fc"]) * np.sqrt(float(primera["fc_kgf_cm2"]))
    EA0 = Ec0 * float(primera["b_cm"]) * float(primera["h_cm"])
    EI0 = float(leyes[ids_seccion[0]]["EI_sec_kgf_cm2"])
    datos_porticos = [
        {
            **p, "EA_columna": EA0, "EI_columna": EI0,
            "EA_viga": EA0, "EI_viga": EI0,
        }
        for p in geometria["geometrias_porticos"]
    ]
    modelo = crear_modelo_edificio_porticos(
        datos_porticos,
        EA_puntal_kgf=float(modelo_cfg["EA_diafragma_kgf"]),
        separaciones_porticos_cm=geometria["separaciones_porticos_cm"],
        incluir_puntales_en_base=bool(modelo_cfg.get("incluir_diafragma_en_base", False)),
    )
    asignacion = _asignar_secciones(modelo["elementos"], modelo["nodos"], reglas)
    faltantes = set(asignacion.values()) - set(entrada["secciones"])
    if faltantes:
        raise ValueError(f"Faltan secciones asignadas: {sorted(faltantes)}.")

    metodo_brazos = str(modelo_cfg.get("brazos_rigidos", "automatico_caras"))
    if metodo_brazos == "automatico_caras":
        brazos = _brazos_en_caras(
            modelo["elementos"], modelo["nodos"], asignacion, entrada["secciones"]
        )
    elif metodo_brazos == "ninguno":
        brazos = {int(e["id"]): (0.0, 0.0) for e in modelo["elementos"]}
    else:
        raise ValueError("modelo.brazos_rigidos debe ser 'automatico_caras' o 'ninguno'.")

    for elemento in modelo["elementos"]:
        eid = int(elemento["id"])
        sid = asignacion[eid]
        sec = entrada["secciones"][sid]
        Ec = float(sec.get(
            "Ec_kgf_cm2",
            float(modelo_cfg["Ec_coef_sqrt_fc"]) * np.sqrt(float(sec["fc_kgf_cm2"])),
        ))
        elemento["EA"] = Ec * float(sec["b_cm"]) * float(sec["h_cm"])
        elemento["EI"] = float(leyes[sid]["EI_sec_kgf_cm2"])
        elemento["brazo_i_cm"], elemento["brazo_j_cm"] = brazos[eid]
        elemento["seccion_id"] = sid
        elemento["EI_origen"] = f"biblioteca_M_theta(P={P0:g} kgf), estado inicial"

    ley_obj = {
        sid: LeyRotulaPlastica(**{
            k: float(leyes[sid]["ley"][k])
            for k in ("My_kgf_cm", "Mu_kgf_cm", "theta_p_u_rad", "Kp_kgf_cm_rad")
        })
        for sid in ids_seccion
    }
    primer_tipo = {}
    for e in modelo["elementos"]:
        primer_tipo.setdefault(str(e["tipo"]), asignacion[int(e["id"])])
    rotulas = asignar_rotulas_extremos(
        modelo["nodos"], modelo["elementos"],
        {tipo: ley_obj[sid] for tipo, sid in primer_tipo.items()},
    )
    elementos = {int(e["id"]): e for e in modelo["elementos"]}
    for rotula in rotulas:
        elemento = elementos[int(rotula["elemento"])]
        sid = asignacion[int(elemento["id"])]
        rotula["ley"] = asdict(ley_obj[sid])
        rotula["criterios_aceptacion"] = deepcopy(leyes[sid]["criterios_aceptacion"])
        rotula["seccion_id"] = sid
        rotula["id_portico"] = str(elemento["id_portico"])

    contrato = {
        "schema_version": "3.4", "modelo": "edificio_porticos_2d_en_serie",
        "caso_id": str(entrada.get("caso_id", "sin_id")),
        "geometria": {
            k: modelo[k] for k in (
                "n_porticos", "n_pisos", "ids_porticos", "n_vanos_por_portico",
                "luces_cm_por_portico", "alturas_cm_por_portico",
                "separaciones_porticos_cm", "origenes_porticos_x_cm",
                "nodos_base", "nodos_techo",
            )
        },
        "nodos": modelo["nodos"], "gdl": modelo["gdl"],
        "elementos": modelo["elementos"], "puntales": modelo["puntales"],
        "rotulas": rotulas, "restricciones_gdl": modelo["restricciones_gdl"],
        "estado_inicial_rotulas": {r["id"]: r["estado_inicial"] for r in rotulas},
        "asignacion_secciones": asignacion,
        "convenciones": {"unidades": "kgf, cm, kgf*cm, rad"},
    }
    return _jsonificable(contrato)


def guardar_contrato_sesion05(contrato: Mapping, ruta: str | Path) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(_jsonificable(contrato), ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta


__all__ = ["construir_contrato_sesion05", "guardar_contrato_sesion05"]
