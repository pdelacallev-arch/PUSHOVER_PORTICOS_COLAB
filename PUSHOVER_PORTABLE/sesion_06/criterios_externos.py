"""Lectura, validación y asignación de criterios IO-LS-CP externos."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Mapping

import numpy as np


ESTADOS_VERIFICADOS = {"VERIFICADO_POR_USUARIO", "REVISADO_POR_INGENIERO"}
ESTADOS_PROVISIONALES = {"EJEMPLO_ACADEMICO_DECLARADO", "PROVISIONAL_DECLARADO"}
SENTIDOS = ("positivo", "negativo")


def leer_catalogo_criterios(ruta: str | Path) -> dict:
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el catálogo de criterios: {ruta}")
    return json.loads(ruta.read_text(encoding="utf-8"))


def validar_catalogo_criterios(catalogo: Mapping, *, permitir_provisionales=False) -> dict:
    datos = deepcopy(dict(catalogo))
    if str(datos.get("schema_version")) != "1.0":
        raise ValueError("El catálogo de criterios requiere schema_version 1.0.")
    if datos.get("unidades", {}).get("rotacion") != "rad":
        raise ValueError("Los criterios externos deben declarar rotación en rad.")
    if not datos.get("norma") or not datos.get("edicion"):
        raise ValueError("El catálogo debe identificar norma y edición.")
    registros = datos.get("criterios")
    if not isinstance(registros, list) or not registros:
        raise ValueError("'criterios' debe ser una lista no vacía.")
    ids = set()
    for indice, registro in enumerate(registros):
        rid = str(registro.get("id", ""))
        if not rid or rid in ids:
            raise ValueError(f"Criterio {indice}: id ausente o duplicado.")
        ids.add(rid)
        if not isinstance(registro.get("aplica_a"), Mapping):
            raise ValueError(f"Criterio {rid}: falta el bloque aplica_a.")
        limites = registro.get("theta_p_rad", {})
        try:
            io, ls, cp = (float(limites[k]) for k in ("IO", "LS", "CP"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Criterio {rid}: IO, LS y CP deben ser numéricos.") from exc
        if not (0 <= io < ls < cp and np.all(np.isfinite([io, ls, cp]))):
            raise ValueError(f"Criterio {rid}: se requiere 0 <= IO < LS < CP.")
        estado = str(registro.get("estado", ""))
        permitido = estado in ESTADOS_VERIFICADOS or (
            permitir_provisionales and estado in ESTADOS_PROVISIONALES
        )
        if not permitido:
            raise ValueError(
                f"Criterio {rid}: estado {estado!r} no permitido para esta ejecución."
            )
        for campo in ("referencia", "condicion_gobernante", "fuente_calculo"):
            if not registro.get(campo):
                raise ValueError(f"Criterio {rid}: falta {campo}.")
        sentidos = registro["aplica_a"].get("sentidos", SENTIDOS)
        if not set(sentidos) <= set(SENTIDOS):
            raise ValueError(f"Criterio {rid}: sentido no válido.")
    return datos


def _nivel_elemento(elemento, nodos, elevaciones):
    y = max(float(nodos[int(elemento["i"])][1]), float(nodos[int(elemento["j"])][1]))
    return int(np.argmin(np.abs(elevaciones - y)))


def _coincide(regla, rotula, elemento, nivel, sentido):
    if "tipo" in regla and str(regla["tipo"]) != str(elemento["tipo"]):
        return False
    if "seccion_id" in regla and str(regla["seccion_id"]) != str(elemento["seccion_id"]):
        return False
    if "porticos" in regla and str(elemento["id_portico"]) not in {str(x) for x in regla["porticos"]}:
        return False
    if "niveles" in regla and nivel not in {int(x) for x in regla["niveles"]}:
        return False
    if "ids_elementos" in regla and int(elemento["id"]) not in {int(x) for x in regla["ids_elementos"]}:
        return False
    extremo = str(rotula.get("ubicacion", {}).get("extremo", ""))
    if "extremos" in regla and extremo not in {str(x) for x in regla["extremos"]}:
        return False
    if "sentidos" in regla and sentido not in {str(x) for x in regla["sentidos"]}:
        return False
    return True


def aplicar_criterios_externos(
    contrato: Mapping,
    catalogo: Mapping,
    *,
    permitir_provisionales=False,
    politica_exceso_capacidad="error",
) -> dict:
    """Asigna exactamente un criterio a cada rótula y sentido de momento."""
    if politica_exceso_capacidad not in {"error", "advertir"}:
        raise ValueError("politica_exceso_capacidad debe ser 'error' o 'advertir'.")
    datos = validar_catalogo_criterios(
        catalogo, permitir_provisionales=permitir_provisionales
    )
    salida = deepcopy(dict(contrato))
    nodos = {int(k): np.asarray(v, dtype=float) for k, v in salida["nodos"].items()}
    elevaciones = np.asarray(sorted({float(xy[1]) for xy in nodos.values()}))
    elementos = {int(e["id"]): e for e in salida["elementos"]}
    advertencias = []
    cobertura = []
    for rotula in salida["rotulas"]:
        elemento = elementos[int(rotula["elemento"])]
        nivel = _nivel_elemento(elemento, nodos, elevaciones)
        por_sentido = {}
        for sentido in SENTIDOS:
            candidatos = [
                r for r in datos["criterios"]
                if _coincide(r["aplica_a"], rotula, elemento, nivel, sentido)
            ]
            if len(candidatos) != 1:
                raise ValueError(
                    f"Rótula {rotula['id']} ({sentido}): se esperaba un criterio y "
                    f"se encontraron {[x.get('id') for x in candidatos]}."
                )
            registro = candidatos[0]
            limites = {k: float(registro["theta_p_rad"][k]) for k in ("IO", "LS", "CP")}
            theta_u = float(rotula.get("ley_por_sentido", {}).get(
                sentido, rotula["ley"]
            )["theta_p_u_rad"])
            if limites["CP"] > theta_u + 1e-12:
                mensaje = (
                    f"{rotula['id']} ({sentido}): CP={limites['CP']:.6g} rad "
                    f"excede theta_p_u={theta_u:.6g} rad."
                )
                if politica_exceso_capacidad == "error":
                    raise ValueError(mensaje)
                advertencias.append(mensaje)
            criterio = {
                "referencia": str(registro["referencia"]),
                "componente": f"{elemento['tipo']} {elemento['seccion_id']}",
                "theta_p_rad": limites,
                "parametros": deepcopy(registro.get("parametros_calculo", {})),
                "supuestos_provisionales": deepcopy(registro.get("supuestos", [])),
                "estado": str(registro["estado"]),
                "condicion_gobernante": str(registro["condicion_gobernante"]),
                "fuente_calculo": str(registro["fuente_calculo"]),
                "criterio_externo_id": str(registro["id"]),
                "fuera_de_capacidad_calculada": [
                    k for k, valor in limites.items() if valor > theta_u + 1e-12
                ],
            }
            por_sentido[sentido] = criterio
            cobertura.append({
                "rotula": str(rotula["id"]), "sentido": sentido,
                "criterio_externo_id": str(registro["id"]),
            })
        rotula["criterios_por_sentido"] = por_sentido
        rotula["criterios_aceptacion"] = deepcopy(por_sentido["positivo"])
    salida["criterios_aceptacion_externos"] = {
        "norma": datos["norma"], "edicion": datos["edicion"],
        "responsable": datos.get("responsable", "no declarado"),
        "permitir_provisionales": bool(permitir_provisionales),
        "politica_exceso_capacidad": politica_exceso_capacidad,
        "cobertura": cobertura, "advertencias": advertencias,
    }
    return salida


__all__ = [
    "leer_catalogo_criterios", "validar_catalogo_criterios",
    "aplicar_criterios_externos",
]
