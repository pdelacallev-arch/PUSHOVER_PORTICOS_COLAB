"""Ley monotónica y estado mínimo de rótulas concentradas de la sesión 5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from sesion_05.modelo_portico import UbicacionRotula, normalizar_ubicacion_rotula


@dataclass(frozen=True)
class LeyRotulaPlastica:
    """Rama plástica bilineal; la elasticidad permanece en el elemento."""

    My_kgf_cm: float
    Mu_kgf_cm: float
    theta_p_u_rad: float
    Kp_kgf_cm_rad: float
    degradacion_residual: object = None

    def __post_init__(self):
        if self.My_kgf_cm <= 0 or self.Mu_kgf_cm < self.My_kgf_cm:
            raise ValueError("Se requiere 0 < My <= Mu.")
        if self.theta_p_u_rad <= 0 or self.Kp_kgf_cm_rad < 0:
            raise ValueError("theta_p_u debe ser positiva y Kp no negativa.")

    def respuesta_activada(self, theta_p_rad: float, signo_momento: float = 1.0):
        """Respuesta sobre la rama plástica ya activada.

        No representa la fase elástica previa. Si se excede theta_p_u se marca
        el estado como no admisible porque el contrato no define degradación.
        """
        theta = abs(float(theta_p_rad))
        signo = 1.0 if signo_momento >= 0 else -1.0
        if theta > self.theta_p_u_rad:
            return {
                "M_kgf_cm": np.nan,
                "K_t_kgf_cm_rad": np.nan,
                "estado": "fuera_de_capacidad",
                "admisible": False,
            }
        M = min(self.My_kgf_cm + self.Kp_kgf_cm_rad * theta, self.Mu_kgf_cm)
        kt = self.Kp_kgf_cm_rad if M < self.Mu_kgf_cm else 0.0
        return {
            "M_kgf_cm": signo * M,
            "K_t_kgf_cm_rad": kt,
            "estado": "plastica",
            "admisible": True,
        }


def ley_desde_contrato_sesion03(contrato: Mapping):
    """Construye la ley plástica sin agregar flexibilidad elástica."""
    r = contrato["rotula"]
    requeridos = [
        "My_kgf_cm", "Mu_kgf_cm", "theta_p_u_rad",
        "Kp_promedio_kgf_cm_rad", "degradacion_residual",
    ]
    faltantes = [k for k in requeridos if k not in r]
    if faltantes:
        raise ValueError(f"Faltan campos de rótula: {faltantes}")
    return LeyRotulaPlastica(
        My_kgf_cm=float(r["My_kgf_cm"]),
        Mu_kgf_cm=float(r["Mu_kgf_cm"]),
        theta_p_u_rad=float(r["theta_p_u_rad"]),
        Kp_kgf_cm_rad=float(r["Kp_promedio_kgf_cm_rad"]),
        degradacion_residual=r["degradacion_residual"],
    )


def crear_rotula_concentrada(
    id_rotula: str,
    id_elemento: int,
    ubicacion: UbicacionRotula,
    ley: LeyRotulaPlastica,
    criterios_aceptacion: Mapping | None = None,
):
    """Crea un registro serializable para análisis y trazabilidad."""
    return {
        "id": str(id_rotula),
        "elemento": int(id_elemento),
        "ubicacion": asdict(ubicacion),
        "ley": asdict(ley),
        "criterios_aceptacion": dict(criterios_aceptacion or {}),
        "estado_inicial": {
            "fase": "elastica_inactiva",
            "theta_p_rad": 0.0,
            "nota": "sin resorte elástico adicional; EI_sec permanece en el elemento",
        },
    }


def clasificar_desempeno(theta_p_rad: float, criterios_aceptacion: Mapping):
    """Clasifica demanda plástica por límites IO, LS y CP ordenados."""
    limites = criterios_aceptacion.get("theta_p_rad", criterios_aceptacion)
    io, ls, cp = (float(limites[k]) for k in ("IO", "LS", "CP"))
    if not (0 <= io < ls < cp):
        raise ValueError("Los límites deben satisfacer 0 <= IO < LS < CP.")
    demanda = abs(float(theta_p_rad))
    if demanda <= io:
        return "IO"
    if demanda <= ls:
        return "LS"
    if demanda <= cp:
        return "CP"
    return "excede_CP"


def _valor_por_tipo(valor, tipo, nombre):
    if isinstance(valor, Mapping) and ({"columna", "viga"} & set(valor)):
        if tipo not in valor:
            raise KeyError(f"Falta {nombre} para el tipo {tipo!r}.")
        return valor[tipo]
    return valor


def asignar_rotulas_extremos(
    nodos,
    elementos: Sequence[Mapping],
    leyes,
    criterios_aceptacion=None,
    extremos_por_tipo=None,
    distancia_desde_cara_cm=0.0,
):
    """Asigna rótulas potenciales en extremos sin activarlas en la rigidez.

    `leyes` y `criterios_aceptacion` pueden ser un único valor o diccionarios
    con claves ``columna`` y ``viga``. Por defecto se asignan los extremos i y
    j de ambos tipos. La lista resultante es el registro para el algoritmo
    incremental; no agrega todavía ``rotulas_tangentes`` a los elementos.
    """
    if extremos_por_tipo is None:
        extremos_por_tipo = {"columna": ("i", "j"), "viga": ("i", "j")}
    d = float(distancia_desde_cara_cm)
    if d < 0:
        raise ValueError("La distancia desde la cara no puede ser negativa.")

    rotulas = []
    ids = set()
    for e in elementos:
        tipo = str(e["tipo"])
        extremos = tuple(extremos_por_tipo.get(tipo, ()))
        ley = _valor_por_tipo(leyes, tipo, "ley de rótula")
        criterios = (
            {} if criterios_aceptacion is None
            else _valor_por_tipo(criterios_aceptacion, tipo, "criterios de aceptación")
        )
        i, j = int(e["i"]), int(e["j"])
        L = float(np.linalg.norm(np.asarray(nodos[j]) - np.asarray(nodos[i])))
        ai = float(e.get("brazo_i_cm", 0.0))
        aj = float(e.get("brazo_j_cm", 0.0))
        for extremo in extremos:
            if extremo not in {"i", "j"}:
                raise ValueError("Cada extremo debe ser 'i' o 'j'.")
            ubicacion = normalizar_ubicacion_rotula(
                L, ai, aj, extremo, d, "cara_apoyo"
            )
            rid = f"E{int(e['id'])}-{extremo}"
            if rid in ids:
                raise ValueError(f"Identificador de rótula duplicado: {rid}.")
            ids.add(rid)
            registro = crear_rotula_concentrada(
                rid, int(e["id"]), ubicacion, ley, criterios
            )
            registro["tipo_elemento"] = tipo
            registro["nodo"] = i if extremo == "i" else j
            rotulas.append(registro)
    return rotulas


def activar_rotulas_tangentes(elementos, rotulas, ids_activas=None):
    """Devuelve copias de elementos con resortes tangentes para rótulas activas."""
    activos = None if ids_activas is None else {str(rid) for rid in ids_activas}
    por_elemento = {}
    for r in rotulas:
        if activos is not None and str(r["id"]) not in activos:
            continue
        por_elemento.setdefault(int(r["elemento"]), []).append({
            "id_rotula": str(r["id"]),
            "x_cm": float(r["ubicacion"]["x_desde_eje_i_cm"]),
            "k_theta_kgf_cm_rad": float(r["ley"]["Kp_kgf_cm_rad"]),
        })

    salida = []
    for e in elementos:
        copia = dict(e)
        tangentes = por_elemento.get(int(e["id"]), [])
        if tangentes:
            copia["rotulas_tangentes"] = tangentes
        else:
            copia.pop("rotulas_tangentes", None)
        salida.append(copia)
    return salida


__all__ = [
    "LeyRotulaPlastica", "ley_desde_contrato_sesion03",
    "crear_rotula_concentrada", "clasificar_desempeno",
    "asignar_rotulas_extremos", "activar_rotulas_tangentes",
]
