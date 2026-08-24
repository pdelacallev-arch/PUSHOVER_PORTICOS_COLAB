"""Construcción trazable de cargas gravitacionales para pórticos 2D.

Convención de entrada: las intensidades ``q`` son positivas hacia abajo. Las
cargas superficiales se expresan en kgf/cm², los anchos tributarios en cm y
las cargas lineales en kgf/cm. El resultado separa cargas nodales y fuerzas
de extremo fijo locales para que el solver conserve los momentos de gravedad.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class CasoGravitacional:
    """Acciones gravitacionales combinadas y listas para el ensamblaje."""

    vector_nodal: np.ndarray
    extremo_fijo_local: dict[int, np.ndarray]
    cargas_distribuidas: dict[int, dict]
    factores_combinacion: dict[str, float]
    descripcion: str
    supuestos: tuple[str, ...] = ()

    def como_dict(self):
        return {
            "vector_nodal": np.asarray(self.vector_nodal, dtype=float),
            "extremo_fijo_local": {
                int(k): np.asarray(v, dtype=float)
                for k, v in self.extremo_fijo_local.items()
            },
            "cargas_distribuidas": self.cargas_distribuidas,
            "factores_combinacion": self.factores_combinacion,
            "descripcion": self.descripcion,
            "supuestos": list(self.supuestos),
        }


def vector_carga_uniforme_local(q_hacia_abajo_kgf_cm: float, L_cm: float):
    """Carga nodal consistente y fuerza de extremo fijo de una viga horizontal.

    Devuelve ``(p_equivalente, f_extremo_fijo)`` en el orden
    ``[N_i, V_i, M_i, N_j, V_j, M_j]``. El eje local ``y`` es positivo hacia
    arriba para vigas orientadas de izquierda a derecha.
    """
    q, L = float(q_hacia_abajo_kgf_cm), float(L_cm)
    if not np.isfinite(q) or q < 0 or not np.isfinite(L) or L <= 0:
        raise ValueError("Se requieren q >= 0 y L > 0 finitos.")
    qy = -q
    p = np.array([0.0, qy * L / 2, qy * L**2 / 12,
                  0.0, qy * L / 2, -qy * L**2 / 12])
    return p, -p


def _valor_nivel(datos, portico, nivel, nombre):
    try:
        por_portico = datos[portico]
        return por_portico[nivel] if nivel in por_portico else por_portico[str(nivel)]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"Falta {nombre} para {portico}, nivel {nivel}.") from exc


def crear_caso_gravitacional_por_portico(
    contrato: Mapping,
    cargas_area_kgf_cm2: Mapping,
    anchos_tributarios_cm: Mapping,
    factores_combinacion: Mapping[str, float],
    *,
    cargas_lineales_kgf_cm: Mapping | None = None,
    cargas_area_por_elemento_kgf_cm2: Mapping | None = None,
    cargas_lineales_por_elemento_kgf_cm: Mapping | None = None,
    cargas_nodales: Mapping[int, Mapping[str, float] | float] | None = None,
    descripcion: str = "combinación gravitacional definida por el usuario",
    supuestos=(),
) -> CasoGravitacional:
    """Asigna cargas por área a cada viga según pórtico, nivel y ancho tributario.

    ``cargas_area_kgf_cm2`` tiene la forma
    ``{portico: {nivel: {caso: q_area}}}``. Los casos se combinan con
    ``factores_combinacion``. Las cargas lineales opcionales tienen la misma
    estructura. Las vigas deben ser horizontales; los brazos rígidos se
    excluyen de la longitud cargada.
    """
    factores = {str(k): float(v) for k, v in factores_combinacion.items()}
    if not factores or any(not np.isfinite(v) or v < 0 for v in factores.values()):
        raise ValueError("Los factores de combinación deben ser no negativos y finitos.")
    nodos = {int(k): np.asarray(v, dtype=float) for k, v in contrato["nodos"].items()}
    gdl = {int(k): np.asarray(v, dtype=int) for k, v in contrato["gdl"].items()}
    elevaciones = sorted({float(xy[1]) for xy in nodos.values()})
    n_gdl = 3 * len(nodos)
    P = np.zeros(n_gdl)
    fijos: dict[int, np.ndarray] = {}
    registros: dict[int, dict] = {}

    for e in contrato["elementos"]:
        if str(e["tipo"]) != "viga":
            continue
        eid, pid = int(e["id"]), str(e["id_portico"])
        i, j = int(e["i"]), int(e["j"])
        if not np.isclose(nodos[i][1], nodos[j][1]):
            raise ValueError(f"Elemento {eid}: la carga por área requiere una viga horizontal.")
        nivel = int(np.argmin(np.abs(np.asarray(elevaciones) - nodos[i][1])))
        if nivel == 0:
            raise ValueError(f"Elemento {eid}: se identificó una viga en el nivel base.")
        if cargas_area_por_elemento_kgf_cm2 is None:
            componentes_area = _valor_nivel(
                cargas_area_kgf_cm2, pid, nivel, "carga de área"
            )
        else:
            componentes_area = cargas_area_por_elemento_kgf_cm2.get(
                eid, cargas_area_por_elemento_kgf_cm2.get(str(eid))
            )
            if componentes_area is None:
                raise KeyError(f"Falta carga de área para el elemento {eid}.")
        ancho_dato = anchos_tributarios_cm[pid]
        if isinstance(ancho_dato, Mapping):
            ancho = float(ancho_dato[nivel] if nivel in ancho_dato else ancho_dato[str(nivel)])
        else:
            ancho = float(ancho_dato)
        if ancho <= 0:
            raise ValueError(f"Ancho tributario no positivo para {pid}, nivel {nivel}.")
        if cargas_lineales_por_elemento_kgf_cm is not None:
            componentes_linea = cargas_lineales_por_elemento_kgf_cm.get(
                eid, cargas_lineales_por_elemento_kgf_cm.get(str(eid))
            )
            if componentes_linea is None:
                raise KeyError(f"Falta carga lineal para el elemento {eid}.")
        else:
            componentes_linea = (
                {} if cargas_lineales_kgf_cm is None
                else _valor_nivel(cargas_lineales_kgf_cm, pid, nivel, "carga lineal")
            )
        q_componentes = {}
        for caso, factor in factores.items():
            qa = float(componentes_area.get(caso, 0.0))
            ql = float(componentes_linea.get(caso, 0.0))
            if qa < 0 or ql < 0:
                raise ValueError("Las cargas gravitacionales de entrada deben ser no negativas.")
            q_componentes[caso] = qa * ancho + ql
        q_total = sum(factores[c] * q for c, q in q_componentes.items())
        L_ejes = float(np.linalg.norm(nodos[j] - nodos[i]))
        Lf = L_ejes - float(e.get("brazo_i_cm", 0.0)) - float(e.get("brazo_j_cm", 0.0))
        _, f0 = vector_carga_uniforme_local(q_total, Lf)
        fijos[eid] = f0
        registros[eid] = {
            "portico": pid, "nivel": nivel, "q_total_kgf_cm": q_total,
            "q_por_caso_kgf_cm": q_componentes, "L_cargada_cm": Lf,
        }

    for nid_raw, componentes in (cargas_nodales or {}).items():
        nid = int(nid_raw)
        if nid not in gdl:
            raise KeyError(f"Carga nodal aplicada a nodo inexistente: {nid}.")
        if isinstance(componentes, Mapping):
            carga = sum(factores.get(str(caso), 0.0) * float(valor)
                        for caso, valor in componentes.items())
        else:
            carga = float(componentes)
        if carga < 0:
            raise ValueError("Las cargas nodales se ingresan positivas hacia abajo.")
        P[gdl[nid][1]] -= carga

    if not fijos and not np.any(P):
        raise ValueError("El caso gravitacional no contiene acciones aplicadas.")
    return CasoGravitacional(
        vector_nodal=P,
        extremo_fijo_local=fijos,
        cargas_distribuidas=registros,
        factores_combinacion=factores,
        descripcion=str(descripcion),
        supuestos=tuple(str(x) for x in supuestos),
    )


__all__ = [
    "CasoGravitacional", "vector_carga_uniforme_local",
    "crear_caso_gravitacional_por_portico",
]
