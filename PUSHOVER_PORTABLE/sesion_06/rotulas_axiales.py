"""Bibliotecas M-theta(P) y asignación por sección para el pushover de nivel 1."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

import numpy as np


_CAMPOS_LEY = ("My_kgf_cm", "Mu_kgf_cm", "theta_p_u_rad", "Kp_kgf_cm_rad")


def _nivel_elemento(elemento, nodos, elevaciones):
    yi = float(nodos[int(elemento["i"])][1])
    yj = float(nodos[int(elemento["j"])][1])
    return int(np.argmin(np.abs(elevaciones - max(yi, yj))))


def asignar_secciones_por_reglas(contrato: Mapping, reglas: Sequence[Mapping]):
    """Devuelve ``{id_elemento: seccion_id}`` usando reglas auditables.

    Cada regla admite ``portico``, ``tipo``, ``niveles`` e ``ids_elementos``.
    Los filtros omitidos actúan como comodines. Cada elemento debe coincidir
    con exactamente una sección; coincidencias con secciones distintas fallan.
    """
    nodos = {int(k): np.asarray(v, dtype=float) for k, v in contrato["nodos"].items()}
    elevaciones = np.asarray(sorted({float(xy[1]) for xy in nodos.values()}))
    salida = {}
    for e in contrato["elementos"]:
        eid = int(e["id"])
        nivel = _nivel_elemento(e, nodos, elevaciones)
        coincidencias = []
        for regla in reglas:
            if "seccion_id" not in regla:
                raise KeyError("Toda regla necesita seccion_id.")
            if "portico" in regla and str(regla["portico"]) != str(e["id_portico"]):
                continue
            if "tipo" in regla and str(regla["tipo"]) != str(e["tipo"]):
                continue
            if "niveles" in regla and nivel not in {int(x) for x in regla["niveles"]}:
                continue
            if "ids_elementos" in regla and eid not in {int(x) for x in regla["ids_elementos"]}:
                continue
            coincidencias.append(str(regla["seccion_id"]))
        if not coincidencias:
            raise ValueError(f"Elemento {eid}: ninguna regla asigna una sección.")
        if len(set(coincidencias)) != 1:
            raise ValueError(f"Elemento {eid}: reglas contradictorias {coincidencias}.")
        salida[eid] = coincidencias[0]
    return salida


def _interpolar_numero(a, b, t):
    return float(a) + float(t) * (float(b) - float(a))


def interpolar_ley_axial(
    familia: Sequence[Mapping],
    P_kgf: float,
    *,
    fuera_de_rango: str = "error",
):
    """Interpola linealmente una familia de leyes ordenada por compresión P."""
    if fuera_de_rango not in {"error", "limitar"}:
        raise ValueError("fuera_de_rango debe ser 'error' o 'limitar'.")
    puntos = sorted((deepcopy(dict(x)) for x in familia), key=lambda x: float(x["P_kgf"]))
    if len(puntos) < 1:
        raise ValueError("La familia de rótulas está vacía.")
    Ps = np.array([float(x["P_kgf"]) for x in puntos])
    if np.any(np.diff(Ps) <= 0):
        raise ValueError("Los niveles P_kgf deben ser estrictamente crecientes.")
    P_solicitado = float(P_kgf)
    P = P_solicitado
    limitado = False
    if P < Ps[0] or P > Ps[-1]:
        if fuera_de_rango == "error":
            raise ValueError(
                f"P={P:.3f} kgf fuera de la biblioteca [{Ps[0]:.3f}, {Ps[-1]:.3f}]."
            )
        P = float(np.clip(P, Ps[0], Ps[-1]))
        limitado = True
    if len(puntos) == 1 or np.isclose(P, Ps[0]):
        inferior = superior = puntos[0]
        t = 0.0
    elif np.isclose(P, Ps[-1]):
        inferior = superior = puntos[-1]
        t = 0.0
    else:
        j = int(np.searchsorted(Ps, P))
        inferior, superior = puntos[j - 1], puntos[j]
        t = (P - float(inferior["P_kgf"])) / (
            float(superior["P_kgf"]) - float(inferior["P_kgf"])
        )
    for punto in (inferior, superior):
        faltantes = set(_CAMPOS_LEY) - set(punto["ley"])
        if faltantes:
            raise KeyError(f"Ley axial incompleta: faltan {sorted(faltantes)}.")
    ley = {
        campo: _interpolar_numero(inferior["ley"][campo], superior["ley"][campo], t)
        for campo in _CAMPOS_LEY
    }
    ley["degradacion_residual"] = None
    if not (0 < ley["My_kgf_cm"] <= ley["Mu_kgf_cm"] and ley["theta_p_u_rad"] > 0
            and ley["Kp_kgf_cm_rad"] >= 0):
        raise ValueError("La interpolación produjo una ley no física.")

    ci = inferior.get("criterios_aceptacion", {})
    cs = superior.get("criterios_aceptacion", {})
    nombres = ("IO", "LS", "CP")
    completos = all(
        nombre in ci.get("theta_p_rad", {}) and nombre in cs.get("theta_p_rad", {})
        for nombre in nombres
    )
    if completos:
        limites = {
            nombre: _interpolar_numero(
                ci["theta_p_rad"][nombre], cs["theta_p_rad"][nombre], t
            ) for nombre in nombres
        }
        if not (0 <= limites["IO"] < limites["LS"] < limites["CP"]):
            raise ValueError("La interpolación produjo límites IO-LS-CP no ordenados.")
        criterios = {
            "referencia": ci.get("referencia", "biblioteca M-theta(P)"),
            "componente": ci.get("componente", "por definir"),
            "theta_p_rad": limites,
            "parametros": {
                "P_solicitado_kgf": P_solicitado, "P_usado_kgf": P,
                "P_inferior_kgf": float(inferior["P_kgf"]),
                "P_superior_kgf": float(superior["P_kgf"]),
                "factor_interpolacion": float(t),
            },
            "supuestos_provisionales": list(ci.get("supuestos_provisionales", [])),
            "estado": "interpolado", "fuera_de_capacidad_calculada": [],
        }
    else:
        criterios = {
            "referencia": "catálogo externo requerido",
            "componente": "por asignar", "theta_p_rad": {}, "parametros": {},
            "supuestos_provisionales": [], "estado": "pendiente_catalogo_externo",
            "fuera_de_capacidad_calculada": [],
        }
    EI = None
    if "EI_sec_kgf_cm2" in inferior and "EI_sec_kgf_cm2" in superior:
        EI = _interpolar_numero(
            inferior["EI_sec_kgf_cm2"], superior["EI_sec_kgf_cm2"], t
        )
    return {
        "P_solicitado_kgf": P_solicitado, "P_usado_kgf": P,
        "P_min_kgf": float(Ps[0]), "P_max_kgf": float(Ps[-1]),
        "limitado_al_rango": limitado, "ley": ley,
        "criterios_aceptacion": criterios, "EI_sec_kgf_cm2": EI,
    }


def actualizar_contrato_por_axial_gravedad(
    contrato: Mapping,
    fuerzas_elementos_gravedad: Mapping,
    asignacion_secciones: Mapping[int, str],
    biblioteca_rotulas: Mapping,
    *,
    fuera_de_rango: str = "error",
    actualizar_EI: bool = True,
    catalogo_secciones: Mapping | None = None,
):
    """Crea contrato 3.5 con leyes positiva/negativa seleccionadas para Pg."""
    salida = deepcopy(dict(contrato))
    fuerzas = {int(k): v for k, v in fuerzas_elementos_gravedad.items()}
    elementos = {int(e["id"]): e for e in salida["elementos"]}
    for eid, e in elementos.items():
        if eid not in asignacion_secciones:
            raise KeyError(f"Elemento {eid}: falta asignación de sección.")
        if eid not in fuerzas:
            raise KeyError(f"Elemento {eid}: faltan fuerzas de gravedad.")
        sid = str(asignacion_secciones[eid])
        if sid not in biblioteca_rotulas:
            raise KeyError(f"No existe biblioteca para la sección {sid!r}.")
        e["seccion_id"] = sid
        e["P_gravedad_kgf"] = float(fuerzas[eid]["P_compresion_kgf"])
        limites_sentidos = [
            (
                min(float(x["P_kgf"]) for x in biblioteca_rotulas[sid][sentido]),
                max(float(x["P_kgf"]) for x in biblioteca_rotulas[sid][sentido]),
            )
            for sentido in ("positivo", "negativo")
        ]
        e["P_biblioteca_min_kgf"] = max(x[0] for x in limites_sentidos)
        e["P_biblioteca_max_kgf"] = min(x[1] for x in limites_sentidos)

    por_elemento = {}
    for r in salida["rotulas"]:
        eid = int(r["elemento"])
        e, P = elementos[eid], float(fuerzas[eid]["P_compresion_kgf"])
        sid = str(e["seccion_id"])
        familia = biblioteca_rotulas[sid]
        selecciones = {
            sentido: interpolar_ley_axial(
                familia[sentido], P, fuera_de_rango=fuera_de_rango
            )
            for sentido in ("positivo", "negativo")
        }
        r["seccion_id"] = sid
        r["id_portico"] = str(e["id_portico"])
        r["P_referencia_kgf"] = P
        r["ley_por_sentido"] = {s: x["ley"] for s, x in selecciones.items()}
        r["criterios_por_sentido"] = {
            s: x["criterios_aceptacion"] for s, x in selecciones.items()
        }
        r["seleccion_axial"] = {
            s: {
                "P_solicitado_kgf": x["P_solicitado_kgf"],
                "P_usado_kgf": x["P_usado_kgf"],
                "limitado_al_rango": x["limitado_al_rango"],
            }
            for s, x in selecciones.items()
        }
        # Compatibilidad con consumidores 3.4: la rama positiva queda en los
        # campos históricos; el solver 3.5 selecciona el sentido al activarse.
        r["ley"] = deepcopy(selecciones["positivo"]["ley"])
        r["criterios_aceptacion"] = deepcopy(
            selecciones["positivo"]["criterios_aceptacion"]
        )
        por_elemento.setdefault(eid, selecciones)

    if actualizar_EI:
        for eid, selecciones in por_elemento.items():
            valores = [x["EI_sec_kgf_cm2"] for x in selecciones.values()]
            valores = [x for x in valores if x is not None]
            if valores:
                elementos[eid]["EI"] = float(np.mean(valores))
                elementos[eid]["EI_origen"] = "biblioteca_M_theta_Pg"
    salida["schema_version"] = "3.5"
    salida["secciones"] = deepcopy(dict(catalogo_secciones or {}))
    salida["biblioteca_rotulas"] = deepcopy(dict(biblioteca_rotulas))
    salida["calibracion_axial"] = {
        "metodo": "nivel_1_ley_M_theta_fija_en_P_gravedad",
        "fuera_de_rango": fuera_de_rango,
        "EI_actualizado_desde_biblioteca": bool(actualizar_EI),
    }
    return salida


def calibrar_contrato_nivel1(
    contrato: Mapping,
    caso_gravitacional,
    asignacion_secciones: Mapping[int, str],
    biblioteca_rotulas: Mapping,
    *,
    configuracion=None,
    fuera_de_rango: str = "error",
    actualizar_EI: bool = True,
    tolerancia_axial_fraccion: float = 0.01,
    max_iteraciones: int = 6,
    catalogo_secciones: Mapping | None = None,
):
    """Itera gravedad -> Pg -> M-theta(Pg) hasta estabilizar los axiales."""
    if tolerancia_axial_fraccion <= 0 or max_iteraciones < 1:
        raise ValueError("La tolerancia debe ser positiva y max_iteraciones >= 1.")
    from sesion_06.pushover import resolver_estado_gravitacional

    actual = deepcopy(dict(contrato))
    P_anterior = None
    historia = []
    gravedad = None
    for iteracion in range(1, max_iteraciones + 1):
        gravedad = resolver_estado_gravitacional(
            actual, caso_gravitacional, configuracion=configuracion
        )
        fuerzas = {int(k): v for k, v in gravedad["fuerzas_elementos"].items()}
        P_actual = {
            int(e["id"]): float(fuerzas[int(e["id"])]["P_compresion_kgf"])
            for e in actual["elementos"] if str(e["tipo"]) == "columna"
        }
        if P_anterior is None:
            cambio = np.inf
        else:
            cambio = max(
                abs(P_actual[eid] - P_anterior[eid]) / max(abs(P_anterior[eid]), 1.0)
                for eid in P_actual
            )
        actual = actualizar_contrato_por_axial_gravedad(
            actual, fuerzas, asignacion_secciones, biblioteca_rotulas,
            fuera_de_rango=fuera_de_rango, actualizar_EI=actualizar_EI,
            catalogo_secciones=catalogo_secciones,
        )
        historia.append({
            "iteracion": iteracion,
            "cambio_axial_maximo_fraccion": None if not np.isfinite(cambio) else float(cambio),
            "rotulas_activas_gravedad": sum(
                x["activa"] for x in gravedad["estado_rotulas"].values()
            ),
        })
        if P_anterior is not None and cambio <= tolerancia_axial_fraccion:
            return {
                "convergio": True, "contrato": actual,
                "estado_gravitacional": gravedad, "historia_calibracion": historia,
            }
        P_anterior = P_actual
    return {
        "convergio": False, "contrato": actual,
        "estado_gravitacional": gravedad, "historia_calibracion": historia,
    }


__all__ = [
    "asignar_secciones_por_reglas", "interpolar_ley_axial",
    "actualizar_contrato_por_axial_gravedad", "calibrar_contrato_nivel1",
]
