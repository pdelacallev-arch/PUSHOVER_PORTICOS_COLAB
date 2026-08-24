"""Pushover monotónico de una serie coplanar de pórticos 2D.

El solucionador usa Newton--Raphson con control directo del desplazamiento
promedio de los nodos de techo. Las incógnitas de cada iteración son los GDL
libres y el factor del patrón lateral. La no linealidad se concentra en
resortes rotacionales ubicados en las caras de los nudos.

Unidades del contrato: kgf, cm y radianes. El análisis es de primer orden; no
incluye P-Delta, interacción P-M ni comprobación frágil por cortante.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from sesion_04.rigidez_portico import (
    matriz_rigidez_local_portico2d,
    matriz_transformacion_portico2d,
)
from sesion_05.modelo_portico import matriz_brazos_rigidos_local
from sesion_05.rotulas import clasificar_desempeno
from sesion_06.cargas_gravedad import CasoGravitacional


@dataclass(frozen=True)
class ConfiguracionPushover:
    """Controles numéricos y criterios explícitos de parada."""

    incremento_control_cm: float = 0.5
    desplazamiento_maximo_cm: float = 30.0
    max_pasos: int = 100
    max_iteraciones: int = 30
    subpasos_gravedad: int = 10
    tolerancia_equilibrio: float = 1.0e-7
    tolerancia_control_cm: float = 1.0e-9
    tolerancia_activacion: float = 1.0e-8
    condicion_maxima: float = 1.0e16
    perdida_resistencia_fraccion: float = 0.20
    variacion_axial_alerta_fraccion: float = 0.20
    detener_fuera_biblioteca_axial: bool = True

    def __post_init__(self):
        if self.incremento_control_cm <= 0 or self.desplazamiento_maximo_cm <= 0:
            raise ValueError("Los desplazamientos de control deben ser positivos.")
        if self.max_pasos < 1 or self.max_iteraciones < 1 or self.subpasos_gravedad < 1:
            raise ValueError("max_pasos, max_iteraciones y subpasos_gravedad deben ser positivos.")
        if self.tolerancia_equilibrio <= 0 or self.tolerancia_control_cm <= 0:
            raise ValueError("Las tolerancias deben ser positivas.")
        if not 0 < self.perdida_resistencia_fraccion < 1:
            raise ValueError("La fracción de pérdida debe estar entre 0 y 1.")
        if self.variacion_axial_alerta_fraccion <= 0:
            raise ValueError("La alerta de variación axial debe ser positiva.")


def leer_contrato_sesion05(ruta: str | Path) -> dict:
    """Lee y valida los campos estructurales mínimos del contrato 3.4."""
    ruta = Path(ruta)
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    requeridos = {
        "schema_version", "modelo", "geometria", "nodos", "gdl",
        "elementos", "puntales", "rotulas", "restricciones_gdl",
    }
    faltantes = requeridos - set(datos)
    if faltantes:
        raise ValueError(f"Contrato incompleto; faltan {sorted(faltantes)}.")
    if str(datos["schema_version"]) not in {"3.4", "3.5"}:
        raise ValueError("La sesión 06 requiere schema_version 3.4 o 3.5.")
    if datos["modelo"] != "edificio_porticos_2d_en_serie":
        raise ValueError("El contrato no corresponde al edificio 2D en serie.")
    if not datos["nodos"] or not datos["elementos"] or not datos["rotulas"]:
        raise ValueError("El contrato no contiene geometría, elementos y rótulas.")
    return datos


def _mapas_modelo(contrato: Mapping):
    nodos = {int(k): np.asarray(v, dtype=float) for k, v in contrato["nodos"].items()}
    gdl = {int(k): np.asarray(v, dtype=int) for k, v in contrato["gdl"].items()}
    if set(nodos) != set(gdl):
        raise ValueError("Los mapas de nodos y GDL no contienen las mismas claves.")
    return nodos, gdl


def _nodos_por_nivel(contrato: Mapping):
    nodos, _ = _mapas_modelo(contrato)
    elevaciones = sorted({float(xy[1]) for xy in nodos.values()})
    return {
        nivel: [nid for nid, xy in nodos.items() if np.isclose(xy[1], y)]
        for nivel, y in enumerate(elevaciones)
    }, np.asarray(elevaciones)


def crear_patron_lateral(
    contrato: Mapping,
    coeficientes_nivel: Sequence[float],
    *,
    normalizar: bool = True,
) -> np.ndarray:
    """Crea un patrón horizontal explícito y lo reparte entre nudos de cada nivel.

    ``coeficientes_nivel`` contiene un valor por piso, sin incluir la base. Si
    se normaliza, la suma del patrón es 1 y el factor de carga tiene unidades
    de kgf, por lo que coincide con el empuje lateral aplicado.
    """
    nodos, gdl = _mapas_modelo(contrato)
    por_nivel, elevaciones = _nodos_por_nivel(contrato)
    coef = np.asarray(coeficientes_nivel, dtype=float)
    if coef.shape != (len(elevaciones) - 1,) or np.any(coef < 0) or not np.any(coef > 0):
        raise ValueError("Se requiere un coeficiente no negativo por piso y al menos uno positivo.")
    if normalizar:
        coef = coef / np.sum(coef)
    patron = np.zeros(3 * len(nodos))
    for nivel, valor in enumerate(coef, start=1):
        ids = por_nivel[nivel]
        for nid in ids:
            patron[gdl[nid][0]] = valor / len(ids)
    return patron


def crear_cargas_gravitacionales(
    contrato: Mapping,
    cargas_nivel_kgf: Sequence[float] | None,
) -> np.ndarray:
    """Distribuye cargas verticales de piso; valores positivos actúan hacia abajo."""
    nodos, gdl = _mapas_modelo(contrato)
    cargas = np.zeros(3 * len(nodos))
    if cargas_nivel_kgf is None:
        return cargas
    por_nivel, elevaciones = _nodos_por_nivel(contrato)
    valores = np.asarray(cargas_nivel_kgf, dtype=float)
    if valores.shape != (len(elevaciones) - 1,) or np.any(valores < 0):
        raise ValueError("Se requiere una carga gravitacional no negativa por piso.")
    for nivel, valor in enumerate(valores, start=1):
        ids = por_nivel[nivel]
        for nid in ids:
            cargas[gdl[nid][1]] = -valor / len(ids)
    return cargas


def _preparar_elementos(
    contrato: Mapping,
    nodos: Mapping[int, np.ndarray],
    extremo_fijo_local: Mapping[int, Sequence[float]] | None = None,
):
    rotulas_por_elemento: dict[int, list[dict]] = {}
    for rotula in contrato["rotulas"]:
        rotulas_por_elemento.setdefault(int(rotula["elemento"]), []).append(dict(rotula))

    preparados = []
    for original in contrato["elementos"]:
        e = dict(original)
        i, j = int(e["i"]), int(e["j"])
        delta = nodos[j] - nodos[i]
        L = float(np.linalg.norm(delta))
        ai, aj = float(e.get("brazo_i_cm", 0.0)), float(e.get("brazo_j_cm", 0.0))
        Lf = L - ai - aj
        if Lf <= 0:
            raise ValueError(f"Elemento {e['id']}: longitud flexible no positiva.")
        T = matriz_transformacion_portico2d(float(np.arctan2(delta[1], delta[0])))
        R = matriz_brazos_rigidos_local(ai, aj)
        kf = matriz_rigidez_local_portico2d(float(e["EA"]), float(e.get("eta", 1.0)) * float(e["EI"]), Lf)
        hinges = sorted(rotulas_por_elemento.get(int(e["id"]), []), key=lambda r: r["ubicacion"]["extremo"])
        if len(hinges) != 2 or {r["ubicacion"]["extremo"] for r in hinges} != {"i", "j"}:
            raise ValueError(f"Elemento {e['id']}: se requieren exactamente las rótulas i y j.")
        for r in hinges:
            x = float(r["ubicacion"]["x_desde_eje_i_cm"])
            esperado = ai if r["ubicacion"]["extremo"] == "i" else L - aj
            if not np.isclose(x, esperado, atol=1e-7):
                raise ValueError("La sesión 06 admite rótulas concentradas en las caras de los extremos.")
        f0 = np.asarray((extremo_fijo_local or {}).get(int(e["id"]), np.zeros(6)), dtype=float)
        if f0.shape != (6,) or not np.all(np.isfinite(f0)):
            raise ValueError(f"Elemento {e['id']}: fuerza de extremo fijo local inválida.")
        preparados.append({
            **e, "T": T, "R": R, "kf": kf, "rotulas": hinges,
            "f0_flexible_gravedad": f0,
        })
    return preparados


def _estado_inicial_rotulas(contrato: Mapping):
    return {
        str(r["id"]): {
            "activa": False,
            "theta_p_rad": 0.0,
            "direccion": 0.0,
            "sentido_momento": None,
            "momento_kgf_cm": 0.0,
            "desempeno": "elastica",
        }
        for r in contrato["rotulas"]
    }


def _ley_criterios_rotula(rotula, sentido):
    sentido = "positivo" if sentido == "positivo" else "negativo"
    ley = rotula.get("ley_por_sentido", {}).get(sentido, rotula["ley"])
    criterios = rotula.get("criterios_por_sentido", {}).get(
        sentido, rotula["criterios_aceptacion"]
    )
    return ley, criterios


def _respuesta_elemento(
    e, d_global, estado_entrada, tolerancia_activacion, factor_gravedad=1.0
):
    """Devuelve fuerza y tangente condensadas, más el estado local de rótulas."""
    d_ejes = e["T"] @ d_global
    d_cara = e["R"] @ d_ejes
    kf = e["kf"]
    B = np.zeros((6, 2))
    B[2, 0] = B[5, 1] = 1.0
    estados = [deepcopy(estado_entrada[str(r["id"])]) for r in e["rotulas"]]

    for _ in range(3):
        activos = [a for a, st in enumerate(estados) if st["activa"]]
        p = np.zeros(2)
        if activos:
            Ba = B[:, activos]
            leyes_activas = [
                _ley_criterios_rotula(
                    e["rotulas"][a], estados[a]["sentido_momento"]
                )[0]
                for a in activos
            ]
            kp = np.array([float(ley["Kp_kgf_cm_rad"]) for ley in leyes_activas])
            m0 = np.array([
                estados[a]["direccion"] * float(ley["My_kgf_cm"])
                for a, ley in zip(activos, leyes_activas)
            ])
            A = Ba.T @ kf @ Ba + np.diag(kp)
            f0 = factor_gravedad * e["f0_flexible_gravedad"]
            rhs = -(Ba.T @ (kf @ d_cara + f0) + m0)
            p[np.asarray(activos)] = np.linalg.solve(A, rhs)
        f_flexible = kf @ (d_cara + B @ p) + factor_gravedad * e["f0_flexible_gravedad"]
        momentos = np.array([f_flexible[2], f_flexible[5]])
        nuevas = []
        for a, (r, st) in enumerate(zip(e["rotulas"], estados)):
            sentido = "positivo" if momentos[a] >= 0 else "negativo"
            ley, _ = _ley_criterios_rotula(r, sentido)
            My = float(ley["My_kgf_cm"])
            if not st["activa"] and abs(momentos[a]) > My * (1.0 + tolerancia_activacion):
                st["activa"] = True
                st["direccion"] = -1.0 if momentos[a] >= 0 else 1.0
                st["sentido_momento"] = sentido
                nuevas.append(a)
        if not nuevas:
            break
    else:
        raise RuntimeError(f"Elemento {e['id']}: no convergió el conjunto activo local.")

    activos = [a for a, st in enumerate(estados) if st["activa"]]
    Kc = e["R"].T @ kf @ e["R"]
    if activos:
        Ba = B[:, activos]
        kp = np.array([
            float(_ley_criterios_rotula(
                e["rotulas"][a], estados[a]["sentido_momento"]
            )[0]["Kp_kgf_cm_rad"])
            for a in activos
        ])
        A = Ba.T @ kf @ Ba + np.diag(kp)
        Kc = e["R"].T @ (kf - kf @ Ba @ np.linalg.solve(A, Ba.T @ kf)) @ e["R"]

    f_ejes = e["R"].T @ f_flexible
    for a, (r, st) in enumerate(zip(e["rotulas"], estados)):
        st["theta_p_rad"] = float(p[a])
        st["momento_kgf_cm"] = float(momentos[a])
        if st["activa"]:
            _, criterios = _ley_criterios_rotula(r, st["sentido_momento"])
            st["desempeno"] = clasificar_desempeno(p[a], criterios)
    fuerzas = {
        "local_ejes": f_ejes.tolist(),
        "local_caras": f_flexible.tolist(),
        "P_compresion_kgf": float(0.5 * (f_flexible[0] - f_flexible[3])),
        "N_i_kgf": float(f_flexible[0]), "V_i_kgf": float(f_flexible[1]),
        "M_i_kgf_cm": float(f_flexible[2]), "N_j_kgf": float(f_flexible[3]),
        "V_j_kgf": float(f_flexible[4]), "M_j_kgf_cm": float(f_flexible[5]),
    }
    return (
        e["T"].T @ f_ejes,
        e["T"].T @ Kc @ e["T"],
        {str(r["id"]): st for r, st in zip(e["rotulas"], estados)},
        fuerzas,
    )


def _ensamblar(
    contrato, preparados, nodos, gdl, u, estado, tol_activacion,
    factor_gravedad=1.0,
):
    n = len(u)
    K = np.zeros((n, n))
    fint = np.zeros(n)
    estado_trial = deepcopy(estado)
    fuerzas_elementos = {}
    for e in preparados:
        vc = np.r_[gdl[int(e["i"])], gdl[int(e["j"])]]
        fe, ke, estados_e, fuerzas_e = _respuesta_elemento(
            e, u[vc], estado, tol_activacion, factor_gravedad
        )
        fint[vc] += fe
        K[np.ix_(vc, vc)] += ke
        estado_trial.update(estados_e)
        fuerzas_elementos[int(e["id"])] = fuerzas_e
    for puntal in contrato["puntales"]:
        gi, gj = gdl[int(puntal["i"])][0], gdl[int(puntal["j"])][0]
        k = float(puntal["k_axial_kgf_cm"])
        du = u[gi] - u[gj]
        fint[[gi, gj]] += k * np.array([du, -du])
        K[np.ix_([gi, gj], [gi, gj])] += k * np.array([[1.0, -1.0], [-1.0, 1.0]])
    return 0.5 * (K + K.T), fint, estado_trial, fuerzas_elementos


def _resolver_gravedad(contrato, preparados, nodos, gdl, libres, cargas, estado, cfg):
    u = np.zeros(3 * len(nodos))
    hay_fijos = any(np.any(np.abs(e["f0_flexible_gravedad"]) > 0) for e in preparados)
    if np.allclose(cargas, 0.0) and not hay_fijos:
        _, _, trial, fuerzas = _ensamblar(
            contrato, preparados, nodos, gdl, u, estado,
            cfg.tolerancia_activacion, 0.0,
        )
        return u, trial, [], fuerzas

    estado_commit = deepcopy(estado)
    historia = []
    fuerzas = {}
    escala_fijos = max(
        [float(np.max(np.abs(e["f0_flexible_gravedad"]))) for e in preparados]
        + [0.0]
    )
    for subpaso in range(1, cfg.subpasos_gravedad + 1):
        gamma = subpaso / cfg.subpasos_gravedad
        estado_base = deepcopy(estado_commit)
        residuo_rel = np.inf
        for it in range(1, cfg.max_iteraciones + 1):
            K, fint, trial, fuerzas = _ensamblar(
                contrato, preparados, nodos, gdl, u, estado_base,
                cfg.tolerancia_activacion, gamma,
            )
            for rid, st_trial in trial.items():
                if st_trial["activa"] and not estado_base[rid]["activa"]:
                    estado_base[rid]["activa"] = True
                    estado_base[rid]["direccion"] = st_trial["direccion"]
                    estado_base[rid]["sentido_momento"] = st_trial["sentido_momento"]
            r = gamma * cargas - fint
            escala = max(
                1.0, gamma * float(np.max(np.abs(cargas[libres]))),
                gamma * escala_fijos,
            )
            residuo_rel = float(np.max(np.abs(r[libres])) / escala)
            if residuo_rel <= cfg.tolerancia_equilibrio:
                estado_commit = trial
                reacciones = fint - gamma * cargas
                reaccion_vertical = float(sum(
                    reacciones[gdl[nid][1]]
                    for nid in contrato["geometria"]["nodos_base"]
                ))
                historia.append({
                    "subpaso": subpaso, "factor_gravedad": gamma,
                    "iteraciones": it, "residuo_relativo": residuo_rel,
                    "reaccion_vertical_base_kgf": reaccion_vertical,
                    "rotulas_activas": sum(st["activa"] for st in trial.values()),
                })
                break
            u[libres] += np.linalg.solve(K[np.ix_(libres, libres)], r[libres])
        else:
            raise RuntimeError(
                f"La etapa gravitacional no convergió en el subpaso {subpaso}."
            )
    return u, estado_commit, historia, fuerzas


def _nodos_por_portico_y_nivel(contrato, nodos):
    ids = list(contrato["geometria"]["ids_porticos"])
    resultado = {pid: {} for pid in ids}
    pertenencia: dict[int, str] = {}
    for e in contrato["elementos"]:
        pid = str(e["id_portico"])
        pertenencia[int(e["i"])] = pid
        pertenencia[int(e["j"])] = pid
    elevaciones = sorted({float(xy[1]) for xy in nodos.values()})
    for pid in ids:
        for nivel, y in enumerate(elevaciones):
            resultado[pid][nivel] = [
                nid for nid, xy in nodos.items()
                if pertenencia.get(nid) == pid and np.isclose(xy[1], y)
            ]
    return resultado, np.asarray(elevaciones)


def _calcular_derivas(contrato, nodos, gdl, u):
    grupos, elevaciones = _nodos_por_portico_y_nivel(contrato, nodos)
    salida = {}
    for pid, niveles in grupos.items():
        ux = np.array([np.mean([u[gdl[n][0]] for n in niveles[k]]) for k in range(len(elevaciones))])
        salida[pid] = (np.diff(ux) / np.diff(elevaciones)).tolist()
    return salida


def _control_validez_axial(contrato, fuerzas_elementos, umbral):
    """Compara el axial actual con Pg y con el rango de la biblioteca 3.5."""
    variaciones = {}
    fuera_rango = []
    alertas = []
    for e in contrato["elementos"]:
        if str(e["tipo"]) != "columna" or "P_gravedad_kgf" not in e:
            continue
        eid = int(e["id"])
        P = float(fuerzas_elementos[eid]["P_compresion_kgf"])
        Pg = float(e["P_gravedad_kgf"])
        variacion = abs(P - Pg) / max(abs(Pg), 1.0)
        variaciones[eid] = variacion
        if variacion > umbral:
            alertas.append(eid)
        if "P_biblioteca_min_kgf" in e and not (
            float(e["P_biblioteca_min_kgf"]) <= P <= float(e["P_biblioteca_max_kgf"])
        ):
            fuera_rango.append(eid)
    return {
        "variacion_axial_maxima_fraccion": max(variaciones.values(), default=0.0),
        "elementos_sobre_umbral": alertas,
        "elementos_fuera_biblioteca": fuera_rango,
    }


def _normalizar_caso_gravitacional(cargas_gravitacionales, n_gdl):
    extremo_fijo = {}
    datos = {
        "descripcion": "sin cargas gravitacionales",
        "factores_combinacion": {}, "cargas_distribuidas": {}, "supuestos": [],
    }
    if cargas_gravitacionales is None:
        Pg = np.zeros(n_gdl)
    elif isinstance(cargas_gravitacionales, CasoGravitacional):
        caso = cargas_gravitacionales.como_dict()
        Pg = np.asarray(caso["vector_nodal"], dtype=float)
        extremo_fijo = caso["extremo_fijo_local"]
        datos.update({
            k: caso[k] for k in (
                "descripcion", "factores_combinacion", "cargas_distribuidas", "supuestos"
            )
        })
    elif isinstance(cargas_gravitacionales, Mapping):
        Pg = np.asarray(cargas_gravitacionales["vector_nodal"], dtype=float)
        extremo_fijo = {
            int(k): np.asarray(v, dtype=float)
            for k, v in cargas_gravitacionales.get("extremo_fijo_local", {}).items()
        }
        datos.update({
            k: deepcopy(cargas_gravitacionales.get(k, datos[k])) for k in datos
        })
    else:
        Pg = np.asarray(cargas_gravitacionales, dtype=float)
        datos["descripcion"] = "vector de cargas nodales"
    if Pg.shape != (n_gdl,):
        raise ValueError("El vector gravitacional debe tener un valor por GDL global.")
    return Pg, extremo_fijo, datos


def resolver_estado_gravitacional(
    contrato: Mapping,
    cargas_gravitacionales: Sequence[float] | Mapping | CasoGravitacional | None,
    *,
    configuracion: ConfiguracionPushover | None = None,
):
    """Resuelve solo la gravedad y recupera N-V-M para calibrar las rótulas."""
    cfg = configuracion or ConfiguracionPushover()
    nodos, gdl = _mapas_modelo(contrato)
    n = 3 * len(nodos)
    Pg, extremo_fijo, datos = _normalizar_caso_gravitacional(
        cargas_gravitacionales, n
    )
    restringidos = np.unique(np.asarray(contrato["restricciones_gdl"], dtype=int))
    libres = np.setdiff1d(np.arange(n), restringidos)
    preparados = _preparar_elementos(contrato, nodos, extremo_fijo)
    estado = _estado_inicial_rotulas(contrato)
    u, estado, historia, fuerzas = _resolver_gravedad(
        contrato, preparados, nodos, gdl, libres, Pg, estado, cfg
    )
    return {
        "convergio": True, "caso_gravitacional": datos,
        "historia_gravedad": historia, "desplazamientos": u.tolist(),
        "estado_rotulas": estado, "fuerzas_elementos": fuerzas,
    }


def analizar_pushover(
    contrato: Mapping,
    patron_lateral: Sequence[float],
    *,
    cargas_gravitacionales: Sequence[float] | Mapping | CasoGravitacional | None = None,
    nodos_control: Sequence[int] | None = None,
    configuracion: ConfiguracionPushover | None = None,
) -> dict:
    """Ejecuta el pushover y devuelve historia, eventos, estados y criterio de parada."""
    cfg = configuracion or ConfiguracionPushover()
    nodos, gdl = _mapas_modelo(contrato)
    n = 3 * len(nodos)
    P = np.asarray(patron_lateral, dtype=float)
    Pg, extremo_fijo, datos_caso_gravedad = _normalizar_caso_gravitacional(
        cargas_gravitacionales, n
    )
    if P.shape != (n,):
        raise ValueError("El patrón lateral debe tener un valor por GDL global.")
    if not np.isclose(np.sum(P[0::3]), 1.0):
        raise ValueError("El patrón lateral debe normalizarse a una fuerza horizontal total unitaria.")
    restringidos = np.unique(np.asarray(contrato["restricciones_gdl"], dtype=int))
    libres = np.setdiff1d(np.arange(n), restringidos)
    techo = list(contrato["geometria"]["nodos_techo"] if nodos_control is None else nodos_control)
    if not techo or any(int(nid) not in nodos for nid in techo):
        raise ValueError("Los nodos de control no son válidos.")
    c = np.zeros(n)
    for nid in techo:
        c[gdl[int(nid)][0]] = 1.0 / len(techo)
    if np.allclose(c[libres], 0.0):
        raise ValueError("El desplazamiento de control no contiene un GDL libre.")

    preparados = _preparar_elementos(contrato, nodos, extremo_fijo)
    estado_elastico = _estado_inicial_rotulas(contrato)
    estado = deepcopy(estado_elastico)
    u, estado, historia_gravedad, fuerzas_gravedad = _resolver_gravedad(
        contrato, preparados, nodos, gdl, libres, Pg, estado, cfg
    )
    it_g = sum(x["iteraciones"] for x in historia_gravedad)
    res_g = historia_gravedad[-1]["residuo_relativo"] if historia_gravedad else 0.0
    lambda_lateral = 0.0
    desplazamiento_inicial = float(c @ u)
    rotulas_gravedad = [
        rid for rid, st in estado.items()
        if st["activa"] and not estado_elastico[rid]["activa"]
    ]
    control_axial_gravedad = _control_validez_axial(
        contrato, fuerzas_gravedad, cfg.variacion_axial_alerta_fraccion
    )
    historia = [{
        "paso": 0, "factor_carga_kgf": 0.0, "cortante_basal_kgf": 0.0,
        "desplazamiento_control_cm": desplazamiento_inicial,
        "deriva_maxima": 0.0, "iteraciones": it_g,
        "residuo_relativo": res_g, "convergencia": True,
        "eventos_no_lineales": rotulas_gravedad,
        "fuerzas_elementos": fuerzas_gravedad,
        "control_validez_axial": control_axial_gravedad,
    }]
    eventos = [
        {"paso": 0, "tipo": "formacion_rotula_gravedad", "rotula": rid}
        for rid in rotulas_gravedad
    ]
    razon_parada = "desplazamiento_maximo"
    convergio = True
    pico = 0.0

    for paso in range(1, cfg.max_pasos + 1):
        objetivo = desplazamiento_inicial + min(
            paso * cfg.incremento_control_cm, cfg.desplazamiento_maximo_cm
        )
        estado_base = deepcopy(estado)
        residuo_rel = np.inf
        estado_trial = deepcopy(estado)
        try:
            for it in range(1, cfg.max_iteraciones + 1):
                K, fint, estado_trial, fuerzas_trial = _ensamblar(
                    contrato, preparados, nodos, gdl, u, estado_base,
                    cfg.tolerancia_activacion, 1.0,
                )
                # El conjunto activo solo puede crecer durante un paso
                # monotónico. Esto evita alternar entre las ramas elástica y
                # plástica cuando una iteración cruza exactamente My.
                for rid, st_trial in estado_trial.items():
                    if st_trial["activa"] and not estado_base[rid]["activa"]:
                        estado_base[rid]["activa"] = True
                        estado_base[rid]["direccion"] = st_trial["direccion"]
                        estado_base[rid]["sentido_momento"] = st_trial["sentido_momento"]
                r = Pg + lambda_lateral * P - fint
                escala = max(1.0, abs(lambda_lateral), float(np.max(np.abs(Pg[libres]))))
                residuo_rel = float(np.max(np.abs(r[libres])) / escala)
                error_control = float(objetivo - c @ u)
                if residuo_rel <= cfg.tolerancia_equilibrio and abs(error_control) <= cfg.tolerancia_control_cm:
                    break
                Kff = K[np.ix_(libres, libres)]
                A = np.block([
                    [Kff, -P[libres, None]],
                    [c[libres][None, :], np.zeros((1, 1))],
                ])
                condicion = float(np.linalg.cond(A))
                if not np.isfinite(condicion) or condicion > cfg.condicion_maxima:
                    raise np.linalg.LinAlgError(f"Sistema aumentado mal condicionado: {condicion:.3e}.")
                incremento = np.linalg.solve(A, np.r_[r[libres], error_control])
                u[libres] += incremento[:-1]
                lambda_lateral += float(incremento[-1])
            else:
                raise RuntimeError("Se agotó el máximo de iteraciones.")
        except (np.linalg.LinAlgError, RuntimeError) as exc:
            convergio = False
            razon_parada = f"inestabilidad_o_no_convergencia: {exc}"
            break

        K, fint, estado_trial, fuerzas_trial = _ensamblar(
            contrato, preparados, nodos, gdl, u, estado_base,
            cfg.tolerancia_activacion, 1.0,
        )
        estado_anterior = estado
        estado = estado_trial
        reacciones = fint - Pg - lambda_lateral * P
        cortante = -float(sum(reacciones[gdl[nid][0]] for nid in contrato["geometria"]["nodos_base"]))
        derivas = _calcular_derivas(contrato, nodos, gdl, u)
        deriva_max = max(abs(x) for valores in derivas.values() for x in valores)
        control_axial = _control_validez_axial(
            contrato, fuerzas_trial, cfg.variacion_axial_alerta_fraccion
        )
        eventos_paso = []
        rotulas_por_id = {str(r["id"]): r for r in contrato["rotulas"]}
        for rid, st in estado.items():
            previo = estado_anterior[rid]
            rotula = rotulas_por_id[rid]
            if st["activa"] and not previo["activa"]:
                evento = {
                    "paso": paso, "tipo": "formacion_rotula", "rotula": rid,
                    "elemento": int(rotula["elemento"]),
                    "portico": str(next(e["id_portico"] for e in contrato["elementos"] if int(e["id"]) == int(rotula["elemento"]))),
                    "desplazamiento_control_cm": float(c @ u),
                    "cortante_basal_kgf": cortante,
                    "momento_kgf_cm": st["momento_kgf_cm"],
                }
                eventos.append(evento)
                eventos_paso.append(rid)
            if st["desempeno"] != previo["desempeno"] and st["activa"]:
                eventos.append({
                    "paso": paso, "tipo": "cambio_desempeno", "rotula": rid,
                    "desde": previo["desempeno"], "hacia": st["desempeno"],
                    "theta_p_rad": abs(st["theta_p_rad"]),
                })
        historia.append({
            "paso": paso, "factor_carga_kgf": lambda_lateral,
            "cortante_basal_kgf": cortante,
            "desplazamiento_control_cm": float(c @ u),
            "deriva_maxima": deriva_max, "derivas_por_portico": derivas,
            "iteraciones": it, "residuo_relativo": residuo_rel,
            "convergencia": True, "eventos_no_lineales": eventos_paso,
            "fuerzas_elementos": fuerzas_trial,
            "control_validez_axial": control_axial,
        })
        pico = max(pico, cortante)

        excedidas = []
        for rid, st in estado.items():
            if not st["activa"]:
                continue
            ley_estado, _ = _ley_criterios_rotula(
                rotulas_por_id[rid], st["sentido_momento"]
            )
            if abs(st["theta_p_rad"]) > float(ley_estado["theta_p_u_rad"]):
                excedidas.append(rid)
        if excedidas:
            razon_parada = "limite_rotacion: " + ", ".join(excedidas)
            break
        if control_axial["elementos_fuera_biblioteca"] and cfg.detener_fuera_biblioteca_axial:
            razon_parada = "fuera_biblioteca_axial: " + ", ".join(
                str(x) for x in control_axial["elementos_fuera_biblioteca"]
            )
            break
        if pico > 0 and cortante < (1.0 - cfg.perdida_resistencia_fraccion) * pico:
            razon_parada = "perdida_resistencia"
            break
        if objetivo >= desplazamiento_inicial + cfg.desplazamiento_maximo_cm - 1e-12:
            break
    else:
        razon_parada = "max_pasos"

    return {
        "schema_version": "1.0",
        "modelo": contrato["modelo"],
        "metodo": "Newton-Raphson con control directo de desplazamiento",
        "configuracion": asdict(cfg),
        "gravedad_aplicada": bool(
            np.any(np.abs(Pg) > 0) or any(np.any(np.abs(v) > 0) for v in extremo_fijo.values())
        ),
        "caso_gravitacional": datos_caso_gravedad,
        "historia_gravedad": historia_gravedad,
        "fuerzas_elementos_gravedad": fuerzas_gravedad,
        "nodos_control": [int(x) for x in techo],
        "historia": historia,
        "eventos": eventos,
        "estado_final_rotulas": estado,
        "desplazamientos_finales": u.tolist(),
        "convergio": convergio,
        "razon_parada": razon_parada,
        "limitaciones": [
            "análisis de primer orden sin P-Delta",
            "las leyes M-theta(Pg) permanecen fijas durante el pushover de nivel 1",
            "sin comprobación frágil por cortante",
            "leyes y criterios heredados del contrato 3.4 son ilustrativos",
        ],
    }


def guardar_resultados(resultados: Mapping, ruta: str | Path) -> None:
    """Guarda resultados JSON sin valores NumPy no serializables."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = [
    "ConfiguracionPushover", "leer_contrato_sesion05", "crear_patron_lateral",
    "crear_cargas_gravitacionales", "resolver_estado_gravitacional",
    "analizar_pushover", "guardar_resultados",
]
