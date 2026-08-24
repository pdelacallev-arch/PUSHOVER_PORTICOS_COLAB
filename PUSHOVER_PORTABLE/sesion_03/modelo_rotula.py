"""Construcción, validación y transferencia de rótulas concentradas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CAMPOS_MINIMOS_ROTULA = (
    "EI_sec_kgf_cm2",
    "My_kgf_cm",
    "Mu_kgf_cm",
    "theta_local_y_rad",
)


def criterios_columna_asce41_17(P, d, ratio_cortante=0.20, n_ramas=2):
    """Criterios académicos provisionales de ASCE 41-17, Tabla 10-8."""
    if ratio_cortante < 0.20:
        raise ValueError("VyE/VColOE no debe tomarse menor que 0.20.")
    if n_ramas < 1:
        raise ValueError("n_ramas debe ser positivo.")
    Av = n_ramas * np.pi * d["db_estribo"] ** 2 / 4
    rho_t = Av / (d["b"] * d["s_estribo"])
    if rho_t < 0.0005:
        raise ValueError("La ecuación de ASCE 41-17 no aplica para rho_t < 0.0005.")
    rho_t_calculo = min(rho_t, 0.0175)
    nu = P / (d["Ag"] * d["fco"])
    nu_calculo = max(nu, 0.10)
    if nu_calculo > 0.50:
        raise ValueError("Este ejemplo implementa la expresión de b solo para nu <= 0.50.")
    a = max(
        0.042
        - 0.043 * nu_calculo
        + 0.63 * rho_t_calculo
        - 0.023 * ratio_cortante,
        0.0,
    )
    b = (
        0.5
        / (
            5.0
            + (nu_calculo / 0.8)
            * (d["fco"] / (rho_t_calculo * d["fyh"]))
        )
        - 0.01
    )
    b = max(b, a)
    limites = {"IO": min(0.15 * a, 0.005), "LS": 0.50 * b, "CP": 0.70 * b}
    return {
        "referencia": "ASCE 41-17, Tabla 10-8",
        "componente": "columna rectangular de concreto armado con estribos",
        "theta_p_rad": limites,
        "parametros": {
            "nu_datos": float(nu),
            "nu_calculo": float(nu_calculo),
            "Av_cm2": float(Av),
            "rho_t": float(rho_t),
            "VyE_VColOE": float(ratio_cortante),
            "a_rad": float(a),
            "b_rad": float(b),
        },
        "supuestos_provisionales": [
            "f'cE = f'c y fytE = fyh del ejemplo",
            "dos ramas efectivas de estribo en la dirección analizada",
            "desarrollo y empalmes adecuados",
            "estribos adecuadamente anclados en el núcleo",
            "VyE/VColOE = 0.20 provisional; sustituir con el valor del pórtico",
        ],
    }


def validar_criterios_aceptacion(configuracion, theta_p_limite):
    configuracion = {} if configuracion is None else dict(configuracion)
    valores = dict(configuracion.get("theta_p_rad", {}))
    limites = {}
    for nivel in ("IO", "LS", "CP"):
        valor = valores.get(nivel)
        if valor is None:
            continue
        valor = float(valor)
        if not np.isfinite(valor) or valor < 0:
            raise ValueError(f"El límite {nivel} debe ser una rotación no negativa.")
        limites[nivel] = valor
    secuencia = [limites[n] for n in ("IO", "LS", "CP") if n in limites]
    if any(b <= a for a, b in zip(secuencia, secuencia[1:])):
        raise ValueError("Los límites deben satisfacer IO < LS < CP.")
    excedidos = [n for n, valor in limites.items() if valor > theta_p_limite]
    return {
        "referencia": configuracion.get("referencia", "por definir"),
        "componente": configuracion.get("componente", "por definir"),
        "theta_p_rad": limites,
        "parametros": configuracion.get("parametros", {}),
        "supuestos_provisionales": configuracion.get("supuestos_provisionales", []),
        "estado": "configurados" if len(limites) == 3 else "pendientes",
        "fuera_de_capacidad_calculada": excedidos,
    }


def propiedades_rotula(registros, resumen, d, Lp=None, criterios=None):
    """Convierte una historia M-phi en respuesta local y backbone plástico."""
    Lp = 0.5 * d["h"] if Lp is None else float(Lp)
    if Lp <= 0:
        raise ValueError("Lp debe ser positiva.")
    phi_hist = np.array([r["phi"] for r in registros], dtype=float)
    M_hist = np.abs(np.array([r["M"] for r in registros], dtype=float))
    phi_y, My = resumen["phi_y_1_cm"], resumen["My_kgf_cm"]
    i_y = int(np.searchsorted(phi_hist, phi_y, side="left"))
    i_max, i_lim = int(np.argmax(M_hist)), len(registros) - 1
    theta_local = phi_hist * Lp
    theta_p = np.maximum((phi_hist - phi_y) * Lp, 0.0)
    theta_p_max = theta_p[i_max]
    Kp_prom = (M_hist[i_max] - My) / theta_p_max if theta_p_max > 0 else 0.0
    phi_bil = np.array([0.0, phi_y, phi_hist[i_max]])
    M_bil = np.array([0.0, My, M_hist[i_max]])
    criterios_validados = validar_criterios_aceptacion(criterios, theta_p[i_lim])
    return {
        "modelo": "backbone_monotonico_M_theta_p_derivado_de_fibras",
        "unidades": {
            "M": "kgf_cm",
            "theta": "rad",
            "EI_sec": "kgf_cm2",
            "Kp_promedio": "kgf_cm/rad",
        },
        "P_kgf": float(resumen["P_kgf"]),
        "Lp_cm": Lp,
        "My_kgf_cm": float(My),
        "Mu_kgf_cm": float(M_hist[i_max]),
        "M_limite_kgf_cm": float(M_hist[i_lim]),
        "theta_local_y_rad": float(theta_local[i_y]),
        "theta_local_u_rad": float(theta_local[i_lim]),
        "theta_p_y_rad": 0.0,
        "theta_p_max_rad": float(theta_p[i_max]),
        "theta_p_u_rad": float(theta_p[i_lim]),
        "EI_sec_kgf_cm2": float(resumen["EI_sec_kgf_cm2"]),
        "Kp_promedio_kgf_cm_rad": float(Kp_prom),
        "evento_final": resumen["evento_final"],
        "degradacion_residual": None,
        "bilinealizacion": {
            "criterio": "puntos_caracteristicos_primera_fluencia_y_maximo",
            "M_phi": {"phi_1_cm": phi_bil.tolist(), "M_kgf_cm": M_bil.tolist()},
            "M_theta_local": {
                "theta_rad": (phi_bil * Lp).tolist(),
                "M_kgf_cm": M_bil.tolist(),
            },
            "rama_plastica": {
                "theta_p_rad": [0.0, float(theta_p_max)],
                "M_kgf_cm": [float(My), float(M_hist[i_max])],
            },
        },
        "criterios_aceptacion": criterios_validados,
        "respuesta_local": {
            "theta_rad": theta_local.tolist(),
            "M_kgf_cm": M_hist.tolist(),
        },
        "backbone_plastico": {
            "theta_p_rad": theta_p[i_y:].tolist(),
            "M_kgf_cm": M_hist[i_y:].tolist(),
        },
        "indices": {"fluencia": i_y, "maximo": i_max, "limite": i_lim},
    }


def cargar_contrato_rotula(ruta):
    """Lee un contrato JSON de rótula sin modificarlo."""
    ruta = Path(ruta)
    return json.loads(ruta.read_text(encoding="utf-8"))


def validar_contrato_rotula(contrato, campos=CAMPOS_MINIMOS_ROTULA):
    """Valida el contrato y devuelve el subdiccionario ``rotula``."""
    if "rotula" not in contrato:
        raise KeyError("El contrato no contiene el bloque 'rotula'.")
    rotula = contrato["rotula"]
    faltantes = [campo for campo in campos if campo not in rotula]
    if faltantes:
        raise KeyError(f"Contrato de sesión 3 incompleto: {faltantes}")
    if not (
        rotula["EI_sec_kgf_cm2"] > 0
        and 0 < rotula["My_kgf_cm"] <= rotula["Mu_kgf_cm"]
        and rotula["theta_local_y_rad"] > 0
    ):
        raise ValueError("El contrato de sesión 3 contiene magnitudes no físicas.")
    return rotula


__all__ = [
    "CAMPOS_MINIMOS_ROTULA",
    "criterios_columna_asce41_17",
    "validar_criterios_aceptacion",
    "propiedades_rotula",
    "cargar_contrato_rotula",
    "validar_contrato_rotula",
]

