"""Núcleo reutilizable del análisis momento-curvatura por fibras.

Convención: compresión positiva, fuerza en kgf, longitud en cm, momento en
kgf·cm y curvatura en cm⁻¹. Las propiedades de confinamiento se reciben de
forma explícita para evitar dependencias ocultas de variables del notebook.
"""

from __future__ import annotations

import numpy as np

from sesion_02.mander_concreto import (
    curva_mander_confinado,
    curva_mander_no_confinado,
    modulo_elasticidad,
)
from sesion_02.trilineal_acero import esfuerzo_endurecimiento


def area_segmento_en_franjas(y_centro, radio, bordes):
    """Devuelve el área de un círculo incluida en cada franja horizontal."""
    z = np.clip(np.asarray(bordes, dtype=float) - y_centro, -radio, radio)
    F = z * np.sqrt(np.maximum(radio**2 - z**2, 0.0))
    F += radio**2 * np.arcsin(z / radio)
    return np.diff(F)


def crear_fibras(d):
    """Discretiza concreto confinado/no confinado y barras longitudinales."""
    b, h, n = d["b"], d["h"], int(d["n_fibras_y"])
    if n < 2:
        raise ValueError("n_fibras_y debe ser al menos 2.")
    dy = h / n
    bordes = np.linspace(-h / 2, h / 2, n + 1)
    y = 0.5 * (bordes[:-1] + bordes[1:])

    altura_conf = np.maximum(
        np.minimum(bordes[1:], d["hc"] / 2)
        - np.maximum(bordes[:-1], -d["hc"] / 2),
        0.0,
    )
    A_conf = d["bc"] * altura_conf
    A_no = b * dy - A_conf

    if "barras" in d:
        if not d["barras"]:
            raise ValueError("La lista explícita de barras no puede estar vacía.")
        xs = np.array([float(barra.get("x_cm", 0.0)) for barra in d["barras"]])
        ys = np.array([float(barra["y_cm"]) for barra in d["barras"]])
        diametros_lista, areas_lista = [], []
        for barra in d["barras"]:
            if "diametro_cm" in barra:
                db = float(barra["diametro_cm"])
                area = float(barra.get("area_cm2", np.pi * db**2 / 4))
            elif "area_cm2" in barra:
                area = float(barra["area_cm2"])
                db = float(np.sqrt(4.0 * area / np.pi))
            else:
                raise KeyError("Cada barra requiere diametro_cm o area_cm2.")
            diametros_lista.append(db)
            areas_lista.append(area)
        diametros = np.asarray(diametros_lista)
        As = np.asarray(areas_lista)
        if np.any(diametros <= 0) or np.any(As <= 0):
            raise ValueError("Diámetros y áreas de barras deben ser positivos.")
        radios = diametros / 2
    else:
        y_barra = (
            h / 2
            - d["recubrimiento_libre"]
            - d["db_estribo"]
            - d["db_long"] / 2
        )
        x_extremo = (
            b / 2
            - d["recubrimiento_libre"]
            - d["db_estribo"]
            - d["db_long"] / 2
        )
        xs_capa = np.linspace(-x_extremo, x_extremo, d["n_barras_capa"])
        xs = np.tile(xs_capa, 2)
        ys = np.r_[
            np.full(d["n_barras_capa"], y_barra),
            np.full(d["n_barras_capa"], -y_barra),
        ]
        As = np.full(ys.size, d["As_barra"])
        radios = np.full(ys.size, d["db_long"] / 2)

    dentro_nucleo = (
        (np.abs(xs) + radios <= d["bc"] / 2 + 1e-12)
        & (np.abs(ys) + radios <= d["hc"] / 2 + 1e-12)
    )
    if not np.all(dentro_nucleo):
        raise ValueError("La geometría ubica barras fuera del núcleo confinado.")
    for y_s, radio in zip(ys, radios):
        A_conf -= area_segmento_en_franjas(y_s, radio, bordes)
    if np.any(A_conf < -1e-10) or np.any(A_no < -1e-10):
        raise ValueError("La malla produce áreas negativas de concreto.")
    A_conf = np.maximum(A_conf, 0.0)

    return {
        "y": y,
        "A_conf": A_conf,
        "A_no": A_no,
        "xs": xs,
        "ys": ys,
        "As": As,
        "dy": dy,
        "bordes": bordes,
    }


def sigma_acero(eps, d):
    """Evalúa el acero trilineal con igual ley en tracción y compresión."""
    eps = np.asarray(eps, dtype=float)
    signo = np.sign(eps)
    eps_eval = np.minimum(np.abs(eps), d["eps_su"])
    sig = np.array(
        [
            esfuerzo_endurecimiento(
                float(e), d["fy"], d["Es"], d["Esh"], d["eps_sh"]
            )
            for e in eps_eval
        ]
    )
    return signo * sig


def respuesta_seccion(eps0, phi, fibras, d, fl_ef, rho_s):
    """Integra fuerzas y momento de una sección para eps0 y phi prescritos."""
    eps_c = eps0 + phi * fibras["y"]
    eps_s = eps0 + phi * fibras["ys"]

    sig_no = curva_mander_no_confinado(
        np.maximum(eps_c, 0.0), d["fco"], d["epsco"]
    )
    sig_conf = curva_mander_confinado(
        np.maximum(eps_c, 0.0),
        d["fco"],
        fl_ef,
        d["epsco"],
        rho_s,
        d["fyh"],
        d["eps_su"],
        modulo_elasticidad(d["fco"]),
    )
    sig_s = sigma_acero(eps_s, d)

    fuerzas_c = sig_no * fibras["A_no"] + sig_conf * fibras["A_conf"]
    fuerzas_s = sig_s * fibras["As"]
    N = fuerzas_c.sum() + fuerzas_s.sum()
    M = (fuerzas_c * fibras["y"]).sum() + (fuerzas_s * fibras["ys"]).sum()
    return {
        "eps0": eps0,
        "phi": phi,
        "N": N,
        "M": M,
        "eps_c": eps_c,
        "eps_s": eps_s,
        "eps_c_max": max(eps_c.max(), 0.0),
        "eps_s_abs_max": np.abs(eps_s).max(),
    }


def resolver_eps0(
    phi, P, fibras, d, fl_ef, rho_s, tol_N=1e-5, max_iter=120
):
    """Equilibra N_int=P mediante bisección de la deformación axial eps0."""
    escala = max(abs(P), d["Ag"] * d["fco"], 1.0)

    def residuo(e0):
        return respuesta_seccion(e0, phi, fibras, d, fl_ef, rho_s)["N"] - P

    candidatos = np.linspace(-0.20, 0.20, 161)
    residuos = np.array([residuo(e) for e in candidatos])
    cambios = np.where(residuos[:-1] * residuos[1:] <= 0)[0]
    if cambios.size == 0:
        raise RuntimeError(f"No se pudo equilibrar P={P:.1f} para φ={phi:.3e}.")
    idx = cambios[np.argmin(np.abs(candidatos[cambios]))]
    a, b = candidatos[idx], candidatos[idx + 1]
    fa = residuo(a)
    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc_r = residuo(c)
        if abs(fc_r) <= tol_N * escala or abs(b - a) < 1e-12:
            return respuesta_seccion(c, phi, fibras, d, fl_ef, rho_s)
        if fa * fc_r <= 0:
            b = c
        else:
            a, fa = c, fc_r
    raise RuntimeError("La bisección alcanzó el máximo de iteraciones.")


def utilizacion_limite(estado, d, eps_cu):
    """Devuelve utilización total, del concreto y del acero."""
    u_concreto = estado["eps_c_max"] / eps_cu
    u_acero = estado["eps_s_abs_max"] / d["eps_su"]
    return max(u_concreto, u_acero), u_concreto, u_acero


def curva_momento_curvatura(
    P,
    fibras,
    d,
    fl_ef,
    rho_s,
    eps_cu,
    phi_max=0.0040,
    n_pasos=320,
):
    """Construye la historia equilibrada hasta el primer límite material."""
    registros = []
    eps_y = d["fy"] / d["Es"]
    for phi in np.linspace(0.0, phi_max, n_pasos):
        try:
            estado = resolver_eps0(phi, P, fibras, d, fl_ef, rho_s)
        except RuntimeError:
            break
        u, u_concreto, u_acero = utilizacion_limite(estado, d, eps_cu)
        if u >= 1.0 and registros:
            phi_lo, phi_hi = registros[-1]["phi"], phi
            for _ in range(50):
                phi_mid = 0.5 * (phi_lo + phi_hi)
                estado_mid = resolver_eps0(
                    phi_mid, P, fibras, d, fl_ef, rho_s
                )
                if utilizacion_limite(estado_mid, d, eps_cu)[0] >= 1.0:
                    phi_hi = phi_mid
                else:
                    phi_lo = phi_mid
            estado = resolver_eps0(phi_hi, P, fibras, d, fl_ef, rho_s)
            _, u_concreto, u_acero = utilizacion_limite(estado, d, eps_cu)
            estado["fluencia_acero"] = estado["eps_s_abs_max"] >= eps_y
            estado["limite_concreto"] = u_concreto >= u_acero
            estado["limite_acero"] = u_acero > u_concreto
            registros.append(estado)
            break
        estado["fluencia_acero"] = estado["eps_s_abs_max"] >= eps_y
        estado["limite_concreto"] = u_concreto >= 1.0
        estado["limite_acero"] = u_acero >= 1.0
        registros.append(estado)
        if estado["limite_concreto"] or estado["limite_acero"]:
            break
    if len(registros) < 3:
        raise RuntimeError("La curva no contiene suficientes estados equilibrados.")
    return registros


def resumir_curva(registros, P):
    """Extrae fluencia, máximo, rigidez secante y evento final."""
    phi = np.array([r["phi"] for r in registros])
    M = np.abs([r["M"] for r in registros])
    i_y = next(
        (i for i, registro in enumerate(registros) if registro["fluencia_acero"]),
        None,
    )
    if i_y is None:
        i_y = int(np.argmax(M))
    i_u = int(np.argmax(M))
    phi_y, My = phi[i_y], M[i_y]
    phi_u, Mu = phi[i_u], M[i_u]
    EI_sec = My / phi_y if phi_y > 0 else np.nan
    return {
        "P_kgf": float(P),
        "phi_y_1_cm": float(phi_y),
        "My_kgf_cm": float(My),
        "phi_u_1_cm": float(phi_u),
        "Mu_kgf_cm": float(Mu),
        "EI_sec_kgf_cm2": float(EI_sec),
        "evento_final": (
            "concreto"
            if registros[-1]["limite_concreto"]
            else "acero"
            if registros[-1]["limite_acero"]
            else "fin_phi"
        ),
    }


__all__ = [
    "area_segmento_en_franjas",
    "crear_fibras",
    "sigma_acero",
    "respuesta_seccion",
    "resolver_eps0",
    "utilizacion_limite",
    "curva_momento_curvatura",
    "resumir_curva",
]
