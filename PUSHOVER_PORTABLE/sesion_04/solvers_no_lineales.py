"""Solucionadores pedagógicos reutilizables de la sesión 4.

El arc-length implementado es un benchmark normalizado de un grado de
libertad. No debe confundirse con control por desplazamiento ni con el
solucionador vectorial del pórtico que se desarrollará en sesiones posteriores.
"""

from __future__ import annotations

import numpy as np


def fuerza_normalizada(x):
    return 1.5 * x - 0.5 * x**3


def tangente_normalizada(x):
    return 1.5 - 1.5 * x**2


def biseccion_monotona(funcion, objetivo, a=0.0, b=1.0, tol=1e-13):
    fa, fb = funcion(a) - objetivo, funcion(b) - objetivo
    if fa * fb > 0:
        raise ValueError("El objetivo no está encerrado.")
    for _ in range(100):
        m = 0.5 * (a + b)
        fm = funcion(m) - objetivo
        if abs(fm) < tol:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def newton_control_carga(
    lambdas,
    x0=0.0,
    tol=1e-10,
    max_iter=25,
    fuerza=fuerza_normalizada,
    tangente=tangente_normalizada,
):
    """Resuelve fuerza(x)=lambda con lambda prescrito."""
    x = float(x0)
    historia = []
    for paso, lam in enumerate(lambdas, start=1):
        convergio = False
        residuo = np.nan
        for it in range(1, max_iter + 1):
            residuo = fuerza(x) - lam
            kt = tangente(x)
            if abs(residuo) <= tol:
                convergio = True
                break
            if abs(kt) < 1e-12:
                break
            dx = -residuo / kt
            x += np.clip(dx, -0.35, 0.35)
        historia.append(
            {
                "paso": paso,
                "lambda": float(lam),
                "x": float(x),
                "iteraciones": it,
                "residuo": float(residuo),
                "convergio": convergio,
            }
        )
        if not convergio:
            break
    return historia


def control_desplazamiento(x_objetivos, fuerza=fuerza_normalizada):
    """Evalúa lambda=fuerza(x) para desplazamientos prescritos en 1 GDL."""
    return [
        {
            "paso": i,
            "x": float(x),
            "lambda": float(fuerza(x)),
            "iteraciones": 0,
            "residuo": 0.0,
            "convergio": True,
        }
        for i, x in enumerate(x_objetivos, start=1)
    ]


def longitud_arco_1gdl(
    n_pasos=55,
    ds=0.04,
    alpha=1.0,
    tol=1e-10,
    max_iter=30,
    fuerza=fuerza_normalizada,
    tangente=tangente_normalizada,
):
    """Continuación esférica predictor-corrector para g=fuerza(x)-lambda."""
    x_n, lam_n = 0.0, 0.0
    direccion_previa = np.array([1.0, tangente(0.0)])
    historia = [
        {
            "paso": 0,
            "x": x_n,
            "lambda": lam_n,
            "iteraciones": 0,
            "residuo": 0.0,
            "restriccion": 0.0,
            "convergio": True,
        }
    ]

    for paso in range(1, n_pasos + 1):
        kt = tangente(x_n)
        vector_tangente = np.array([1.0, kt])
        if np.dot(vector_tangente, direccion_previa) < 0:
            vector_tangente *= -1.0
        escala = np.sqrt(
            vector_tangente[0] ** 2 + (alpha * vector_tangente[1]) ** 2
        )
        incremento = ds * vector_tangente / escala
        x, lam = x_n + incremento[0], lam_n + incremento[1]
        convergio = False

        for it in range(1, max_iter + 1):
            dx_total, dl_total = x - x_n, lam - lam_n
            g = fuerza(x) - lam
            c = dx_total**2 + (alpha * dl_total) ** 2 - ds**2
            if max(abs(g), abs(c) / ds) <= tol:
                convergio = True
                break
            J = np.array(
                [
                    [tangente(x), -1.0],
                    [2.0 * dx_total, 2.0 * alpha**2 * dl_total],
                ]
            )
            try:
                correccion = np.linalg.solve(J, -np.array([g, c]))
            except np.linalg.LinAlgError:
                break
            x += correccion[0]
            lam += correccion[1]

        historia.append(
            {
                "paso": paso,
                "x": float(x),
                "lambda": float(lam),
                "iteraciones": it,
                "residuo": float(g),
                "restriccion": float(c),
                "convergio": convergio,
            }
        )
        if not convergio:
            break
        direccion_previa = np.array([x - x_n, lam - lam_n])
        x_n, lam_n = x, lam
    return historia


def extraer_historia(historia):
    """Devuelve vectores x y lambda de una historia de continuación."""
    return (
        np.array([registro["x"] for registro in historia]),
        np.array([registro["lambda"] for registro in historia]),
    )


__all__ = [
    "fuerza_normalizada",
    "tangente_normalizada",
    "biseccion_monotona",
    "newton_control_carga",
    "control_desplazamiento",
    "longitud_arco_1gdl",
    "extraer_historia",
]
