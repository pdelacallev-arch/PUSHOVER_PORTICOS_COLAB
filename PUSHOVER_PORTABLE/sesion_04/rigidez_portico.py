"""Matrices reutilizables de un elemento de pórtico plano Euler-Bernoulli."""

from __future__ import annotations

import numpy as np


def matriz_rigidez_local_portico2d(EA, EI, L):
    """Matriz local 6x6 con orden [ui, vi, θi, uj, vj, θj]."""
    if EA <= 0 or EI <= 0 or L <= 0:
        raise ValueError("EA, EI y L deben ser positivos.")
    a = EA / L
    b = 12.0 * EI / L**3
    c = 6.0 * EI / L**2
    d = 4.0 * EI / L
    e = 2.0 * EI / L
    return np.array(
        [
            [a, 0, 0, -a, 0, 0],
            [0, b, c, 0, -b, c],
            [0, c, d, 0, -c, e],
            [-a, 0, 0, a, 0, 0],
            [0, -b, -c, 0, b, -c],
            [0, c, e, 0, -c, d],
        ],
        dtype=float,
    )


def matriz_transformacion_portico2d(angulo_rad):
    """Transformación de desplazamientos globales a locales."""
    c, s = np.cos(angulo_rad), np.sin(angulo_rad)
    return np.array(
        [
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=float,
    )


def rigidez_elemento_portico2d(EA, EI_sec, L, angulo_rad=0.0, eta=1.0):
    """Devuelve matrices local/global con EI_t=eta*EI_sec."""
    if not (0 < eta <= 1.0):
        raise ValueError("eta debe satisfacer 0 < eta <= 1.")
    kl = matriz_rigidez_local_portico2d(EA, eta * EI_sec, L)
    T = matriz_transformacion_portico2d(angulo_rad)
    return {"local": kl, "global": T.T @ kl @ T, "T": T, "EI_t": eta * EI_sec}


__all__ = [
    "matriz_rigidez_local_portico2d",
    "matriz_transformacion_portico2d",
    "rigidez_elemento_portico2d",
]

