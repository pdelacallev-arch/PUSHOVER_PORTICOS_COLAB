"""Gráficos exclusivamente 2D para los resultados de la sesión 06."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def graficar_curva_capacidad(resultados, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    h = resultados["historia"]
    d = np.array([x["desplazamiento_control_cm"] for x in h])
    v = np.array([x["cortante_basal_kgf"] for x in h]) / 1000.0
    separacion_marcadores = max(1, int(np.ceil(len(d) / 30)))
    ax.plot(
        d, v, "o-", lw=1.8, ms=3, markevery=separacion_marcadores,
        label="Edificio completo",
    )
    formaciones = [
        evento for evento in resultados.get("eventos", [])
        if evento.get("tipo") == "formacion_rotula"
    ]
    if formaciones:
        primera = min(formaciones, key=lambda evento: int(evento["paso"]))
        ax.scatter(
            float(primera["desplazamiento_control_cm"]),
            float(primera["cortante_basal_kgf"]) / 1000.0,
            marker="D", s=38, color="darkorange", zorder=4,
            label="Primera rótula",
        )
    imax = int(np.argmax(v))
    ax.scatter(d[imax], v[imax], color="crimson", zorder=3, label="Resistencia máxima")
    ax.set(xlabel="Desplazamiento promedio de techo (cm)", ylabel="Cortante basal (tf)",
           title="Curva de capacidad del edificio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def graficar_derivas_finales(resultados, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ultimo = resultados["historia"][-1]
    for pid, derivas in ultimo.get("derivas_por_portico", {}).items():
        ax.plot(100 * np.abs(derivas), np.arange(1, len(derivas) + 1), "o-", label=pid)
    ax.set(xlabel="Deriva de entrepiso (%)", ylabel="Nivel",
           title="Derivas por pórtico en el último paso convergido")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def graficar_rotulas_2d(contrato, resultados, ax=None):
    """Dibuja los tres pórticos coplanares y colorea sus rótulas activas en su ubicación física."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 5))
    nodos = {int(k): np.asarray(v, dtype=float) for k, v in contrato["nodos"].items()}
    elementos = {int(e["id"]): e for e in contrato["elementos"]}
    for e in contrato["elementos"]:
        xy = np.vstack([nodos[int(e["i"])], nodos[int(e["j"])]] )
        ax.plot(xy[:, 0], xy[:, 1], color="0.65", lw=1.0)
    for p in contrato.get("puntales", []):
        xy = np.vstack([nodos[int(p["i"])], nodos[int(p["j"])]] )
        ax.plot(xy[:, 0], xy[:, 1], "--", color="steelblue", lw=1.0)

    colores = {"IO": "gold", "LS": "darkorange", "CP": "crimson", "excede_CP": "purple"}
    usados = set()
    rotulas = {str(r["id"]): r for r in contrato["rotulas"]}
    for rid, estado in resultados["estado_final_rotulas"].items():
        if not estado["activa"]:
            continue
        rotula = rotulas[rid]
        elem_id = int(rotula.get("elemento", -1))
        if elem_id in elementos:
            elem = elementos[elem_id]
            xi = nodos[int(elem["i"])]
            xj = nodos[int(elem["j"])]
            vec = xj - xi
            L = np.linalg.norm(vec)
            ubic = rotula.get("ubicacion", {})
            if "x_desde_eje_i_cm" in ubic and L > 1e-9:
                x_local = float(ubic["x_desde_eje_i_cm"])
                xy = xi + (x_local / L) * vec
            elif "distancia_desde_eje_cm" in ubic and L > 1e-9:
                dist = float(ubic["distancia_desde_eje_cm"])
                if ubic.get("extremo", "i") == "i":
                    xy = xi + (dist / L) * vec
                else:
                    xy = xj - (dist / L) * vec
            elif "nodo" in rotula and int(rotula["nodo"]) in nodos:
                xy = nodos[int(rotula["nodo"])]
            else:
                xy = xi
        elif "nodo" in rotula and int(rotula["nodo"]) in nodos:
            xy = nodos[int(rotula["nodo"])]
        else:
            continue

        clase = estado["desempeno"]
        ax.scatter(xy[0], xy[1], s=26, color=colores.get(clase, "gold"), zorder=4,
                   label=clase if clase not in usados else None)
        usados.add(clase)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="X global (cm)", ylabel="Elevación (cm)",
           title="Secuencia coplanar de pórticos y estado final de rótulas")
    ax.grid(True, alpha=0.2)
    if usados:
        ax.legend(title="Desempeño")
    return ax


def guardar_figuras(contrato, resultados, carpeta):
    import matplotlib.pyplot as plt

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre in ("curva_capacidad.png", "derivas_porticos.png", "rotulas_2d.png"):
        fig, ax = plt.subplots(figsize=(12, 5) if nombre == "rotulas_2d.png" else (7, 5))
        if nombre == "rotulas_2d.png":
            graficar_rotulas_2d(contrato, resultados, ax)
        elif nombre == "derivas_porticos.png":
            graficar_derivas_finales(resultados, ax)
        else:
            graficar_curva_capacidad(resultados, ax)
        fig.tight_layout()
        fig.savefig(carpeta / nombre, dpi=180)
        plt.close(fig)


__all__ = [
    "graficar_curva_capacidad", "graficar_derivas_finales",
    "graficar_rotulas_2d", "guardar_figuras",
]
