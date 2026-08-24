"""Generación de familias M-theta para distintas secciones y cargas axiales."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from sesion_03.modelo_rotula import (
    criterios_columna_asce41_17,
    propiedades_rotula,
)
from sesion_03.modelo_seccion import (
    crear_fibras,
    curva_momento_curvatura,
    resumir_curva,
)


def _reflejar_fibras(fibras):
    reflejadas = deepcopy(fibras)
    reflejadas["ys"] = -np.asarray(fibras["ys"], dtype=float)
    reflejadas["xs"] = -np.asarray(fibras["xs"], dtype=float)
    return reflejadas


def _ley_reducida(rotula):
    return {
        "My_kgf_cm": float(rotula["My_kgf_cm"]),
        "Mu_kgf_cm": float(rotula["Mu_kgf_cm"]),
        "theta_p_u_rad": float(rotula["theta_p_u_rad"]),
        "Kp_kgf_cm_rad": float(rotula["Kp_promedio_kgf_cm_rad"]),
        "degradacion_residual": rotula["degradacion_residual"],
    }


def generar_biblioteca_rotulas_axiales(
    configuraciones: Mapping,
    *,
    phi_max_1_cm: float = 0.004,
    n_pasos: int = 320,
):
    """Genera ramas positiva/negativa para cada sección y cada valor de P.

    Cada configuración requiere ``datos_seccion``, ``fl_ef``, ``rho_s``,
    ``eps_cu``, ``Lp_cm`` y ``P_kgf``. Para columnas puede indicar
    ``ratio_cortante`` y ``n_ramas``. Para otros componentes debe proporcionar
    ``criterios_factory(P, datos, sentido)`` o ``criterios_aceptacion``.
    """
    biblioteca = {}
    for sid, configuracion_original in configuraciones.items():
        cfg = dict(configuracion_original)
        requeridos = {"datos_seccion", "fl_ef", "rho_s", "eps_cu", "Lp_cm", "P_kgf"}
        faltantes = requeridos - set(cfg)
        if faltantes:
            raise KeyError(f"Sección {sid}: faltan {sorted(faltantes)}.")
        datos = deepcopy(dict(cfg["datos_seccion"]))
        fibras_base = crear_fibras(datos)
        P_valores = np.asarray(cfg["P_kgf"], dtype=float)
        if P_valores.ndim != 1 or P_valores.size == 0 or np.any(np.diff(P_valores) <= 0):
            raise ValueError(f"Sección {sid}: P_kgf debe ser estrictamente creciente.")
        tipo = str(cfg.get("tipo", "columna"))
        familia = {"positivo": [], "negativo": []}
        for sentido in ("positivo", "negativo"):
            fibras = fibras_base if sentido == "positivo" else _reflejar_fibras(fibras_base)
            for P in P_valores:
                registros = curva_momento_curvatura(
                    float(P), fibras, datos, float(cfg["fl_ef"]),
                    float(cfg["rho_s"]), float(cfg["eps_cu"]),
                    phi_max=float(phi_max_1_cm), n_pasos=int(n_pasos),
                )
                resumen = resumir_curva(registros, float(P))
                fabrica = cfg.get("criterios_factory")
                if fabrica is not None:
                    criterios = fabrica(float(P), datos, sentido)
                elif "criterios_aceptacion" in cfg:
                    criterios = deepcopy(cfg["criterios_aceptacion"])
                elif tipo == "columna" and cfg.get("calcular_criterios", True):
                    criterios = criterios_columna_asce41_17(
                        float(P), datos,
                        ratio_cortante=float(cfg.get("ratio_cortante", 0.20)),
                        n_ramas=int(cfg.get("n_ramas", 2)),
                    )
                else:
                    criterios = None
                rotula = propiedades_rotula(
                    registros, resumen, datos, Lp=float(cfg["Lp_cm"]),
                    criterios=criterios,
                )
                familia[sentido].append({
                    "P_kgf": float(P), "ley": _ley_reducida(rotula),
                    "EI_sec_kgf_cm2": float(rotula["EI_sec_kgf_cm2"]),
                    "criterios_aceptacion": rotula["criterios_aceptacion"],
                    "evento_final": rotula["evento_final"],
                })
        biblioteca[str(sid)] = familia
    return biblioteca


def generar_biblioteca_medrano(
    *,
    phi_max_1_cm: float = 0.004,
    n_pasos: int = 120,
) -> tuple[dict, dict]:
    """Compatibilidad histórica; delega en el generador genérico por entrada."""
    from sesion_03.casos_medrano import RUTA_ENTRADA
    return generar_biblioteca_desde_entrada(
        RUTA_ENTRADA, phi_max_1_cm=phi_max_1_cm, n_pasos=n_pasos
    )


def generar_biblioteca_desde_entrada(
    ruta_entrada: str | Path,
    *,
    phi_max_1_cm: float | None = None,
    n_pasos: int | None = None,
) -> tuple[dict, dict]:
    """Genera M-theta(P) para cualquier catálogo de secciones JSON válido."""
    from sesion_03.casos_medrano import (
        cargar_entrada,
        cargas_axiales_columna,
        criterios_viga_asumidos,
        preparar_seccion,
    )

    ruta = Path(ruta_entrada)
    entrada = cargar_entrada(ruta)
    opciones = entrada.get("biblioteca_rotulas", {})
    phi_max = float(
        opciones.get("phi_max_1_cm", 0.004)
        if phi_max_1_cm is None else phi_max_1_cm
    )
    pasos = int(opciones.get("n_pasos", 120) if n_pasos is None else n_pasos)
    configuraciones, metadata = {}, {}
    for sid in entrada["secciones"]:
        base = preparar_seccion(sid, ruta)
        datos = base["datos_seccion"]
        seccion = entrada["secciones"][sid]
        if base["tipo"] == "columna":
            P = cargas_axiales_columna(sid, ruta=ruta)
        else:
            P = seccion.get("P_kgf") or opciones.get(
                "P_vigas_kgf", [-20_000.0, 0.0, 20_000.0]
            )
        cfg = {
            "tipo": base["tipo"], "datos_seccion": datos,
            "fl_ef": base["fl_ef"], "rho_s": base["rho_s"],
            "eps_cu": base["eps_cu"], "Lp_cm": base["Lp_cm"],
            "P_kgf": P,
            "calcular_criterios": False,
        }
        if base["tipo"] == "columna":
            cfg.update({
                "ratio_cortante": float(seccion.get("ratio_cortante", 0.20)),
                "n_ramas": base["n_ramas"],
            })
        configuraciones[sid] = cfg
        metadata[sid] = {
            "tipo": base["tipo"], "b_cm": datos["b"], "h_cm": datos["h"],
            "fc_kgf_cm2": datos["fco"], "fy_kgf_cm2": datos["fy"],
            "As_total_cm2": datos["As_total"], "barras": datos["barras"],
            "Lp_cm": base["Lp_cm"], "fl_ef_kgf_cm2": base["fl_ef"],
            "rho_s": base["rho_s"], "eps_cu": base["eps_cu"],
            "fuente": base["fuente"], "estado_datos": "DEFINIDO_EN_ENTRADA_MAESTRA",
        }
    biblioteca = generar_biblioteca_rotulas_axiales(
        configuraciones, phi_max_1_cm=phi_max, n_pasos=pasos
    )
    for sid, meta in metadata.items():
        biblioteca[sid]["metadata"] = meta
    return biblioteca, metadata


def guardar_contrato_sesion03(
    biblioteca: Mapping,
    ruta: str | Path,
    *,
    schema_version: str = "3.5",
    metadata_adicional: Mapping | None = None,
) -> Path:
    """Exporta el contrato JSON oficial de la sesión 03 para consumo de sesiones 05 y 06."""
    ruta_p = Path(ruta)
    ruta_p.parent.mkdir(parents=True, exist_ok=True)
    contrato = {
        "schema_version": schema_version,
        "sesion_origen": "sesion_03",
        "descripcion": "Biblioteca de rótulas plásticas M-theta(P) y propiedades no lineales de secciones",
        "unidades": {"fuerza": "kgf", "longitud": "cm", "angulo": "rad", "esfuerzo": "kgf/cm2"},
        "secciones": sorted(list(biblioteca.keys())),
        "biblioteca_rotulas": biblioteca,
        "metadata_adicional": dict(metadata_adicional or {}),
    }
    ruta_p.write_text(json.dumps(contrato, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta_p


def leer_contrato_sesion03(ruta: str | Path) -> dict:
    """Lee y valida el contrato JSON de la sesión 03."""
    ruta_p = Path(ruta)
    if not ruta_p.exists():
        raise FileNotFoundError(f"No se encontró el contrato de sesión 03 en: {ruta_p}")
    datos = json.loads(ruta_p.read_text(encoding="utf-8"))
    if "biblioteca_rotulas" not in datos:
        # Compatibilidad si el JSON contiene directamente la biblioteca
        return {"schema_version": "legacy", "biblioteca_rotulas": datos}
    return datos


__all__ = [
    "generar_biblioteca_rotulas_axiales",
    "generar_biblioteca_medrano",
    "generar_biblioteca_desde_entrada",
    "guardar_contrato_sesion03",
    "leer_contrato_sesion03",
]
