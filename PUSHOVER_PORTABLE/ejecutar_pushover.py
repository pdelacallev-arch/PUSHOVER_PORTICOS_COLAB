"""Ejecutor único y portable de las sesiones 03 a 06.

Uso:
    python ejecutar_pushover.py --entrada casos/edificio_2pisos.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

from sesion_03.biblioteca_rotulas import (
    generar_biblioteca_desde_entrada,
    guardar_contrato_sesion03,
)
from sesion_04.rigidez_portico import matriz_rigidez_local_portico2d
from sesion_05.contrato_portable import (
    construir_contrato_sesion05,
    guardar_contrato_sesion05,
)
from sesion_06.caso_medrano import (
    crear_caso_gravitacional_desde_datos,
    resumen_cargas,
)
from sesion_06.pushover import (
    ConfiguracionPushover,
    analizar_pushover,
    crear_patron_lateral,
    guardar_resultados,
)
from sesion_06.rotulas_axiales import calibrar_contrato_nivel1
from sesion_06.criterios_externos import aplicar_criterios_externos


def _leer_json(ruta: Path) -> dict:
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
    return json.loads(ruta.read_text(encoding="utf-8"))


def cargar_entrada_maestra(ruta: str | Path) -> tuple[dict, Path, dict]:
    """Resuelve únicamente referencias relativas al manifiesto maestro."""
    maestra = Path(ruta).resolve()
    manifiesto = _leer_json(maestra)
    if str(manifiesto.get("schema_version")) != "1.0":
        raise ValueError("La entrada maestra requiere schema_version 1.0.")
    archivos = manifiesto.get("archivos", {})
    requeridos = {"geometria_y_secciones", "cargas", "criterios_aceptacion"}
    if not requeridos <= set(archivos):
        raise ValueError(f"Faltan archivos en la entrada maestra: {sorted(requeridos-set(archivos))}")
    referencias = {}
    for nombre in requeridos:
        referencia = Path(archivos[nombre])
        if referencia.is_absolute():
            raise ValueError("Las referencias de la entrada maestra deben ser relativas.")
        referencias[nombre] = (maestra.parent / referencia).resolve()
    entrada = _leer_json(referencias["geometria_y_secciones"])
    cargas = _leer_json(referencias["cargas"])
    criterios = _leer_json(referencias["criterios_aceptacion"])
    entrada.update({
        "caso_id": manifiesto.get("caso_id"),
        "modelo": manifiesto.get("modelo", {}),
        "calibracion": manifiesto.get("calibracion", {}),
        "pushover": manifiesto.get("pushover", {}),
        "configuracion_criterios": manifiesto.get("criterios_aceptacion", {}),
        "_catalogo_criterios": criterios,
    })
    if not entrada.get("caso_id"):
        raise ValueError("La entrada maestra debe definir caso_id.")
    return entrada, referencias["geometria_y_secciones"], cargas


def _validar_sesion04(biblioteca, entrada):
    """Comprueba matrices de cada sección sin elegir C1 ni fijar una Pg."""
    P0 = float(entrada["modelo"]["P_referencia_inicial_kgf"])
    L = float(entrada["edificio"].get(
        "altura_entrepiso_cm",
        entrada["edificio"].get("alturas_cm", [300.0])[0],
    ))
    verificaciones = {}
    from sesion_06.rotulas_axiales import interpolar_ley_axial
    for sid, seccion in entrada["secciones"].items():
        seleccion = interpolar_ley_axial(
            biblioteca[sid]["positivo"], P0, fuera_de_rango="limitar"
        )
        EI = float(seleccion["EI_sec_kgf_cm2"])
        Ec = float(seccion.get(
            "Ec_kgf_cm2",
            float(entrada["modelo"]["Ec_coef_sqrt_fc"])
            * np.sqrt(float(seccion["fc_kgf_cm2"])),
        ))
        EA = Ec * float(seccion["b_cm"]) * float(seccion["h_cm"])
        k = matriz_rigidez_local_portico2d(EA, EI, L)
        verificaciones[sid] = {
            "P_referencia_kgf": P0, "EA_kgf": EA, "EI_kgf_cm2": EI,
            "L_referencia_cm": L, "matriz_simetrica": bool(np.allclose(k, k.T)),
            "autovalor_minimo_no_nulo": float(
                min(x for x in np.linalg.eigvalsh(k) if x > 1e-6)
            ),
        }
    return {
        "schema_version": "2.3", "sesion_origen": "sesion_04",
        "caso_id": entrada["caso_id"], "verificaciones": verificaciones,
    }


def _coeficientes_patron(definicion, n_pisos):
    if isinstance(definicion, list):
        return [float(x) for x in definicion]
    if definicion == "triangular":
        return list(np.arange(1, n_pisos + 1, dtype=float))
    if definicion == "uniforme":
        return [1.0] * n_pisos
    raise ValueError("patron_lateral debe ser 'triangular', 'uniforme' o una lista.")


def evaluar_calidad_curva(resultados: dict, configuracion: dict | None = None) -> dict:
    """Verifica que la corrida haya recorrido una zona no lineal con resolución útil."""
    cfg = configuracion or {}
    min_puntos = int(cfg.get("min_puntos", 2))
    requerir_eventos = bool(cfg.get("requerir_eventos_no_lineales", False))
    historia = resultados.get("historia", [])
    formaciones = [
        evento for evento in resultados.get("eventos", [])
        if evento.get("tipo") == "formacion_rotula"
    ]
    desplazamientos = [
        float(paso["desplazamiento_control_cm"]) for paso in historia
    ]
    razones = []
    if len(historia) < min_puntos:
        razones.append(
            f"curva insuficientemente discretizada: {len(historia)} < {min_puntos} puntos"
        )
    if requerir_eventos and not formaciones:
        razones.append("no se registró formación de rótulas durante el pushover")
    return {
        "aprobada": not razones,
        "puntos": len(historia),
        "min_puntos_requeridos": min_puntos,
        "eventos_formacion_rotula": len(formaciones),
        "eventos_no_lineales_requeridos": requerir_eventos,
        "desplazamiento_inicial_cm": desplazamientos[0] if desplazamientos else None,
        "desplazamiento_final_cm": desplazamientos[-1] if desplazamientos else None,
        "razones_rechazo": razones,
    }


def ejecutar_cadena(ruta_entrada, ruta_salida=None):
    entrada, ruta_secciones, cargas = cargar_entrada_maestra(ruta_entrada)
    salida_declarada = Path(
        ruta_salida or (Path("resultados_portables") / entrada["caso_id"])
    )
    salida = salida_declarada.resolve()
    salida.mkdir(parents=True, exist_ok=True)

    biblioteca, metadata = generar_biblioteca_desde_entrada(ruta_secciones)
    guardar_contrato_sesion03(
        biblioteca, salida / "sesion_03" / "contrato_sesion03.json",
        metadata_adicional={"caso_id": entrada["caso_id"], "secciones": metadata},
    )

    contrato_s4 = _validar_sesion04(biblioteca, entrada)
    ruta_s4 = salida / "sesion_04" / "contrato_sesion04.json"
    ruta_s4.parent.mkdir(parents=True, exist_ok=True)
    ruta_s4.write_text(json.dumps(contrato_s4, ensure_ascii=False, indent=2), encoding="utf-8")

    contrato_s5 = construir_contrato_sesion05(entrada, biblioteca)
    guardar_contrato_sesion05(
        contrato_s5, salida / "sesion_05" / "contrato_sesion05.json"
    )

    caso_gravedad = crear_caso_gravitacional_desde_datos(
        contrato_s5, cargas, entrada
    )
    cfg = ConfiguracionPushover(**entrada["pushover"].get("configuracion", {}))
    asignacion = {int(k): str(v) for k, v in contrato_s5["asignacion_secciones"].items()}
    calibracion_cfg = entrada.get("calibracion", {})
    calibracion = calibrar_contrato_nivel1(
        contrato_s5, caso_gravedad, asignacion, biblioteca,
        configuracion=cfg,
        fuera_de_rango=str(calibracion_cfg.get("fuera_de_rango", "error")),
        actualizar_EI=bool(calibracion_cfg.get("actualizar_EI", True)),
        tolerancia_axial_fraccion=float(
            calibracion_cfg.get("tolerancia_axial_fraccion", 0.01)
        ),
        max_iteraciones=int(calibracion_cfg.get("max_iteraciones", 6)),
        catalogo_secciones=entrada["secciones"],
    )
    if not calibracion["convergio"]:
        raise RuntimeError("La calibración gravedad -> Pg -> M-theta(Pg) no convergió.")
    contrato_calibrado = calibracion["contrato"]
    cfg_criterios = entrada.get("configuracion_criterios", {})
    contrato_calibrado = aplicar_criterios_externos(
        contrato_calibrado, entrada["_catalogo_criterios"],
        permitir_provisionales=bool(cfg_criterios.get("permitir_provisionales", False)),
        politica_exceso_capacidad=str(
            cfg_criterios.get("politica_exceso_capacidad", "error")
        ),
    )
    guardar_contrato_sesion05(
        contrato_calibrado,
        salida / "sesion_05" / "contrato_sesion05_calibrado.json",
    )

    n_pisos = int(contrato_calibrado["geometria"]["n_pisos"])
    coef = _coeficientes_patron(
        entrada["pushover"].get("patron_lateral", "triangular"), n_pisos
    )
    patron = crear_patron_lateral(contrato_calibrado, coef)
    resultados = analizar_pushover(
        contrato_calibrado, patron,
        cargas_gravitacionales=caso_gravedad,
        configuracion=cfg,
    )
    calidad_cfg = entrada["pushover"].get("calidad_curva", {})
    calidad = evaluar_calidad_curva(resultados, calidad_cfg)
    calidad["exigir"] = bool(calidad_cfg.get("exigir", False))
    resultados["calidad_curva"] = calidad
    carpeta_s6 = salida / "sesion_06"
    guardar_resultados(resultados, carpeta_s6 / "resultados_pushover.json")
    if entrada["pushover"].get("generar_figuras", True):
        import matplotlib
        matplotlib.use("Agg")
        from sesion_06.visualizacion import guardar_figuras
        guardar_figuras(contrato_calibrado, resultados, carpeta_s6)

    resumen = resumen_cargas(contrato_calibrado, caso_gravedad)
    ultimo = resultados["historia"][-1]
    informe = {
        "schema_version": "1.0", "caso_id": entrada["caso_id"],
        "entrada_maestra": str(Path(ruta_entrada)),
        "salida": str(salida_declarada),
        "calibracion_convergio": calibracion["convergio"],
        "iteraciones_calibracion": len(calibracion["historia_calibracion"]),
        "pushover_convergio": bool(resultados["convergio"]),
        "razon_parada": resultados["razon_parada"],
        "pasos_convergidos": len(resultados["historia"]) - 1,
        "desplazamiento_final_cm": ultimo["desplazamiento_control_cm"],
        "cortante_basal_final_kgf": ultimo["cortante_basal_kgf"],
        "calidad_curva": calidad,
        "carga_gravitacional_total_kgf": resumen["carga_gravitacional_total_kgf"],
        "modelo": {
            "pisos": contrato_calibrado["geometria"]["n_pisos"],
            "porticos": contrato_calibrado["geometria"]["ids_porticos"],
            "nodos": len(contrato_calibrado["nodos"]),
            "elementos": len(contrato_calibrado["elementos"]),
            "rotulas": len(contrato_calibrado["rotulas"]),
        },
        "configuracion_pushover": asdict(cfg),
        "criterios_aceptacion": {
            "origen": "catalogo_externo",
            "norma": contrato_calibrado["criterios_aceptacion_externos"]["norma"],
            "edicion": contrato_calibrado["criterios_aceptacion_externos"]["edicion"],
            "advertencias": len(
                contrato_calibrado["criterios_aceptacion_externos"]["advertencias"]
            ),
        },
    }
    (salida / "resumen_ejecucion.json").write_text(
        json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return informe


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ejecuta automáticamente las sesiones 03–06 de un edificio."
    )
    parser.add_argument("--entrada", required=True, help="JSON maestro del caso")
    parser.add_argument(
        "--salida",
        help="Directorio de salida; por defecto resultados_portables/<caso_id>",
    )
    args = parser.parse_args(argv)
    try:
        informe = ejecutar_cadena(args.entrada, args.salida)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(informe, ensure_ascii=False, indent=2))
    if not informe["pushover_convergio"]:
        return 2
    if informe["calidad_curva"]["exigir"] and not informe["calidad_curva"]["aprobada"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
