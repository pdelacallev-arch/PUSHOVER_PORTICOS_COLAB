"""Geometría, brazos rígidos, rótulas concentradas y ensamblaje 2D.

Los nodos físicos se ubican en los ejes de los apoyos. Los brazos rígidos
conectan esos ejes con las caras del nudo. Una rótula puede definirse desde la
cara o desde el eje; internamente siempre se almacena su coordenada local
medida desde el eje i.

La rigidez con rótulas usa resortes rotacionales tangentes y condensación
estática. El término de resistencia acumulada de la rótula pertenece al
algoritmo incremental de la sesión 6; esta sesión construye su cinemática y su
matriz tangente sin duplicar la flexibilidad elástica del elemento.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from sesion_04.rigidez_portico import (
    matriz_rigidez_local_portico2d,
    matriz_transformacion_portico2d,
)


@dataclass(frozen=True)
class UbicacionRotula:
    """Ubicación normalizada de una rótula a lo largo del eje local."""

    extremo: str
    referencia_entrada: str
    distancia_entrada_cm: float
    x_desde_eje_i_cm: float
    distancia_desde_eje_cm: float
    distancia_desde_cara_cm: float


def normalizar_geometria_edificio(datos: Mapping):
    """Normaliza la geometría del JSON de entrada sin fijar pisos ni vanos.

    Se admiten dos formatos dentro del bloque ``edificio``:

    1. Explícito: ``porticos=[{id, luces_cm, alturas_cm}, ...]``.
    2. Compacto: ``n_pisos``, ``n_vanos_por_portico``, ``luz_cm``,
       ``altura_entrepiso_cm`` y ``porticos_direccion_analisis``.

    El resultado alimenta directamente ``crear_modelo_edificio_porticos``.
    """
    if not isinstance(datos, Mapping):
        raise TypeError("La entrada del edificio debe ser un diccionario.")
    edificio = datos.get("edificio", datos)
    if not isinstance(edificio, Mapping):
        raise ValueError("El bloque 'edificio' debe ser un diccionario.")

    porticos_explicitos = edificio.get("porticos")
    if porticos_explicitos is not None:
        if (not isinstance(porticos_explicitos, Sequence)
                or isinstance(porticos_explicitos, (str, bytes))
                or len(porticos_explicitos) < 2):
            raise ValueError("'edificio.porticos' debe contener al menos dos pórticos.")
        geometrias = []
        for indice, portico in enumerate(porticos_explicitos, start=1):
            if not isinstance(portico, Mapping):
                raise TypeError(f"Pórtico {indice}: la definición debe ser un diccionario.")
            if "luces_cm" not in portico or "alturas_cm" not in portico:
                raise ValueError(
                    f"Pórtico {indice}: faltan 'luces_cm' o 'alturas_cm'."
                )
            geometrias.append({
                "id": str(portico.get("id", f"P{indice}")),
                "luces_cm": [float(x) for x in portico["luces_cm"]],
                "alturas_cm": [float(x) for x in portico["alturas_cm"]],
            })
    else:
        ids = edificio.get("porticos_direccion_analisis")
        if ids is None:
            n_porticos = int(edificio.get("porticos_resistentes_por_direccion", 0))
            ids = [f"P{i}" for i in range(1, n_porticos + 1)]
        if (not isinstance(ids, Sequence) or isinstance(ids, (str, bytes))
                or len(ids) < 2):
            raise ValueError(
                "Defina al menos dos ids en 'porticos_direccion_analisis'."
            )
        ids = [str(pid) for pid in ids]
        n_porticos = len(ids)

        luces_por_portico = edificio.get("luces_cm_por_portico")
        if luces_por_portico is None:
            luces_compartidas = edificio.get("luces_cm")
            if luces_compartidas is not None:
                luces_por_portico = [list(luces_compartidas) for _ in ids]
            else:
                if "luz_cm" not in edificio or "n_vanos_por_portico" not in edificio:
                    raise ValueError(
                        "Defina 'luces_cm_por_portico', 'luces_cm' o el par "
                        "'n_vanos_por_portico' + 'luz_cm'."
                    )
                n_vanos = edificio["n_vanos_por_portico"]
                if isinstance(n_vanos, Sequence) and not isinstance(n_vanos, (str, bytes)):
                    conteos = [int(x) for x in n_vanos]
                    if len(conteos) != n_porticos:
                        raise ValueError(
                            "'n_vanos_por_portico' debe tener un valor por pórtico."
                        )
                else:
                    conteos = [int(n_vanos)] * n_porticos
                luz = float(edificio["luz_cm"])
                luces_por_portico = [[luz] * n for n in conteos]

        alturas_por_portico = edificio.get("alturas_cm_por_portico")
        if alturas_por_portico is None:
            alturas_compartidas = edificio.get("alturas_cm")
            if alturas_compartidas is not None:
                alturas_por_portico = [list(alturas_compartidas) for _ in ids]
            else:
                if "n_pisos" not in edificio or "altura_entrepiso_cm" not in edificio:
                    raise ValueError(
                        "Defina 'alturas_cm_por_portico', 'alturas_cm' o el par "
                        "'n_pisos' + 'altura_entrepiso_cm'."
                    )
                alturas = [float(edificio["altura_entrepiso_cm"])] * int(edificio["n_pisos"])
                alturas_por_portico = [alturas.copy() for _ in ids]

        if len(luces_por_portico) != n_porticos or len(alturas_por_portico) != n_porticos:
            raise ValueError("La geometría debe proporcionar luces y alturas para cada pórtico.")
        geometrias = [
            {
                "id": pid,
                "luces_cm": [float(x) for x in luces],
                "alturas_cm": [float(x) for x in alturas],
            }
            for pid, luces, alturas in zip(ids, luces_por_portico, alturas_por_portico)
        ]

    if len({p["id"] for p in geometrias}) != len(geometrias):
        raise ValueError("Los identificadores de pórtico deben ser únicos.")
    for portico in geometrias:
        if not portico["luces_cm"] or not portico["alturas_cm"]:
            raise ValueError(f"Pórtico {portico['id']}: luces y alturas no pueden estar vacías.")
        if any(x <= 0 for x in portico["luces_cm"] + portico["alturas_cm"]):
            raise ValueError(f"Pórtico {portico['id']}: luces y alturas deben ser positivas.")

    elevaciones = [np.r_[0.0, np.cumsum(p["alturas_cm"])] for p in geometrias]
    if any(e.shape != elevaciones[0].shape or not np.allclose(e, elevaciones[0])
           for e in elevaciones[1:]):
        raise ValueError(
            "Todos los pórticos deben compartir las mismas elevaciones de piso."
        )

    separaciones = edificio.get(
        "separaciones_porticos_cm", edificio.get("separacion_porticos_cm")
    )
    if separaciones is None:
        raise ValueError(
            "Defina 'separaciones_porticos_cm' o 'separacion_porticos_cm'."
        )
    separaciones = _normalizar_separaciones_porticos(len(geometrias), separaciones)
    return {
        "geometrias_porticos": geometrias,
        "separaciones_porticos_cm": separaciones.tolist(),
        "n_porticos": len(geometrias),
        "n_pisos": len(geometrias[0]["alturas_cm"]),
    }


def cargar_geometria_edificio(ruta):
    """Lee y valida la geometría desde un archivo JSON de entrada."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encuentra el archivo de entrada: {ruta}")
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    geometria = normalizar_geometria_edificio(datos)
    geometria["ruta_entrada"] = str(ruta)
    return geometria


def generar_nodos_portico(luces_cm: Sequence[float], alturas_cm: Sequence[float]):
    """Genera nodos por niveles, de izquierda a derecha, con índices desde 1."""
    luces = np.asarray(luces_cm, dtype=float)
    alturas = np.asarray(alturas_cm, dtype=float)
    if luces.ndim != 1 or alturas.ndim != 1 or luces.size == 0 or alturas.size == 0:
        raise ValueError("luces_cm y alturas_cm deben ser vectores no vacíos.")
    if np.any(luces <= 0) or np.any(alturas <= 0):
        raise ValueError("Todas las luces y alturas deben ser positivas.")

    xs = np.r_[0.0, np.cumsum(luces)]
    ys = np.r_[0.0, np.cumsum(alturas)]
    return {
        nivel * len(xs) + vano + 1: np.array([x, y], dtype=float)
        for nivel, y in enumerate(ys)
        for vano, x in enumerate(xs)
    }


def generar_conectividad_portico(n_vanos: int, n_pisos: int):
    """Conectividad de columnas y vigas para la numeración de nodos del curso."""
    if n_vanos < 1 or n_pisos < 1:
        raise ValueError("n_vanos y n_pisos deben ser enteros positivos.")
    n_por_nivel = n_vanos + 1
    elementos = []
    eid = 1
    for piso in range(n_pisos):
        base = piso * n_por_nivel + 1
        techo = (piso + 1) * n_por_nivel + 1
        for eje in range(n_por_nivel):
            elementos.append({"id": eid, "tipo": "columna", "i": base + eje, "j": techo + eje})
            eid += 1
        for vano in range(n_vanos):
            elementos.append({"id": eid, "tipo": "viga", "i": techo + vano, "j": techo + vano + 1})
            eid += 1
    return elementos


def nodos_de_nivel(n_vanos: int, nivel: int):
    """Devuelve los identificadores de los nodos de un nivel, desde la base 0."""
    if not isinstance(n_vanos, (int, np.integer)) or n_vanos < 1:
        raise ValueError("n_vanos debe ser un entero positivo.")
    if not isinstance(nivel, (int, np.integer)) or nivel < 0:
        raise ValueError("nivel debe ser un entero no negativo.")
    primero = nivel * (n_vanos + 1) + 1
    return list(range(primero, primero + n_vanos + 1))


def construir_elementos_portico(
    nodos,
    conectividad,
    EA_columna,
    EI_columna,
    EA_viga=None,
    EI_viga=None,
    brazo_columna_cm=0.0,
    brazo_viga_cm=0.0,
    eta_columna=1.0,
    eta_viga=1.0,
    con_brazos=True,
):
    """Agrega propiedades y brazos rígidos a la conectividad de un pórtico.

    Las columnas del primer piso no tienen brazo inferior en la cimentación.
    En pisos superiores se consideran brazos de columna en ambos extremos.
    Los valores de vigas se igualan a los de columnas cuando se omiten.
    """
    EA_viga = EA_columna if EA_viga is None else EA_viga
    EI_viga = EI_columna if EI_viga is None else EI_viga
    propiedades = {
        "columna": (float(EA_columna), float(EI_columna), float(eta_columna)),
        "viga": (float(EA_viga), float(EI_viga), float(eta_viga)),
    }
    if any(EA <= 0 or EI <= 0 or not (0 < eta <= 1)
           for EA, EI, eta in propiedades.values()):
        raise ValueError("Se requieren EA > 0, EI > 0 y 0 < eta <= 1.")

    bc, bv = float(brazo_columna_cm), float(brazo_viga_cm)
    if bc < 0 or bv < 0:
        raise ValueError("Los brazos rígidos no pueden ser negativos.")

    elementos = []
    for e in conectividad:
        tipo = str(e["tipo"])
        if tipo not in propiedades:
            raise ValueError(f"Tipo de elemento no soportado: {tipo!r}.")
        i, j = int(e["i"]), int(e["j"])
        if i not in nodos or j not in nodos:
            raise KeyError("La conectividad referencia un nodo inexistente.")
        EA, EI, eta = propiedades[tipo]
        if not con_brazos:
            ai = aj = 0.0
        elif tipo == "viga":
            ai = aj = bv
        else:
            ai = 0.0 if np.isclose(float(nodos[i][1]), 0.0) else bc
            aj = bc

        L, _ = geometria_elemento(nodos[i], nodos[j])
        validar_brazos_rigidos(L, ai, aj)
        elementos.append({
            **e,
            "EA": EA,
            "EI": EI,
            "brazo_i_cm": ai,
            "brazo_j_cm": aj,
            "eta": eta,
        })
    return elementos


def crear_modelo_portico_2d(
    luces_cm,
    alturas_cm,
    EA_columna,
    EI_columna,
    EA_viga=None,
    EI_viga=None,
    brazo_columna_cm=0.0,
    brazo_viga_cm=0.0,
    eta_columna=1.0,
    eta_viga=1.0,
    con_brazos=True,
):
    """Construye geometría, conectividad, propiedades, GDL y grupos de nodos."""
    luces = np.asarray(luces_cm, dtype=float)
    alturas = np.asarray(alturas_cm, dtype=float)
    nodos = generar_nodos_portico(luces, alturas)
    n_vanos, n_pisos = int(luces.size), int(alturas.size)
    conectividad = generar_conectividad_portico(n_vanos, n_pisos)
    elementos = construir_elementos_portico(
        nodos, conectividad,
        EA_columna, EI_columna, EA_viga, EI_viga,
        brazo_columna_cm, brazo_viga_cm,
        eta_columna, eta_viga, con_brazos,
    )
    return {
        "n_vanos": n_vanos,
        "n_pisos": n_pisos,
        "luces_cm": luces,
        "alturas_cm": alturas,
        "nodos": nodos,
        "conectividad": conectividad,
        "elementos": elementos,
        "gdl": numerar_gdl(nodos),
        "nodos_base": nodos_de_nivel(n_vanos, 0),
        "nodos_techo": nodos_de_nivel(n_vanos, n_pisos),
    }


def _normalizar_separaciones_porticos(n_porticos, separaciones_porticos_cm):
    separaciones = np.asarray(separaciones_porticos_cm, dtype=float)
    if separaciones.ndim == 0:
        separaciones = np.full(n_porticos - 1, float(separaciones))
    if separaciones.shape != (n_porticos - 1,) or np.any(separaciones <= 0):
        raise ValueError(
            "separaciones_porticos_cm debe ser un escalar o un vector de "
            "longitud n_porticos-1 con valores positivos."
        )
    return separaciones


def generar_puntales_transferencia(
    n_porticos: int,
    n_vanos: int,
    n_pisos: int,
    n_nodos_local: int,
    EA_puntal_kgf: float,
    separaciones_porticos_cm,
    incluir_base: bool = False,
):
    """Genera un puntal horizontal entre pórticos vecinos por cada nivel.

    Cada puntal conecta el nudo derecho del pórtico ``p`` con el nudo izquierdo
    del pórtico ``p+1``. Todos los pórticos y puntales pertenecen al mismo plano
    global 2D. Su rigidez axial es ``k=EA/L`` y actúa sobre los GDL ``u``.
    """
    if not isinstance(n_porticos, (int, np.integer)) or n_porticos < 2:
        raise ValueError("Se requieren al menos dos pórticos para generar puntales.")
    if not isinstance(n_vanos, (int, np.integer)) or n_vanos < 1:
        raise ValueError("n_vanos debe ser un entero positivo.")
    if not isinstance(n_pisos, (int, np.integer)) or n_pisos < 1:
        raise ValueError("n_pisos debe ser un entero positivo.")
    EA = float(EA_puntal_kgf)
    if not np.isfinite(EA) or EA <= 0:
        raise ValueError("EA_puntal_kgf debe ser positivo y finito.")

    separaciones = _normalizar_separaciones_porticos(
        n_porticos, separaciones_porticos_cm
    )

    niveles = range(0 if incluir_base else 1, n_pisos + 1)
    puntales = []
    pid = 1
    for nivel in niveles:
        nodos_locales = nodos_de_nivel(n_vanos, nivel)
        nodo_derecho = nodos_locales[-1]
        nodo_izquierdo = nodos_locales[0]
        for p in range(n_porticos - 1):
            L = float(separaciones[p])
            ni = p * n_nodos_local + nodo_derecho
            nj = (p + 1) * n_nodos_local + nodo_izquierdo
            puntales.append({
                "id": pid,
                "tipo": "puntal_transferencia",
                "i": ni,
                "j": nj,
                "portico_i": p + 1,
                "portico_j": p + 2,
                "nivel": nivel,
                "linea_i": n_vanos + 1,
                "linea_j": 1,
                "EA_kgf": EA,
                "L_cm": L,
                "k_axial_kgf_cm": EA / L,
                "modelo": "barra_axial_horizontal_2d",
            })
            pid += 1
    return puntales


def ensamblar_puntales_transferencia(K, puntales, gdl):
    """Añade la rigidez de puntales a una matriz ya ensamblada."""
    K_total = np.asarray(K, dtype=float).copy()
    resultados = {}
    for puntal in puntales:
        i, j = int(puntal["i"]), int(puntal["j"])
        if i not in gdl or j not in gdl:
            raise KeyError("Un puntal referencia un nodo inexistente.")
        k = float(puntal["k_axial_kgf_cm"])
        if k <= 0 or not np.isfinite(k):
            raise ValueError("La rigidez de un puntal debe ser positiva y finita.")
        vc = np.array([gdl[i][0], gdl[j][0]], dtype=int)
        K_total[np.ix_(vc, vc)] += k * np.array([[1.0, -1.0], [-1.0, 1.0]])
        resultados[int(puntal["id"])] = {"vc": vc, "k_axial_kgf_cm": k}
    return 0.5 * (K_total + K_total.T), resultados


def ensamblar_rigidez_edificio(nodos, elementos, puntales, gdl=None):
    """Ensambla elementos de pórtico y puntales de transferencia."""
    K_marcos, resultados_elemento, gdl = ensamblar_rigidez_global(
        nodos, elementos, gdl
    )
    K, resultados_puntal = ensamblar_puntales_transferencia(
        K_marcos, puntales, gdl
    )
    return K, resultados_elemento, resultados_puntal, gdl


def crear_modelo_edificio_porticos(
    datos_porticos: Sequence[Mapping],
    EA_puntal_kgf=1.0,
    separaciones_porticos_cm=1.0,
    incluir_puntales_en_base=False,
):
    """Construye una serie 2D a partir de pórticos definidos individualmente.

    Cada entrada de ``datos_porticos`` debe contener, como mínimo,
    ``luces_cm``, ``alturas_cm``, ``EA_columna`` y ``EI_columna``. Puede
    especificar propiedades distintas de vigas, brazos rígidos, factores
    ``eta`` y un diccionario ``propiedades_elementos`` indexado por el id local
    del elemento. Los pórticos deben compartir las mismas elevaciones de piso
    para que los puntales que los conectan sean horizontales.
    """
    if not isinstance(datos_porticos, Sequence) or isinstance(datos_porticos, (str, bytes)):
        raise TypeError("datos_porticos debe ser una secuencia de diccionarios.")
    if len(datos_porticos) < 2:
        raise ValueError("Se requieren al menos dos pórticos definidos individualmente.")

    modelos_locales = []
    ids_porticos = []
    for indice, datos in enumerate(datos_porticos, start=1):
        if not isinstance(datos, Mapping):
            raise TypeError("Cada pórtico debe definirse mediante un diccionario.")
        faltantes = {
            "luces_cm", "alturas_cm", "EA_columna", "EI_columna"
        } - set(datos)
        if faltantes:
            raise KeyError(f"Pórtico {indice}: faltan {sorted(faltantes)}.")
        pid = str(datos.get("id", f"P{indice}"))
        if pid in ids_porticos:
            raise ValueError(f"Identificador de pórtico repetido: {pid!r}.")
        ids_porticos.append(pid)
        local = crear_modelo_portico_2d(
            datos["luces_cm"], datos["alturas_cm"],
            datos["EA_columna"], datos["EI_columna"],
            datos.get("EA_viga"), datos.get("EI_viga"),
            datos.get("brazo_columna_cm", 0.0),
            datos.get("brazo_viga_cm", 0.0),
            datos.get("eta_columna", 1.0), datos.get("eta_viga", 1.0),
            datos.get("con_brazos", True),
        )
        overrides = datos.get("propiedades_elementos", {})
        if not isinstance(overrides, Mapping):
            raise TypeError(f"Pórtico {pid}: propiedades_elementos debe ser un diccionario.")
        for elemento in local["elementos"]:
            cambio = overrides.get(elemento["id"], overrides.get(str(elemento["id"]), {}))
            if cambio:
                permitidas = {"EA", "EI", "eta", "brazo_i_cm", "brazo_j_cm"}
                desconocidas = set(cambio) - permitidas
                if desconocidas:
                    raise KeyError(
                        f"Pórtico {pid}, elemento {elemento['id']}: "
                        f"propiedades no admitidas {sorted(desconocidas)}."
                    )
                elemento.update({k: float(v) for k, v in cambio.items()})
        local["id_portico"] = pid
        modelos_locales.append(local)

    n_porticos = len(modelos_locales)
    n_pisos = modelos_locales[0]["n_pisos"]
    elevaciones_ref = np.r_[0.0, np.cumsum(modelos_locales[0]["alturas_cm"])]
    for local in modelos_locales[1:]:
        elevaciones = np.r_[0.0, np.cumsum(local["alturas_cm"])]
        if elevaciones.shape != elevaciones_ref.shape or not np.allclose(
            elevaciones, elevaciones_ref
        ):
            raise ValueError(
                "Todos los pórticos deben compartir las mismas elevaciones "
                "de piso para conectarse mediante puntales horizontales."
            )

    separaciones = _normalizar_separaciones_porticos(
        n_porticos, separaciones_porticos_cm
    )
    anchos = np.array([float(np.sum(m["luces_cm"])) for m in modelos_locales])
    origenes_x = np.zeros(n_porticos)
    for p in range(1, n_porticos):
        origenes_x[p] = origenes_x[p - 1] + anchos[p - 1] + separaciones[p - 1]

    nodos = {}
    elementos = []
    nodos_por_portico = {}
    elementos_por_portico = {}
    offset_nodo = 0
    offset_elemento = 0
    for p, local in enumerate(modelos_locales, start=1):
        nodos_por_portico[p] = {}
        elementos_por_portico[p] = []
        for nid, coord in local["nodos"].items():
            gid = offset_nodo + int(nid)
            coord_global = np.asarray(coord, dtype=float).copy()
            coord_global[0] += origenes_x[p - 1]
            nodos[gid] = coord_global
            nodos_por_portico[p][int(nid)] = gid
        for e in local["elementos"]:
            eg = {
                **e,
                "id": offset_elemento + int(e["id"]),
                "id_local": int(e["id"]),
                "i": offset_nodo + int(e["i"]),
                "j": offset_nodo + int(e["j"]),
                "portico": p,
                "id_portico": local["id_portico"],
            }
            elementos.append(eg)
            elementos_por_portico[p].append(eg)
        offset_nodo += len(local["nodos"])
        offset_elemento += len(local["elementos"])

    EA_puntal = float(EA_puntal_kgf)
    if EA_puntal <= 0 or not np.isfinite(EA_puntal):
        raise ValueError("EA_puntal_kgf debe ser positivo y finito.")
    niveles = range(0 if incluir_puntales_en_base else 1, n_pisos + 1)
    puntales = []
    for nivel in niveles:
        for p in range(1, n_porticos):
            local_i, local_j = modelos_locales[p - 1], modelos_locales[p]
            nid_i = nodos_de_nivel(local_i["n_vanos"], nivel)[-1]
            nid_j = nodos_de_nivel(local_j["n_vanos"], nivel)[0]
            L = float(separaciones[p - 1])
            puntales.append({
                "id": len(puntales) + 1,
                "tipo": "puntal_transferencia",
                "i": nodos_por_portico[p][nid_i],
                "j": nodos_por_portico[p + 1][nid_j],
                "portico_i": p,
                "portico_j": p + 1,
                "nivel": nivel,
                "EA_kgf": EA_puntal,
                "L_cm": L,
                "k_axial_kgf_cm": EA_puntal / L,
                "modelo": "barra_axial_horizontal_2d",
            })

    gdl = numerar_gdl(nodos)
    nodos_base = [
        nodos_por_portico[p][nid]
        for p, local in enumerate(modelos_locales, start=1)
        for nid in local["nodos_base"]
    ]
    nodos_techo = [
        nodos_por_portico[p][nid]
        for p, local in enumerate(modelos_locales, start=1)
        for nid in local["nodos_techo"]
    ]
    nodos_por_nivel = {
        nivel: [
            nodos_por_portico[p][nid]
            for p, local in enumerate(modelos_locales, start=1)
            for nid in nodos_de_nivel(local["n_vanos"], nivel)
        ]
        for nivel in range(n_pisos + 1)
    }
    return {
        "n_porticos": n_porticos,
        "ids_porticos": ids_porticos,
        "n_vanos_por_portico": [m["n_vanos"] for m in modelos_locales],
        "n_pisos": n_pisos,
        "luces_cm_por_portico": [m["luces_cm"] for m in modelos_locales],
        "alturas_cm_por_portico": [m["alturas_cm"] for m in modelos_locales],
        "separaciones_porticos_cm": separaciones,
        "origenes_porticos_x_cm": origenes_x,
        "anchos_porticos_cm": anchos,
        "nodos": nodos,
        "elementos": elementos,
        "puntales": puntales,
        "gdl": gdl,
        "nodos_base": nodos_base,
        "nodos_techo": nodos_techo,
        "nodos_por_nivel": nodos_por_nivel,
        "nodos_por_portico": nodos_por_portico,
        "elementos_por_portico": elementos_por_portico,
        "modelos_locales": modelos_locales,
        "restricciones_gdl": np.concatenate([gdl[nid] for nid in nodos_base]),
        "n_nodos_por_portico": [len(m["nodos"]) for m in modelos_locales],
    }


def numerar_gdl(nodos: Mapping[int, Sequence[float]]):
    """Asigna [u, v, theta] consecutivos a cada nodo ordenado."""
    return {nid: np.arange(3 * k, 3 * k + 3, dtype=int) for k, nid in enumerate(sorted(nodos))}


def vector_colocacion(nodo_i: int, nodo_j: int, gdl: Mapping[int, np.ndarray]):
    """Vector de colocación de un elemento de pórtico 2D."""
    if nodo_i not in gdl or nodo_j not in gdl:
        raise KeyError("Los nodos del elemento deben existir en la numeración de GDL.")
    return np.r_[gdl[nodo_i], gdl[nodo_j]]


def geometria_elemento(coord_i: Sequence[float], coord_j: Sequence[float]):
    """Devuelve longitud entre ejes y ángulo global del elemento."""
    ci, cj = np.asarray(coord_i, dtype=float), np.asarray(coord_j, dtype=float)
    delta = cj - ci
    L = float(np.linalg.norm(delta))
    if L <= 0:
        raise ValueError("La longitud entre ejes debe ser positiva.")
    return L, float(np.arctan2(delta[1], delta[0]))


def validar_brazos_rigidos(L_ejes_cm: float, brazo_i_cm=0.0, brazo_j_cm=0.0):
    """Valida offsets colineales y devuelve la longitud entre caras."""
    L, ai, aj = map(float, (L_ejes_cm, brazo_i_cm, brazo_j_cm))
    if L <= 0 or ai < 0 or aj < 0:
        raise ValueError("L debe ser positiva y los brazos rígidos no negativos.")
    Lf = L - ai - aj
    if Lf <= 0:
        raise ValueError("Los brazos rígidos ocupan toda la longitud del elemento.")
    return Lf


def matriz_brazos_rigidos_local(brazo_i_cm=0.0, brazo_j_cm=0.0):
    """Transforma GDL de ejes a GDL de caras en coordenadas locales.

    d_cara = R @ d_eje, con v_ci=v_i+ai*theta_i y
    v_cj=v_j-aj*theta_j.
    """
    ai, aj = float(brazo_i_cm), float(brazo_j_cm)
    if ai < 0 or aj < 0:
        raise ValueError("Los brazos rígidos no pueden ser negativos.")
    R = np.eye(6)
    R[1, 2] = ai
    R[4, 5] = -aj
    return R


def normalizar_ubicacion_rotula(
    L_ejes_cm: float,
    brazo_i_cm: float,
    brazo_j_cm: float,
    extremo: str,
    distancia_cm: float = 0.0,
    referencia: str = "cara_apoyo",
    tol: float = 1e-9,
):
    """Normaliza una rótula a x medido desde el eje i.

    `distancia_cm` se mide desde el apoyo indicado hacia el interior del
    elemento. Se aceptan las referencias `cara_apoyo` y `eje_apoyo`.
    """
    L = float(L_ejes_cm)
    ai, aj = float(brazo_i_cm), float(brazo_j_cm)
    validar_brazos_rigidos(L, ai, aj)
    extremo = str(extremo).lower()
    referencia = str(referencia).lower()
    d = float(distancia_cm)
    if extremo not in {"i", "j"}:
        raise ValueError("extremo debe ser 'i' o 'j'.")
    if referencia not in {"cara_apoyo", "eje_apoyo"}:
        raise ValueError("referencia debe ser 'cara_apoyo' o 'eje_apoyo'.")
    if d < 0:
        raise ValueError("La distancia de la rótula no puede ser negativa.")

    cara_i, cara_j = ai, L - aj
    if extremo == "i":
        x = ai + d if referencia == "cara_apoyo" else d
        d_eje, d_cara = x, x - cara_i
    else:
        x = cara_j - d if referencia == "cara_apoyo" else L - d
        d_eje, d_cara = L - x, cara_j - x

    if x < cara_i - tol or x > cara_j + tol:
        raise ValueError("La rótula debe ubicarse dentro de la zona deformable entre caras.")
    x = min(max(x, cara_i), cara_j)
    return UbicacionRotula(
        extremo=extremo,
        referencia_entrada=referencia,
        distancia_entrada_cm=d,
        x_desde_eje_i_cm=float(x),
        distancia_desde_eje_cm=float(d_eje),
        distancia_desde_cara_cm=float(d_cara),
    )


def segmentar_por_rotulas(
    L_ejes_cm: float,
    brazo_i_cm: float,
    brazo_j_cm: float,
    ubicaciones: Iterable[UbicacionRotula],
    tol: float = 1e-9,
):
    """Devuelve estaciones y longitudes elásticas entre caras y rótulas."""
    validar_brazos_rigidos(L_ejes_cm, brazo_i_cm, brazo_j_cm)
    xs = [float(brazo_i_cm)]
    xs.extend(float(u.x_desde_eje_i_cm) for u in ubicaciones)
    xs.append(float(L_ejes_cm - brazo_j_cm))
    xs = sorted(xs)
    unicos = [xs[0]]
    for x in xs[1:]:
        if abs(x - unicos[-1]) > tol:
            unicos.append(x)
    longitudes = np.diff(unicos)
    if np.any(longitudes <= tol):
        raise ValueError("Existen estaciones coincidentes o segmentos sin longitud.")
    return {"estaciones_cm": np.asarray(unicos), "longitudes_cm": longitudes}


def rigidez_local_brazos_rigidos(EA, EI, L_ejes_cm, brazo_i_cm=0.0, brazo_j_cm=0.0):
    """Rigidez local 6x6 entre ejes con zona flexible entre caras."""
    Lf = validar_brazos_rigidos(L_ejes_cm, brazo_i_cm, brazo_j_cm)
    kf = matriz_rigidez_local_portico2d(EA, EI, Lf)
    R = matriz_brazos_rigidos_local(brazo_i_cm, brazo_j_cm)
    return R.T @ kf @ R


def _fila(nq, expresion):
    fila = np.zeros(nq)
    for indice, coeficiente in expresion.items():
        fila[indice] += coeficiente
    return fila


def rigidez_local_segmentada(
    EA,
    EI,
    L_ejes_cm,
    brazo_i_cm=0.0,
    brazo_j_cm=0.0,
    rotulas: Sequence[Mapping[str, float]] | None = None,
):
    """Super-elemento local con rótulas tangentes en posiciones arbitrarias.

    Cada rótula se define mediante `x_cm` y `k_theta_kgf_cm_rad`. Una lista
    vacía representa el estado elástico sin flexibilidad adicional. Las
    variables internas se condensan para devolver una matriz 6x6 entre ejes.
    """
    EA, EI = float(EA), float(EI)
    L = float(L_ejes_cm)
    ai, aj = float(brazo_i_cm), float(brazo_j_cm)
    validar_brazos_rigidos(L, ai, aj)
    if EA <= 0 or EI <= 0:
        raise ValueError("EA y EI deben ser positivos.")

    datos_rotulas = []
    for r in rotulas or []:
        x = float(r["x_cm"])
        k = float(r["k_theta_kgf_cm_rad"])
        if x < ai - 1e-9 or x > L - aj + 1e-9:
            raise ValueError("Una rótula está fuera de la zona deformable.")
        if k < 0 or not np.isfinite(k):
            raise ValueError("La rigidez tangente de la rótula debe ser no negativa y finita.")
        datos_rotulas.append((x, k))
    datos_rotulas.sort()
    for (x1, _), (x2, _) in zip(datos_rotulas, datos_rotulas[1:]):
        if abs(x2 - x1) <= 1e-9:
            raise ValueError("No se permiten dos rótulas en la misma estación.")

    # Cada estación se expresa como combinaciones lineales de q. Los primeros
    # seis q son los GDL externos en los ejes; el resto son GDL internos.
    siguiente = 6
    estaciones = []
    resortes = []
    posiciones = [ai, *[x for x, _ in datos_rotulas], L - aj]
    posiciones_unicas = []
    for x in posiciones:
        if not posiciones_unicas or abs(x - posiciones_unicas[-1]) > 1e-9:
            posiciones_unicas.append(x)

    mapa_k = {round(x, 9): k for x, k in datos_rotulas}
    for n, x in enumerate(posiciones_unicas):
        es_i, es_j = n == 0, n == len(posiciones_unicas) - 1
        tiene_rotula = round(x, 9) in mapa_k
        if es_i:
            u = {0: 1.0}
            v = {1: 1.0, 2: ai}
            theta_exterior = {2: 1.0}
            theta_izq = theta_exterior
            if tiene_rotula:
                theta_der = {siguiente: 1.0}
                siguiente += 1
                resortes.append((theta_izq, theta_der, mapa_k[round(x, 9)]))
            else:
                theta_der = theta_exterior
        elif es_j:
            u = {3: 1.0}
            v = {4: 1.0, 5: -aj}
            theta_exterior = {5: 1.0}
            theta_der = theta_exterior
            if tiene_rotula:
                theta_izq = {siguiente: 1.0}
                siguiente += 1
                resortes.append((theta_izq, theta_der, mapa_k[round(x, 9)]))
            else:
                theta_izq = theta_exterior
        else:
            u, v = {siguiente: 1.0}, {siguiente + 1: 1.0}
            siguiente += 2
            theta_izq, theta_der = {siguiente: 1.0}, {siguiente + 1: 1.0}
            siguiente += 2
            resortes.append((theta_izq, theta_der, mapa_k[round(x, 9)]))
        estaciones.append({"x": x, "u": u, "v": v, "ti": theta_izq, "td": theta_der})

    nq = siguiente
    K = np.zeros((nq, nq))
    for izquierda, derecha in zip(estaciones, estaciones[1:]):
        Le = derecha["x"] - izquierda["x"]
        if Le <= 1e-9:
            raise ValueError("Un segmento elástico no tiene longitud positiva.")
        B = np.vstack([
            _fila(nq, izquierda["u"]),
            _fila(nq, izquierda["v"]),
            _fila(nq, izquierda["td"]),
            _fila(nq, derecha["u"]),
            _fila(nq, derecha["v"]),
            _fila(nq, derecha["ti"]),
        ])
        K += B.T @ matriz_rigidez_local_portico2d(EA, EI, Le) @ B

    for theta_izq, theta_der, ktheta in resortes:
        b = _fila(nq, theta_der) - _fila(nq, theta_izq)
        K += ktheta * np.outer(b, b)

    if nq == 6:
        Kc = K
        cond_kii = None
    else:
        Kee, Kei = K[:6, :6], K[:6, 6:]
        Kie, Kii = K[6:, :6], K[6:, 6:]
        cond_kii = float(np.linalg.cond(Kii))
        Kc = Kee - Kei @ np.linalg.solve(Kii, Kie)
    return {
        "local": 0.5 * (Kc + Kc.T),
        "n_gdl_internos": nq - 6,
        "condicion_Kii": cond_kii,
        "estaciones_cm": np.array(posiciones_unicas),
    }


def rigidez_elemento_modelado(
    EA,
    EI,
    coord_i,
    coord_j,
    brazo_i_cm=0.0,
    brazo_j_cm=0.0,
    eta=1.0,
    rotulas=None,
):
    """Rigidez local/global de un miembro con offsets y rótulas tangentes."""
    if not (0 < eta <= 1):
        raise ValueError("eta debe satisfacer 0 < eta <= 1.")
    L, angulo = geometria_elemento(coord_i, coord_j)
    resultado = rigidez_local_segmentada(
        EA, eta * EI, L, brazo_i_cm, brazo_j_cm, rotulas=rotulas
    )
    T = matriz_transformacion_portico2d(angulo)
    resultado.update({
        "global": T.T @ resultado["local"] @ T,
        "T": T,
        "L_ejes_cm": L,
        "L_entre_caras_cm": L - brazo_i_cm - brazo_j_cm,
        "angulo_rad": angulo,
        "EI_t": eta * EI,
    })
    return resultado


def ensamblar_rigidez_global(nodos, elementos, gdl=None):
    """Ensambla K global; cada elemento es un diccionario validado."""
    gdl = numerar_gdl(nodos) if gdl is None else gdl
    n_gdl = 3 * len(nodos)
    K = np.zeros((n_gdl, n_gdl))
    resultados = {}
    for e in elementos:
        i, j = int(e["i"]), int(e["j"])
        res = rigidez_elemento_modelado(
            e["EA"], e["EI"], nodos[i], nodos[j],
            e.get("brazo_i_cm", 0.0), e.get("brazo_j_cm", 0.0),
            e.get("eta", 1.0), e.get("rotulas_tangentes"),
        )
        vc = vector_colocacion(i, j, gdl)
        K[np.ix_(vc, vc)] += res["global"]
        resultados[int(e["id"])] = {**res, "vc": vc}
    return K, resultados, gdl

def resolver_sistema_restringido(K, cargas, gdl_restringidos):
    """Resuelve desplazamientos y reacciones sin formar la inversa."""
    K = np.asarray(K, dtype=float)
    P = np.asarray(cargas, dtype=float)
    if K.shape[0] != K.shape[1] or P.shape != (K.shape[0],):
        raise ValueError("Dimensiones incompatibles entre K y cargas.")
    restringidos = np.unique(np.asarray(gdl_restringidos, dtype=int))
    libres = np.setdiff1d(np.arange(K.shape[0]), restringidos)
    Kll = K[np.ix_(libres, libres)]
    if np.linalg.matrix_rank(Kll) < len(libres):
        raise np.linalg.LinAlgError("La estructura restringida conserva un mecanismo.")
    u = np.zeros(K.shape[0])
    u[libres] = np.linalg.solve(Kll, P[libres])
    reacciones = K @ u - P
    return {"u": u, "reacciones": reacciones, "libres": libres, "restringidos": restringidos}


__all__ = [
    "UbicacionRotula", "normalizar_geometria_edificio", "cargar_geometria_edificio",
    "generar_nodos_portico", "generar_conectividad_portico",
    "nodos_de_nivel", "construir_elementos_portico", "crear_modelo_portico_2d",
    "generar_puntales_transferencia", "ensamblar_puntales_transferencia",
    "ensamblar_rigidez_edificio", "crear_modelo_edificio_porticos",
    "numerar_gdl", "vector_colocacion", "geometria_elemento",
    "validar_brazos_rigidos", "matriz_brazos_rigidos_local",
    "normalizar_ubicacion_rotula", "segmentar_por_rotulas",
    "rigidez_local_brazos_rigidos", "rigidez_local_segmentada",
    "rigidez_elemento_modelado", "ensamblar_rigidez_global",
    "resolver_sistema_restringido",
]
