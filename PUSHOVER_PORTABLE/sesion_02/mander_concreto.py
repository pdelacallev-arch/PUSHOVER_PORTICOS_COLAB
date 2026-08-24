# -*- coding: utf-8 -*-
"""
=============================================================================
 CURVA ESFUERZO-DEFORMACION DEL CONCRETO SEGUN EL MODELO DE MANDER (1988)
          Concreto confinado y concreto no confinado
=============================================================================
 Curso:  Analisis Pushover de Porticos de Concreto Armado con Python
 Sesion: 2 - Modelamiento constitutivo del concreto
 Unidades: esfuerzos en kg/cm2, longitudes en cm (como en el resto del curso)
 Librerias: solo NumPy (calculo) y Matplotlib (graficos)

 QUE HACE ESTE PROGRAMA
 ----------------------
 1. Implementa la curva sigma-epsilon de Mander et al. (1988):
       fc = f'cc * x * r / (r - 1 + x^r)          (curva de Popovics, 1973)
    donde x = epsilon / eps'cc  y  r = Ec / (Ec - Esec)
 2. Calcula la resistencia del concreto CONFINADO:
       f'cc = f'co * ( -1.254 + 2.254*sqrt(1 + 7.94*f'l/f'co) - 2*f'l/f'co )
       eps'cc = eps'co * ( 1 + 5 * (f'cc/f'co - 1) )
       eps_cu = 0.004 + 1.4 * rho_s * fyh * eps_su / f'cc
 3. Calcula la presion lateral de confinamiento f'l a partir de los
    estribos (As, espaciamiento s, nucleo bc x dc, esfuerzo fyh) y el
    factor de efectividad del confinamiento ke.
 4. Grafica y compara las curvas confinada y no confinada.

 COMO USARLO
 -----------
   python mander_concreto.py
 El programa resuelve un ejemplo paso a paso e imprime cada resultado
 intermedio para que puedas auditar el procedimiento.

 REFERENCIA
 ----------
 Mander, J. B., Priestley, M. J. N., & Park, R. (1988).
 "Theoretical Stress-Strain Model for Confined Concrete".
 Journal of Structural Engineering, ASCE, 114(8), 1804-1826.
=============================================================================
"""

# =============================================================================
# 0. IMPORTACION DE LIBRERIAS
#    Solo usamos NumPy para los calculos con vectores y Matplotlib para
#    el grafico final. No se necesita ninguna libreria adicional.
# =============================================================================
import numpy as np


# =============================================================================
# NIVEL 1 - LA CURVA EN SI (la parte mas importante)
# =============================================================================

def modulo_elasticidad(fco):
    """
    Modulo de elasticidad inicial del concreto Ec (kg/cm2).

    Ec = 15000 * sqrt(f'co)

    Esta expresion es la version en kg/cm2 de la formula clasica del ACI:
        Ec = 4700 * sqrt(f'c)  (en MPa, con f'c en MPa)

    Parametros
    ----------
    fco : float
        Resistencia a compresion del concreto no confinado f'co (kg/cm2).

    Retorna
    -------
    Ec : float
        Modulo de elasticidad (kg/cm2).

    Ejemplo: f'co = 210 kg/cm2  ->  Ec = 15000 * sqrt(210) = 217,370 kg/cm2
    """
    Ec = 15000.0 * np.sqrt(fco)
    return Ec


def curva_popovics(eps, fcc, epscc, Ec=None):
    """
    Curva sigma-epsilon de Popovics (1973) usada por Mander para ambos
    concretos (confinado y no confinado):

        fc = f'cc * x * r / (r - 1 + x^r)

    donde:
        x = epsilon / eps'cc           (deformacion normalizada)
        r = Ec / (Ec - Esec)           (factor de forma de la curva)
        Esec = f'cc / eps'cc           (modulo secante en el pico)

    Comprobaciones rapidas de la formula:
        * En epsilon = 0        -> x = 0  -> fc = 0      (concreto sin carga)
        * En epsilon = eps'cc   -> x = 1  -> fc = f'cc   (el pico de la curva)
        * Si r -> grande        -> curva casi elastica hasta el pico
        * Si r -> pequeno       -> rama descendente muy suave

    Parametros
    ----------
    eps  : ndarray
        Vector de deformaciones unitarias (adimensional, cm/cm).
    fcc  : float
        Resistencia pico de la curva: f'co para no confinado,
        f'cc (confinado) para concreto confinado (kg/cm2).
    epscc : float
        Deformacion en el pico: eps'co o eps'cc (adimensional).
    Ec   : float, opcional
        Modulo de elasticidad (kg/cm2). Si no se entrega, se calcula
        con modulo_elasticidad() usando fcc como si fuera f'co.

    Retorna
    -------
    sigma : ndarray
        Vector de esfuerzos de compresion fc (kg/cm2), mismo tamano que eps.
    """
    # Si el usuario no dio Ec, lo calculamos (solo se usa la parte inicial
    # de la curva, por eso no importa que fcc > f'co).
    if Ec is None:
        Ec = modulo_elasticidad(fcc)

    # Modulo secante: pendiente de la recta que une el origen con el pico
    Esec = fcc / epscc

    # Factor de forma r (siempre mayor que 1 para que exista rama descendente)
    r = Ec / (Ec - Esec)

    # Deformacion normalizada x
    x = eps / epscc

    # Ecuacion de Popovics aplicada a TODO el vector eps de una sola vez
    # (NumPy trabaja con vectores, no necesitamos un "for")
    sigma = fcc * x * r / (r - 1.0 + x**r)

    # El concreto no resiste traccion: si epsilon < 0, el esfuerzo es 0.
    # La formula de arriba daria valores negativos (absurdos fisicamente).
    sigma = np.where(eps < 0.0, 0.0, sigma)

    return sigma


def curva_mander_no_confinado(eps, fco, epsco=0.002, eps_sp=0.005, Ec=None):
    """
    Curva sigma-epsilon del concreto NO CONFINADO (recubrimiento).

    Segun Mander et al. (1988), se compone de DOS partes:
      1. RAMA PRINCIPAL (0 <= eps <= 2*eps'co): curva de Popovics con
         pico en (eps'co, f'co).
      2. RAMA DESCENDENTE LINEAL (2*eps'co < eps <= eps_sp): linea recta 
         desde la curva en 2*eps'co hasta cero en eps_sp. Despues de 
         eps_sp el recubrimiento se desprende (spalling) y fc = 0.

    Parametros
    ----------
    eps    : ndarray
        Vector de deformaciones (adimensional).
    fco    : float
        Resistencia a compresion del concreto f'co (kg/cm2).
    epsco  : float, opcional
        Deformacion en el pico del no confinado, típicamente 0.002.
    eps_sp : float, opcional
        Deformacion de desprendimiento del recubrimiento (spalling),
        típicamente 0.005 segun la literatura para este modelo.
    Ec     : float, opcional
        Modulo de elasticidad (kg/cm2).

    Retorna
    -------
    sigma : ndarray
        Vector de esfuerzos (kg/cm2).
    """
    if Ec is None:
        Ec = modulo_elasticidad(fco)

    # --- Rama 1: Popovics hasta 2*eps'co --------------------------------
    rama_popovics = curva_popovics(eps, fco, epsco, Ec)

    # --- Rama 2: recta descendente desde 2*eps'co hasta eps_sp ----------
    # Esfuerzo en 2*epsco usando Popovics
    eps_limite = 2.0 * epsco
    f_limite = curva_popovics(np.array([eps_limite]), fco, epsco, Ec)[0]
    
    # Prevenimos division por cero por si eps_sp es muy pequeno
    eps_sp_efectivo = max(eps_sp, eps_limite + 0.0001)
    
    # Pendiente de la recta
    pendiente = (0.0 - f_limite) / (eps_sp_efectivo - eps_limite)
    rama_descendente = f_limite + pendiente * (eps - eps_limite)

    # --- Combinamos las ramas con np.where ------------------------------
    sigma = np.where(eps <= eps_limite,
                     rama_popovics,
                     np.where(eps <= eps_sp_efectivo, rama_descendente, 0.0))

    return sigma


def curva_mander_confinado(eps, fco, fl, epsco=0.002,
                           rho_s=0.0, fyh=4200.0, eps_su=0.12, Ec=None):
    """
    Curva sigma-epsilon del concreto CONFINADO (nucleo).

    Se compone de DOS partes:
      1. Curva de Popovics con el pico elevado (eps'cc, f'cc) calculado
         con resistencia_confinada() a partir de la presion lateral f'l.
      2. Corte en la deformacion ultima eps_cu (Mander, ecuacion 6):
            eps_cu = 0.004 + 1.4 * rho_s * fyh * eps_su / f'cc
         Despues de eps_cu el acero transversal (estribo) ya fallo y el
         nucleo pierde confinamiento: suponemos fc = 0 (simplificacion
         didactica; Mander modela una perdida gradual de resistencia).

    Parametros
    ----------
    eps   : ndarray
        Vector de deformaciones (adimensional).
    fco   : float
        Resistencia del concreto no confinado f'co (kg/cm2).
    fl    : float
        Presion lateral de confinamiento EFECTIVA f'l (kg/cm2).
        Se obtiene con presion_lateral() y factor_efectividad().
    epsco : float, opcional
        Deformacion en el pico del no confinado (0.002 tipico).
    rho_s : float, opcional
        Relacion volumetrica de acero transversal (adimensional).
    fyh   : float, opcional
        Esfuerzo de fluencia de los estribos (kg/cm2).
    eps_su : float, opcional
        Deformacion ultima del acero de los estribos (adimensional).
    Ec    : float, opcional
        Modulo de elasticidad (kg/cm2).

    Retorna
    -------
    sigma : ndarray
        Vector de esfuerzos (kg/cm2).
    """
    if Ec is None:
        Ec = modulo_elasticidad(fco)

    # 1) Resistencia pico del concreto confinado y su deformacion
    fcc, epscc = resistencia_confinada(fco, fl, epsco)

    # 2) Deformacion ultima (falla del estribo)
    epscu = deformacion_ultima(fcc, rho_s, fyh, eps_su)

    # 3) Curva de Popovics con el pico confinado
    sigma = curva_popovics(eps, fcc, epscc, Ec)

    # 4) Recorte: fuera de [0, eps_cu] el esfuerzo es cero
    sigma = np.where((eps < 0.0) | (eps > epscu), 0.0, sigma)

    return sigma


# =============================================================================
# NIVEL 2 - COMO SE OBTIENEN LOS PARAMETROS DE CONFINAMIENTO
#           (estas funciones alimentan a las del NIVEL 1)
# =============================================================================

def resistencia_confinada(fco, fl, epsco=0.002):
    """
    Resistencia f'cc y deformacion eps'cc del concreto confinado.

    Ecuaciones de Mander (1988), ecs. (4) y (5):

        f'cc = f'co * ( -1.254 + 2.254*sqrt(1 + 7.94*f'l/f'co)
                         - 2*f'l/f'co )

        eps'cc = eps'co * ( 1 + 5*(f'cc/f'co - 1) )

    Nota: si f'l = 0 (sin confinamiento) la primera ecuacion devuelve
    f'cc = f'co, como debe ser (se puede comprobar a mano).

    Parametros
    ----------
    fco   : float
        Resistencia del concreto no confinado (kg/cm2).
    fl    : float
        Presion lateral de confinamiento efectiva f'l (kg/cm2).
    epsco : float, opcional
        Deformacion en el pico del no confinado (0.002 tipico).

    Retorna
    -------
    fcc   : float
        Resistencia del concreto confinado (kg/cm2).
    epscc : float
        Deformacion en el pico del confinado (adimensional).
    """
    # Ecuacion (4) de Mander
    fcc = fco * (-1.254 + 2.254 * np.sqrt(1.0 + 7.94 * fl / fco)
                 - 2.0 * fl / fco)

    # Ecuacion (5) de Mander
    epscc = epsco * (1.0 + 5.0 * (fcc / fco - 1.0))

    return fcc, epscc


def presion_lateral(Asx, Asy, s, bc, dc, fyh):
    """
    Presion lateral de confinamiento NOMINAL ejercida por los estribos
    en cada direccion (Mander 1988, ec. 7):

        f'lx = Asx * fyh / (s * dc)     (direccion x, ramas que cubren dc)
        f'ly = Asy * fyh / (s * bc)     (direccion y, ramas que cubren bc)

    Interpretacion: es la fuerza que aportan las ramas del estribo
    (As*fyh) repartida en el area de nucleo que confinan (s*dimension).

    Parametros
    ----------
    Asx : float
        Area total de ramas del estribo en la direccion x (cm2).
        Ejemplo: estribo cerrado con 2 ramas de diametro d -> Asx = 2*pi*d^2/4
    Asy : float
        Area total de ramas del estribo en la direccion y (cm2).
    s   : float
        Espaciamiento vertical de los estribos (cm).
    bc  : float
        Dimension del nucleo en x, entre centros de estribo (cm).
    dc  : float
        Dimension del nucleo en y, entre centros de estribo (cm).
    fyh : float
        Esfuerzo de fluencia del acero del estribo (kg/cm2).

    Retorna
    -------
    flx : float
        Presion lateral nominal en x (kg/cm2).
    fly : float
        Presion lateral nominal en y (kg/cm2).
    """
    flx = Asx * fyh / (s * dc)
    fly = Asy * fyh / (s * bc)
    return flx, fly


def factor_efectividad_confinamiento(bc, dc, s, db_estribo,
                                     distancias_w, As_long_total):
    """
    Factor de efectividad del confinamiento ke (Mander 1988, ec. 8).
    Corrige la presion lateral nominal porque el arco de carga entre
    estribos y entre barras hace que parte del nucleo NO este confinado:

        ke = (1 - sum(w_i^2)/(6*bc*dc)) * (1 - s'/(2*bc)) * (1 - s'/(2*dc))
             ----------------------------------------------------------
                                   (1 - rho_cc)

    donde:
        s'   = s - diametro del estribo   (espaciamiento libre)
        w_i  = distancias libres (cara a cara) entre barras longitudinales
               confinadas perimetrales (cm)
        rho_cc = As_long_total / (bc*dc)  (cuantia longitudinal del nucleo)

    Parametros
    ----------
    bc, dc      : float
        Dimensiones del nucleo entre centros de estribo (cm).
    s           : float
        Espaciamiento de estribos (cm).
    db_estribo  : float
        Diametro de la barra del estribo (cm).
    distancias_w : lista
        Lista con las distancias libres w_i (cara a cara) entre barras 
        longitudinales perimetrales (cm). Debe incluir todas las caras.
    As_long_total : float
        Area total del refuerzo longitudinal (cm2).

    Retorna
    -------
    ke : float
        Factor de efectividad (entre 0 y 1).
    """
    # Espaciamiento libre entre estribos
    s_prima = s - db_estribo

    # Suma de los cuadrados de las distancias entre barras
    suma_w2 = 0.0
    for w_i in distancias_w:
        suma_w2 = suma_w2 + w_i**2

    # Cuantia de acero longitudinal respecto al nucleo
    rho_cc = As_long_total / (bc * dc)

    # Ecuacion (8) de Mander, en tres factores (mas facil de leer)
    f1 = 1.0 - suma_w2 / (6.0 * bc * dc)          # efecto entre barras
    f2 = (1.0 - s_prima / (2.0 * bc))             # efecto entre estribos (x)
    f3 = (1.0 - s_prima / (2.0 * dc))             # efecto entre estribos (y)
    denominador = 1.0 - rho_cc

    ke = f1 * f2 * f3 / denominador

    return ke


def relacion_volumetrica(Asx, Asy, s, bc, dc):
    """
    Relacion volumetrica de acero transversal rho_s (Mander 1988, ec. 6).

        rho_s = Asx/(s*dc) + Asy/(s*bc)

    Es el volumen de estribo dividido entre el volumen de nucleo
    (por unidad de altura). Se usa para calcular la deformacion ultima.

    Parametros
    ----------
    Asx, Asy : float
        Areas de ramas en cada direccion (cm2).
    s        : float
        Espaciamiento de estribos (cm).
    bc, dc   : float
        Dimensiones del nucleo (cm).

    Retorna
    -------
    rho_s : float
        Relacion volumetrica (adimensional).
    """
    rho_s = Asx / (s * dc) + Asy / (s * bc)
    return rho_s


def deformacion_ultima(fcc, rho_s, fyh, eps_su):
    """
    Deformacion ultima del concreto confinado eps_cu (Mander 1988, ec. 6),
    para el caso de falla del acero transversal:

        eps_cu = 0.004 + 1.4 * rho_s * fyh * eps_su / f'cc

    Parametros
    ----------
    fcc    : float
        Resistencia del concreto confinado (kg/cm2).
    rho_s  : float
        Relacion volumetrica de acero transversal (adimensional).
    fyh    : float
        Esfuerzo de fluencia del estribo (kg/cm2).
    eps_su : float
        Deformacion ultima del acero del estribo (adimensional).
        Tipico: 0.12 para acero grado 60.

    Retorna
    -------
    eps_cu : float
        Deformacion ultima (adimensional).
    """
    eps_cu = 0.004 + 1.4 * rho_s * fyh * eps_su / fcc
    return eps_cu


# =============================================================================
# 3. EJEMPLO PASO A PASO (se ejecuta al correr el archivo)
#    Columna de 40x40 cm con estribos cerrados de 3/8" @ 10 cm
#    y 4 barras longitudinales de 3/4". Concreto f'c = 210 kg/cm2.
# =============================================================================

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    print("=" * 70)
    print("  MODELO DE MANDER (1988): CONCRETO CONFINADO Y NO CONFINADO")
    print("=" * 70)

    # ------------------------------------------------------------------
    # PASO 1. DATOS DE LA SECCION DEL EJEMPLO
    #         (cambia estos valores para probar otra columna)
    # ------------------------------------------------------------------
    fco = 210.0            # f'co: resistencia del concreto (kg/cm2)
    epsco = 0.002          # eps'co: deformacion en el pico no confinado

    # Geometria de la seccion 40x40 cm
    b = 40.0               # base de la seccion (cm)
    h = 40.0               # altura de la seccion (cm)
    d_centro_estribo = 5.0 # distancia del borde al CENTRO del estribo (cm)
    bc = b - 2 * d_centro_estribo   # nucleo en x (cm)
    dc = h - 2 * d_centro_estribo   # nucleo en y (cm)

    # Estribos cerrados de 3/8" (diametro 0.95 cm), 2 ramas por direccion
    db_estribo = 0.95      # diametro del estribo (cm)
    s = 10.0               # espaciamiento de estribos (cm)
    fyh = 4200.0           # fluencia del estribo (kg/cm2)
    eps_su = 0.12          # deformacion ultima del acero del estribo
    n_ramas = 2            # ramas del estribo en cada direccion
    area_rama = np.pi * db_estribo**2 / 4.0   # area de UNA rama (cm2)
    Asx = n_ramas * area_rama   # area total de ramas en x (cm2)
    Asy = n_ramas * area_rama   # area total de ramas en y (cm2)

    # Refuerzo longitudinal: 4 barras de 3/4" (diametro 1.91 cm) en esquinas
    db_long = 1.91         # diametro de la barra longitudinal (cm)
    n_barras = 4
    area_barra = np.pi * db_long**2 / 4.0
    As_long_total = n_barras * area_barra

    print("\nPASO 1 - DATOS DE LA COLUMNA DEL EJEMPLO")
    print(f"  Seccion: {b:.0f} x {h:.0f} cm, nucleo {bc:.1f} x {dc:.1f} cm")
    print(f"  f'co = {fco:.0f} kg/cm2,  eps'co = {epsco:.3f}")
    print(f"  Estribos: {db_estribo:.2f} cm de diametro @ s = {s:.0f} cm,",
          f" 2 ramas por direccion, fyh = {fyh:.0f} kg/cm2")
    print(f"  Refuerzo longitudinal: {n_barras} barras de {db_long:.2f} cm,",
          f"Asl = {As_long_total:.2f} cm2")

    # ------------------------------------------------------------------
    # PASO 2. MODULO DE ELASTICIDAD DEL CONCRETO
    # ------------------------------------------------------------------
    Ec = modulo_elasticidad(fco)

    print("\nPASO 2 - MODULO DE ELASTICIDAD")
    print(f"  Ec = 15000 * sqrt({fco:.0f}) = {Ec:.0f} kg/cm2")

    # ------------------------------------------------------------------
    # PASO 3. PRESION LATERAL NOMINAL DE LOS ESTRIBOS
    # ------------------------------------------------------------------
    flx, fly = presion_lateral(Asx, Asy, s, bc, dc, fyh)

    print("\nPASO 3 - PRESION LATERAL NOMINAL DE LOS ESTRIBOS")
    print(f"  f'lx = {Asx:.2f}*{fyh:.0f}/({s:.0f}*{dc:.0f}) = {flx:.2f} kg/cm2")
    print(f"  f'ly = {Asy:.2f}*{fyh:.0f}/({s:.0f}*{bc:.0f}) = {fly:.2f} kg/cm2")

    # ------------------------------------------------------------------
    # PASO 4. FACTOR DE EFECTIVIDAD ke Y PRESION EFECTIVA f'l
    # ------------------------------------------------------------------
    # Distancias libres entre barras longitudinales: con 4 barras en
    # esquinas, en cada direccion hay una sola distancia w' entre los
    # Distancias libres cara a cara entre barras longitudinales:
    # Con 4 barras en las esquinas, hay 4 arcos de descarga (2 en x, 2 en y).
    recubrimiento_libre = 4.0          # recubrimiento libre al estribo (cm)
    
    # Posicion del centro de la barra desde el BORDE exterior de la seccion:
    pos_centro_barra = recubrimiento_libre + db_estribo + db_long / 2.0
    
    # Distancia entre centros de barras extremas
    w_centros_x = b - 2.0 * pos_centro_barra
    w_centros_y = h - 2.0 * pos_centro_barra
    
    # Distancia libre (cara a cara), Mander 1988 ec. 8
    w1_libre = w_centros_x - db_long
    w2_libre = w_centros_y - db_long
    
    # Lista de TODAS las distancias libres perimetrales (4 caras)
    distancias_w = [w1_libre, w1_libre, w2_libre, w2_libre]

    ke = factor_efectividad_confinamiento(bc, dc, s, db_estribo,
                                          distancias_w, As_long_total)

    # Presion efectiva: para seccion rectangular con f'lx ~= f'ly se usa
    # el promedio de ambas direcciones, multiplicado por ke.
    fl_promedio = (flx + fly) / 2.0
    fl_efectiva = ke * fl_promedio

    print("\nPASO 4 - FACTOR DE EFECTIVIDAD DEL CONFINAMIENTO")
    print(f"  Distancias libres entre barras: w1 = {w1_libre:.2f} cm, w2 = {w2_libre:.2f} cm")
    print(f"  ke = {ke:.3f}")
    print(f"  f'l (efectiva) = ke * f'l_promedio = {fl_efectiva:.2f} kg/cm2")

    # ------------------------------------------------------------------
    # PASO 5. RESISTENCIA DEL CONCRETO CONFINADO
    # ------------------------------------------------------------------
    fcc, epscc = resistencia_confinada(fco, fl_efectiva, epsco)

    print("\nPASO 5 - RESISTENCIA DEL CONCRETO CONFINADO (ecs. 4 y 5)")
    print(f"  f'cc = {fcc:.1f} kg/cm2   (aumento vs f'co: {100*(fcc/fco-1):.1f}%)")
    print(f"  eps'cc = {epscc:.4f}")

    # ------------------------------------------------------------------
    # PASO 6. DEFORMACION ULTIMA DEL CONFINADO
    # ------------------------------------------------------------------
    rho_s = relacion_volumetrica(Asx, Asy, s, bc, dc)
    eps_cu = deformacion_ultima(fcc, rho_s, fyh, eps_su)

    print("\nPASO 6 - DEFORMACION ULTIMA (falla del estribo)")
    print(f"  rho_s = {rho_s:.4f}")
    print(f"  eps_cu = 0.004 + 1.4*{rho_s:.4f}*{fyh:.0f}*{eps_su:.2f}/{fcc:.1f}")
    print(f"  eps_cu = {eps_cu:.4f}")

    # ------------------------------------------------------------------
    # PASO 7. CONSTRUIMOS LAS DOS CURVAS COMPLETAS
    # ------------------------------------------------------------------
    # Vector de deformaciones de 0 a un valor que cubra ambas curvas
    eps = np.linspace(0.0, 0.030, 500)

    sigma_no_conf = curva_mander_no_confinado(eps, fco, epsco)
    sigma_conf = curva_mander_confinado(eps, fco, fl_efectiva,
                                        epsco, rho_s, fyh, eps_su, Ec)

    # ------------------------------------------------------------------
    # PASO 8. VERIFICACIONES DE ORDEN DE MAGNITUD
    #         (comprobamos que la implementacion se comporta como debe)
    # ------------------------------------------------------------------
    print("\nPASO 8 - VERIFICACIONES")

    # 1) En epsilon = 0 el esfuerzo debe ser 0
    s0 = curva_popovics(np.array([0.0]), fco, epsco, Ec)[0]
    print(f"  1. Esfuerzo en eps = 0: {s0:.2f} kg/cm2 (debe ser 0.00)")

    # 2) En el pico no confinado (eps'co) debe dar f'co
    spico_no = curva_popovics(np.array([epsco]), fco, epsco, Ec)[0]
    print(f"  2. Esfuerzo en eps'co = {epsco:.3f}: {spico_no:.2f} kg/cm2",
          f"(debe ser {fco:.0f})")

    # 3) En el pico confinado (eps'cc) debe dar f'cc
    spico_conf = curva_popovics(np.array([epscc]), fcc, epscc, Ec)[0]
    print(f"  3. Esfuerzo en eps'cc = {epscc:.4f}: {spico_conf:.2f} kg/cm2",
          f"(debe ser {fcc:.1f})")

    # 4) El confinado debe ser mas resistente y mas deformable
    print(f"  4. f'cc > f'co ?  {fcc:.1f} > {fco:.0f} -> {fcc > fco}")
    print(f"  5. eps'cc > eps'co ? {epscc:.4f} > {epsco:.3f} -> {epscc > epsco}")

    # ------------------------------------------------------------------
    # PASO 9. GRAFICO COMPARATIVO
    # ------------------------------------------------------------------
    plt.figure(figsize=(9, 6))

    # Curvas completas
    plt.plot(eps, sigma_no_conf, "b-", linewidth=2,
             label="No confinado (recubrimiento)")
    plt.plot(eps, sigma_conf, "r-", linewidth=2,
             label=f"Confinado (f'cc = {fcc:.0f} kg/cm2)")

    # Puntos notables de la curva no confinada
    plt.plot([epsco], [fco], "bo", markersize=8, zorder=5)
    plt.text(epsco + 0.0008, fco, f"({epsco:.3f}, {fco:.0f})", color="b")
    
    # Inicio de recta descendente (2*eps'co) y spalling
    eps_sp_ejemplo = 0.005
    f_2epsco = curva_popovics(np.array([2.0 * epsco]), fco, epsco, Ec)[0]
    plt.plot([2.0 * epsco], [f_2epsco], "b^", markersize=8, zorder=5)
    plt.text(2.0 * epsco + 0.0008, f_2epsco + 5, f"2 eps'co ({f_2epsco:.0f})", color="b")
    plt.plot([eps_sp_ejemplo], [0.0], "bx", markersize=10, zorder=5)
    plt.text(eps_sp_ejemplo + 0.0008, 10, "spalling", color="b")

    # Puntos notables de la curva confinada
    plt.plot([epscc], [fcc], "ro", markersize=8, zorder=5)
    plt.text(epscc + 0.0008, fcc + 8, f"({epscc:.3f}, {fcc:.0f})", color="r")
    plt.plot([eps_cu], [0.0], "rx", markersize=10, zorder=5)
    plt.text(eps_cu - 0.0012, -12, f"eps_cu = {eps_cu:.3f}", color="r")

    # Referencias graficas
    plt.xlabel("Deformacion unitaria eps (cm/cm)")
    plt.ylabel("Esfuerzo de compresion fc (kg/cm2)")
    plt.title("Modelo de Mander (1988): concreto confinado y no confinado\n"
              f"Columna {b:.0f}x{h:.0f} cm, f'co = {fco:.0f} kg/cm2, "
              f"estribos @ {s:.0f} cm")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")

    # Limites del grafico para que se vean bien ambas curvas
    plt.xlim(0.0, 0.030)
    plt.ylim(-20, fcc * 1.15)

    # Con esta linea se muestra la ventana con el grafico.
    # (En Jupyter usa: %matplotlib inline  antes de correr el script)
    plt.show()

    # Opcional: guardar el grafico como imagen PNG (descomenta la linea)
    # plt.savefig("curva_mander.png", dpi=150)

    print("\nFIN DEL EJEMPLO")
    print("  Interpretacion: el confinamiento eleva la resistencia")
    print("  (f'cc > f'co) y sobre todo aumenta MUCHO la deformacion")
    print("  ultima (eps_cu >> 0.004), que es la clave de la ductilidad")
    print("  de las columnas sismorresistentes.")
