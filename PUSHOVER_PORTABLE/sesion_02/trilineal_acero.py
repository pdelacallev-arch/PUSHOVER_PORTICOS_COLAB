"""Curva esfuerzo-deformación del acero — ejemplo básico para principiantes.

Este programa calcula y muestra la curva esfuerzo-deformación del acero
con dos modelos sencillos:
  1. Elastoplástico perfecto
  2. Elastoplástico con endurecimiento (trilineal)

Usa Python 3 básico: variables, listas, funciones, bucles y condicionales.
No guarda ningún archivo: los resultados se muestran en la terminal.
"""


# ============================================================
# 1. DATOS DE ENTRADA
# ============================================================
# Parámetros del acero grado 60 (unidades: kg/cm²)
fy = 4200.0        # esfuerzo de fluencia
Es = 2100000.0     # módulo de elasticidad
Esh = 21000.0      # pendiente de la rama de endurecimiento
eps_sh = 0.010     # deformación al inicio del endurecimiento
eps_su = 0.10      # deformación de rotura

# ============================================================
# 2. FUNCIONES DE CÁLCULO
# ============================================================
def esfuerzo_perfecto(eps, fy, Es):
    """Modelo 1: elastoplástico perfecto.
    Devuelve el esfuerzo para una deformación eps."""
    eps_y = fy / Es
    if eps <= eps_y:
        return Es * eps
    else:
        return fy


def esfuerzo_endurecimiento(eps, fy, Es, Esh, eps_sh):
    """Modelo 2: elastoplástico con endurecimiento (trilineal).
    Devuelve el esfuerzo para una deformación eps."""
    eps_y = fy / Es
    if eps <= eps_y:
        return Es * eps
    elif eps <= eps_sh:
        return fy
    else:
        return fy + Esh * (eps - eps_sh)


# ============================================================
# 3. CÁLCULO PRINCIPAL
# ============================================================
def ejecutar(mostrar_figura=True):
    """Calcula la curva y la muestra en la terminal."""
    # Deformaciones representativas (incluyen los puntos clave:
    # fluencia 0.002, inicio de endurecimiento 0.010 y rotura 0.100).
    deformaciones = [0.000, 0.001, 0.002, 0.005, 0.010, 0.020, 0.030,
                     0.040, 0.050, 0.060, 0.070, 0.080, 0.090, 0.100]

    # Listas vacías para guardar los esfuerzos de cada modelo.
    esfuerzos_perfecto = []
    esfuerzos_endurecimiento = []

    # Recorremos cada deformación y aplicamos las funciones de cálculo.
    for eps in deformaciones:
        esfuerzos_perfecto.append(esfuerzo_perfecto(eps, fy, Es))
        esfuerzos_endurecimiento.append(
            esfuerzo_endurecimiento(eps, fy, Es, Esh, eps_sh)
        )

    # Mostramos la tabla en la terminal.
    print("Curva esfuerzo-deformación del acero")
    print("  deformación | perfecto | endurecimiento")
    for i in range(len(deformaciones)):
        print(
            f"{deformaciones[i]:.3f} | {esfuerzos_perfecto[i]:7.1f} | "
            f"{esfuerzos_endurecimiento[i]:7.1f}"
        )

    # Figura opcional en pantalla (no se guarda ningún archivo).
    if mostrar_figura:
        plt.plot(deformaciones, esfuerzos_perfecto, "--", label="Perfecto")
        plt.plot(deformaciones, esfuerzos_endurecimiento, label="Con endurecimiento")
        plt.xlabel("Deformación (ε)")
        plt.ylabel("Esfuerzo (kg/cm²)")
        plt.legend()
        plt.show()


# ============================================================
# 4. PUNTO DE ENTRADA
# ============================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ejecutar()
