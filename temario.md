ANÁLISIS PUSHOVER DE EDIFICIOS DE PÓRTICOS DE CONCRETO ARMADO CON PYTHON

* Duración Total: 16 horas académicas (08 sesiones de 2 horas cada una)

DESCRIPCIÓN DEL CURSO
El curso está diseñado para introducir a los ingenieros estructurales en el Análisis Estático No Lineal (Pushover) aplicando la filosofía de Diseño Sismorresistente Basado en Desempeño. A lo largo del programa, se combinarán los fundamentos teóricos de los niveles de amenaza y estados límite con la potencia del lenguaje de programación Python. Los participantes aprenderán a modelar la no linealidad del concreto armado, automatizar algoritmos de control de desplazamiento, trazar curvas de capacidad y determinar el punto de desempeño estructural según normativas internacionales (ASCE 41, ATC-40) y criterios locales, optimizando el procesamiento de datos y la generación de reportes técnicos.

CONTENIDO PROGRAMÁTICO POR SESIONES

Sesión 1. Introducción al Análisis No Lineal y Filosofía de Desempeño

* Fundamentos del Diseño Basado en Desempeño (DDBD) frente al diseño convencional por fuerzas
* Conceptos clave: linealidad vs. no linealidad geométrica y de materiales
* Objetivos de desempeño y niveles de daño estructural (IO, LS, CP) según el ASCE 41
* Introducción al ecosistema de Python para ingeniería estructural (NumPy, SciPy, Matplotlib)

Sesión 2. Modelamiento Constitutivo del Concreto y Acero en Python

* Comportamiento uniaxial de materiales: concreto confinado y no confinado (Modelo de Mander)
* Comportamiento del acero de refuerzo (Modelo elastoplástico perfecto y con endurecimiento por deformación)
* Programación e implementación de curvas esfuerzo-deformación con NumPy
* Graficación y validación de diagramas constitutivos con Matplotlib

Sesión 3. Análisis Momento-Curvatura y Rótulas Plásticas

* Mecánica de la sección transversal de concreto armado sujeta a flexocompresión
* Teoría y cálculo del diagrama Momento-Curvatura mediante discretización en fibras
* Definición de rótulas plásticas concentradas frente a plasticidad distribuida
* Automatización de la obtención de la longitud de rótula plástica y propiedades plásticas en vigas y columnas

Sesión 4. Matriz de Rigidez No Lineal y Algoritmos de Solución en Python

* Formulación de la matriz de rigidez local y global considerando la degradación de rigidez
* Estrategias de solución para sistemas de ecuaciones no lineales: métodos iterativos
* Implementación en Python del algoritmo de Newton-Raphson tradicional
* Programación de esquemas de control por carga y control por desplazamiento (Arc-Length simplificado)

Sesión 5. Modelado Estructural del Pórtico de Concreto Armado

* Configuración geométrica y propiedades mecánicas de un pórtico bidimensional (2D)
* Definición de nodos, elementos tipo barra (Beam-Column elements) y condiciones de frontera
* Ensamblaje automatizado de la matriz de rigidez global del pórtico en Python
* Incorporación de las relaciones constitutivas no lineales en los extremos de los elementos

Sesión 6. Ejecución del Análisis Pushover Monotónico Incremental

* Criterios para la selección del patrón de cargas laterales (uniforme, modal y proporcional a la masa)
* Rutina en Python para la aplicación incremental del vector de fuerzas / desplazamientos
* Monitoreo del desplazamiento del nodo tonto (techo) vs. el cortante basal
* Extracción y graficación automatizada de la Curva de Capacidad del edificio

Sesión 7. Determinación del Punto de Desempeño Espectral

* Conceptos del Método del Espectro de Capacidad (ATC-40) y Método de Coeficientes de Desplazamiento (FEMA 356 / ASCE 41)
* Conversión de la curva de capacidad a un formato de Espectro de Aceleración-Desplazamiento (ADRS)
* Intersección de la demanda sísmica (espectro de diseño local) con la capacidad estructural usando SciPy
* Programación del cálculo del amortiguamiento viscoso equivalente y la ductilidad de la estructura

Sesión 8. Evaluación de Resultados y Automatización del Informe Técnico

* Verificación del estado de las rótulas plásticas en el punto de desempeño frente a los límites normativos
* Identificación de los mecanismos de colapso (viga débil - columna fuerte)
* Generación automática de la Memoria de Cálculo y diagramas de daño estructural con Python (DataFrames con Pandas)
* Clausura del curso y discusión de aplicaciones avanzadas (OpenSeesPy)
