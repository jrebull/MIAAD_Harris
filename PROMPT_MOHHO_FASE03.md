# SuperPrompt: MOHHO — Green Card Optimization (Fase 03)

## META
- **Proyecto:** Visa Predict AI — Optimización Multiobjetivo de Green Cards con Harris Hawks Optimization
- **Materia:** Optimización Inteligente (MIAAD, UACJ) — Mtro. Raúl Gibrán Porras Alaniz
- **Equipo:** Javier Rebull (263483) + Yazmín Flores (261548)
- **Repo:** `https://github.com/jrebull/MIAAD_Harris`
- **Local:** `/Users/haowei/Documents/MIAAD/SMART/Harris`
- **Git user:** `jrebull` — commit messages en español
- **Ejecución:** `claude --dangerously-skip-permissions`

---

## 1. SETUP INICIAL

El repo YA está clonado. Solo hacer pull y crear las carpetas que falten:

```bash
cd /Users/haowei/Documents/MIAAD/SMART/Harris
git pull origin main

# Crear estructura de carpetas (las que ya existen no se tocan)
mkdir -p references/Fase03_Latex/Figures
mkdir -p src
mkdir -p app
mkdir -p results/figures
mkdir -p tests
```

### Ubicación de archivos LaTeX:

```
references/
├── Fase01_Latex/          ← LaTeX de la Fase 01 (YA EXISTE, solo lectura)
├── Fase02_Latex/          ← LaTeX de la Fase 02 (LEER + ACTUALIZAR con los 6 gaps)
│   └── HarrisFase02.tex   ← Modelo matemático — FUENTE DE VERDAD
└── Fase03_Latex/          ← GENERAR aquí el reporte de Fase 03
    └── Fase03Reporte.tex   ← Reporte técnico de resultados
```

**ANTES DE CUALQUIER CÓDIGO:** Leer `references/Fase02_Latex/HarrisFase02.tex` completo. Es la especificación matemática que el código debe implementar fielmente. Si durante la implementación se descubre un error en el LaTeX, actualizar `HarrisFase02.tex` y documentar el cambio.

### Estructura objetivo del proyecto:

```
MIAAD_Harris/
├── README.md
├── requirements.txt
├── .gitignore
├── references/
│   ├── Fase01_Latex/                 ← YA EXISTE (solo lectura)
│   ├── Fase02_Latex/
│   │   └── HarrisFase02.tex         ← LEER + ACTUALIZAR (modelo matemático)
│   └── Fase03_Latex/
│       ├── Fase03Reporte.tex         ← GENERAR (reporte técnico)
│       └── Figures/                  ← Figuras generadas para el reporte
├── src/
│   ├── __init__.py
│   ├── config.py                 ← Parámetros del problema y del algoritmo
│   ├── data.py                   ← Datos reales de visas (10 países × 5 categorías)
│   ├── problem.py                ← Definición del problema: f1, f2, restricciones, spillover
│   ├── decoder.py                ← SPV + Decodificador Greedy
│   ├── hho.py                    ← Harris Hawks Optimization (6 operadores)
│   ├── mohho.py                  ← Multi-Objective HHO (archivo Pareto, crowding, líder)
│   ├── experiment.py             ← Runner de experimentos con múltiples corridas
│   └── baseline.py               ← Simulación del sistema FIFO actual como comparación
├── app/
│   └── streamlit_app.py          ← Dashboard interactivo
├── results/                      ← Generado por experiment.py
│   ├── pareto_front.csv
│   ├── convergence.csv
│   ├── run_XX.json
│   └── figures/
│       ├── pareto_front.png
│       ├── convergence.png
│       ├── assignment_heatmap.png
│       └── comparison_baseline.png
└── tests/
    ├── test_decoder.py
    ├── test_fitness.py
    ├── test_hho.py
    ├── test_pareto.py
    └── test_integration.py
```

---

## 2. ESPECIFICACIÓN MATEMÁTICA EXACTA

Lee `references/Fase02_Latex/HarrisFase02.tex` como fuente de verdad. A continuación el resumen ejecutivo que el código DEBE implementar fielmente:

### 2.1 Conjuntos

| Símbolo | Nombre | Definición |
|---------|--------|-----------|
| C | Países | {1,...,10}: India, China, Filipinas, México, Corea del Sur, Brasil, Canadá, Reino Unido, Nigeria, Resto del Mundo |
| J | Categorías | {1,...,5}: EB-1, EB-2, EB-3, EB-4, EB-5 |
| G | Grupos | {1,...,50}: cada combinación (país, categoría) = C × J |

### 2.2 Parámetros

| Símbolo | Valor | Descripción |
|---------|-------|-------------|
| V | 140,000 | Visas EB anuales disponibles |
| V_total | 366,000 | Visas totales (familiares + empleo), base del 7% |
| P_c | 25,620 | Tope por país = 0.07 × V_total |
| P_RdM | V − Σ_{c≠RdM} min(n_c, P_c) | Tope especial para Resto del Mundo |
| K_j | [40040, 40040, 40040, 9940, 9940] | Topes base por categoría |
| t_actual | 2026 | Año fiscal actual |

### 2.3 Datos de los 10 países (en `data.py`)

```python
# Estructura: {pais: {categoria: {"n": demanda, "d": fecha_prioridad}}}
# IMPORTANTE: usar datos del Visa Bulletin Feb 2026 y estimaciones USCIS
# w_g = t_actual - d_g (tiempo de espera en años)

VISA_DATA = {
    "India":         {"EB-1": {"n": 50000, "d": 2022}, "EB-2": {"n": 800000, "d": 2013}, "EB-3": {"n": 200000, "d": 2012}, "EB-4": {"n": 2000, "d": 2025}, "EB-5": {"n": 5000, "d": 2020}},
    "China":         {"EB-1": {"n": 15000, "d": 2023}, "EB-2": {"n": 120000, "d": 2019}, "EB-3": {"n": 30000, "d": 2020}, "EB-4": {"n": 1000, "d": 2025}, "EB-5": {"n": 20000, "d": 2018}},
    "Filipinas":     {"EB-1": {"n": 3000, "d": 2025}, "EB-2": {"n": 25000, "d": 2022}, "EB-3": {"n": 40000, "d": 2018}, "EB-4": {"n": 500, "d": 2025}, "EB-5": {"n": 200, "d": 2025}},
    "Mexico":        {"EB-1": {"n": 2000, "d": 2025}, "EB-2": {"n": 15000, "d": 2025}, "EB-3": {"n": 25000, "d": 2021}, "EB-4": {"n": 3000, "d": 2025}, "EB-5": {"n": 500, "d": 2025}},
    "Corea del Sur": {"EB-1": {"n": 2000, "d": 2025}, "EB-2": {"n": 12000, "d": 2025}, "EB-3": {"n": 8000, "d": 2024}, "EB-4": {"n": 300, "d": 2025}, "EB-5": {"n": 1000, "d": 2025}},
    "Brasil":        {"EB-1": {"n": 1500, "d": 2025}, "EB-2": {"n": 8000, "d": 2025}, "EB-3": {"n": 5000, "d": 2025}, "EB-4": {"n": 200, "d": 2025}, "EB-5": {"n": 300, "d": 2025}},
    "Canada":        {"EB-1": {"n": 1000, "d": 2025}, "EB-2": {"n": 6000, "d": 2025}, "EB-3": {"n": 4000, "d": 2025}, "EB-4": {"n": 150, "d": 2025}, "EB-5": {"n": 200, "d": 2025}},
    "Reino Unido":   {"EB-1": {"n": 800, "d": 2025}, "EB-2": {"n": 5000, "d": 2025}, "EB-3": {"n": 3000, "d": 2025}, "EB-4": {"n": 100, "d": 2025}, "EB-5": {"n": 150, "d": 2025}},
    "Nigeria":       {"EB-1": {"n": 500, "d": 2025}, "EB-2": {"n": 4000, "d": 2025}, "EB-3": {"n": 3000, "d": 2025}, "EB-4": {"n": 200, "d": 2025}, "EB-5": {"n": 100, "d": 2025}},
    "Resto del Mundo":{"EB-1": {"n": 20000, "d": 2025}, "EB-2": {"n": 300000, "d": 2025}, "EB-3": {"n": 100000, "d": 2025}, "EB-4": {"n": 5000, "d": 2025}, "EB-5": {"n": 3000, "d": 2025}},
}
```

**NOTA CRÍTICA:** Estos son valores estimados. Si al correr el decodificador, la demanda total por categoría es menor que K_j para alguna j, RECALCULAR porque eso activaría spillover real y potencialmente revive f3. Si eso ocurre, ACTUALIZAR `references/Fase02_Latex/HarrisFase02.tex` para reflejar que f3 sí varía.

### 2.4 Spillover (calcular UNA VEZ al inicio)

```
D_j = Σ_{g: j(g)=j} n_g   (demanda total de categoría j)
K4_eff = K4,  K5_eff = K5
S4 = max(0, K4 - D4),  S5 = max(0, K5 - D5)
K1_eff = K1 + S4 + S5
S1 = max(0, K1_eff - D1)
K2_eff = K2 + S1
S2 = max(0, K2_eff - D2)
K3_eff = K3 + S2
```

### 2.5 Funciones Objetivo (BIOBJETIVO)

**f1 — Carga de espera no atendida (MINIMIZAR):**

```
f1(x) = Σ_g (n_g - x_g) * w_g  /  Σ_g n_g
```

donde `w_g = t_actual - d_g`. Minimizar f1 → priorizar grupos con alto w_g (larga espera).

**VERIFICACIÓN OBLIGATORIA:** Después de implementar, correr este test:
- Permutación que atiende India primero → f1 debe ser MENOR que permutación que ignora India
- Si f1 sale al revés, LA IMPLEMENTACIÓN ESTÁ MAL

**f2 — Máxima disparidad entre países (MINIMIZAR):**

```
W_c(x) = Σ_{g: c(g)=c} x_g * w_g  /  Σ_{g: c(g)=c} x_g    (si Σ x_g > 0)
        = w_c_max                                               (si Σ x_g = 0)

f2(x) = max_{c1, c2} |W_c1(x) - W_c2(x)|
```

### 2.6 Restricciones (manejadas por el decodificador, NO por penalización)

| ID | Expresión | Descripción |
|----|-----------|-------------|
| R1 | Σ x_g ≤ V | Tope anual 140,000 |
| R2 | Σ_{g: c(g)=c} x_g ≤ P_c ∀c | Tope 7% por país (P_RdM especial para Resto del Mundo) |
| R3 | Σ_{g: j(g)=j} x_g ≤ K_j_eff ∀j | Tope por categoría con spillover |
| R4 | 0 ≤ x_g ≤ n_g ∀g | No exceder demanda |
| R5 | x_g ∈ Z+ | Enteros no negativos |

### 2.7 Codificación SPV + Decodificador Greedy

**Capa 1:** Halcón H ∈ R^G (vector continuo, G=50)
**Capa 2:** π = argsort(H) → permutación (orden de prioridad)
**Capa 3:** Decodificador greedy → x ∈ Z^G+ (asignación factible)

```python
def decode(pi, V, P, K_eff, n):
    """
    Decodificador greedy: TODA permutación produce asignación factible.
    
    Args:
        pi: lista de índices de grupo en orden de prioridad
        V: total visas disponibles
        P: dict {país: tope_país}
        K_eff: dict {categoría: tope_efectivo}
        n: dict {grupo: demanda}
    Returns:
        x: dict {grupo: visas_asignadas}
    """
    x = {g: 0 for g in pi}
    V_rest = V
    uso_pais = defaultdict(int)
    uso_cat = defaultdict(int)
    
    for g in pi:
        cap_pais = P[country(g)] - uso_pais[country(g)]
        cap_cat = K_eff[category(g)] - uso_cat[category(g)]
        x_g = min(n[g], V_rest, cap_pais, cap_cat)
        x[g] = x_g
        V_rest -= x_g
        uso_pais[country(g)] += x_g
        uso_cat[category(g)] += x_g
    
    return x
```

### 2.8 HHO — 6 Operadores (de Heidari et al. 2019)

Implementar EXACTAMENTE como se describe en `references/Fase01_Latex/` (Fase 1). Leer ese LaTeX para los detalles completos de cada operador con sus ecuaciones. Resumen:

```
E = 2 * E0 * (1 - t/T)     # Energía de escape, E0 ∈ [-1,1] aleatorio
J = 2 * (1 - r5)            # Fuerza de salto

|E| ≥ 1 → EXPLORACIÓN:
  q ≥ 0.5 → Op1: X = X_rand - r1 * |X_rand - 2*r2*X_i|
  q < 0.5 → Op2: X = (X_rabbit - X_mean) - r3*(LB + r4*(UB-LB))

|E| < 1 → EXPLOTACIÓN:
  r ≥ 0.5, |E| ≥ 0.5 → Op3: Asedio suave
  r ≥ 0.5, |E| < 0.5 → Op4: Asedio duro
  r < 0.5, |E| ≥ 0.5 → Op5: Asedio suave + Lévy
  r < 0.5, |E| < 0.5 → Op6: Asedio duro + Lévy

Lévy flight: LF(x) = 0.01 * u*σ / |v|^(1/β), β=1.5
σ = [Γ(1+β)*sin(πβ/2) / (Γ((1+β)/2)*β*2^((β-1)/2))]^(1/β)

Operadores 5-6 usan selección greedy: probar Y, si no mejora probar Z, si no conservar.
```

**IMPORTANTE:** Los operadores mueven vectores en R^G. La conversión a asignación se hace SIEMPRE via SPV+decoder DESPUÉS del movimiento. Los operadores NUNCA tocan x_g directamente.

### 2.9 MOHHO — Componentes Multiobjetivo

1. **Archivo externo de Pareto** (tamaño máximo configurable, e.g. 100)
2. **Dominancia:** a ≻ b ⟺ ∀m∈{1,2}: f_m(a) ≤ f_m(b) ∧ ∃m: f_m(a) < f_m(b)
3. **Selección de líder** por distancia de crowding (ruleta ponderada)
4. **Actualización del archivo** después de cada evaluación

---

## 3. PRINCIPIOS DE CÓDIGO

### 3.1 Clean Code
- Funciones < 30 líneas, nombres descriptivos en inglés
- Type hints en TODAS las funciones
- Docstrings (Google style) en TODAS las funciones y clases
- Constantes en MAYÚSCULAS en `config.py`
- Sin magic numbers — todo referenciable a la Sección del LaTeX

### 3.2 SOLID
- **S:** Cada módulo tiene UNA responsabilidad (decoder.py solo decodifica, hho.py solo operadores, etc.)
- **O:** Problem class extensible para cambiar datos sin modificar MOHHO
- **L:** Si se cambia HHO por otro algoritmo, el decoder y fitness no cambian
- **I:** Interfaces pequeñas: `evaluate(x) → (f1, f2)`, `decode(pi) → x`, `step(population) → population`
- **D:** MOHHO depende de abstracciones (Problem protocol), no de datos concretos

### 3.3 MLOps
- Todas las corridas guardadas en `results/` con timestamp y semilla
- Configuración reproducible vía `config.py` (semillas, parámetros)
- Logging estructurado (JSON) de cada generación
- Métricas de convergencia: hypervolume, spread, spacing del frente de Pareto
- Requirements.txt con versiones pinned

### 3.4 Testing
- `test_decoder.py`: verificar que TODA permutación genera asignación factible (R1-R5). Test con ejemplo del LaTeX (G=6, V=100).
- `test_fitness.py`: verificar f1, f2 contra valores calculados a mano del LaTeX (f1=2.17, f2=13.47). Verificar dirección de f1 (servir India → f1 baja).
- `test_hho.py`: verificar que cada operador produce vector en R^G. Verificar transiciones de energía E.
- `test_pareto.py`: verificar dominancia con casos conocidos. Verificar que archivo no contiene soluciones dominadas.
- `test_integration.py`: correr MOHHO 10 generaciones, verificar que el frente de Pareto es no vacío y todas las soluciones son factibles.

---

## 4. EXPERIMENTOS

### 4.1 Configuración experimental

```python
# config.py
POPULATION_SIZE = 50        # N halcones
MAX_ITERATIONS = 500        # T iteraciones
NUM_RUNS = 30               # Corridas independientes
ARCHIVE_SIZE = 100          # Máximo soluciones en archivo Pareto
SEED_BASE = 42              # Semilla base (run i usa SEED_BASE + i)
LB = 0.0                    # Límite inferior del espacio continuo
UB = 1.0                    # Límite superior
BETA_LEVY = 1.5             # Parámetro de Lévy
```

### 4.2 Métricas a reportar

1. **Hypervolume (HV):** Volumen dominado por el frente respecto a punto de referencia. Calcular con `pymoo` o implementar.
2. **Spread (Δ):** Distribución de soluciones en el frente.
3. **Número de soluciones no dominadas** por corrida.
4. **Comparación vs baseline FIFO:** Calcular f1 y f2 del sistema actual y graficar.

### 4.3 Baseline FIFO (`baseline.py`)

Simular el sistema actual: procesar grupos en orden FIFO estricto (fecha de prioridad más antigua primero, dentro de cada categoría), respetando todos los topes. Esto da UN punto (f1_fifo, f2_fifo) contra el cual comparar todo el frente de Pareto.

---

## 5. STREAMLIT APP (`app/streamlit_app.py`)

### Funcionalidades requeridas:

1. **Frente de Pareto interactivo:** Scatter plot f1 vs f2 con punto FIFO marcado. Click en un punto muestra la tabla de asignación.
2. **Heatmap de asignación:** Matriz país × categoría para la solución seleccionada.
3. **Comparación con baseline:** Barras mostrando Δf1 y Δf2 vs FIFO.
4. **Convergencia:** Línea de hypervolume por iteración.
5. **Selector de corrida:** Dropdown para ver resultados de cada corrida independiente.
6. **Paleta UACJ:** Azul #003CA6, Amarillo #FFD600, Gris #555559.

---

## 6. REPORTE LaTeX (`references/Fase03_Latex/Fase03Reporte.tex`)

Usar EXACTAMENTE el mismo preamble/estilo que `references/Fase02_Latex/HarrisFase02.tex` (Palatino, tcolorbox, colores UACJ, APA 7).

### Estructura del reporte:

1. **Portada** (mismo formato que Fase 02, título "Proyecto Final: Fase 03 — Implementación y Resultados")
2. **Tabla de contenidos**
3. **Sección 1: Resumen de correcciones al modelo** — Documentar los 6 gaps del Fase 02 que se corrigieron:
   - G1: Formulación compacta min{f1,f2} s.t. R1-R6
   - G2: Prueba formal de conflicto f1 vs f2 con permutaciones concretas que producen soluciones no dominadas mutuamente
   - G3: Definición formal del frente de Pareto: P* = {x ∈ F | ¬∃x' ∈ F : x' ≻ x}
   - G4: w_g en tabla de parámetros
   - G5: Cita de SPV (Bean, 1994)
   - G6: Precisión en texto del decodificador
4. **Sección 2: Implementación** — Arquitectura del código, módulos, decisiones de diseño
5. **Sección 3: Configuración experimental** — Parámetros, número de corridas, métricas
6. **Sección 4: Resultados** — Frente de Pareto, convergencia, estadísticas sobre 30 corridas
7. **Sección 5: Comparación con sistema actual** — FIFO vs MOHHO
8. **Sección 6: Análisis y discusión** — Trade-offs, implicaciones de política pública
9. **Sección 7: Conclusiones**
10. **Referencias** — Incluir Bean (1994) para SPV, Heidari et al. (2019), etc.

### Figuras requeridas en el reporte:
- Frente de Pareto (scatter, con punto FIFO)
- Convergencia del hypervolume
- Heatmap de asignación de la solución "equilibrio"
- Boxplot de HV sobre 30 corridas
- Tabla comparativa: mejor f1, mejor f2, equilibrio vs FIFO

**IMPORTANTE:** Todos los valores numéricos en el reporte deben ser EXTRAÍDOS de los archivos CSV/JSON generados por el código. NUNCA inventar números. Si un número aparece en el LaTeX, debe haber un archivo en `results/` que lo respalde.

---

## 7. CICLO DE AUDITORÍA (OBLIGATORIO)

Después de implementar, ejecutar este ciclo hasta que pase TODOS los checks:

### Check 1: Aritmética del decoder
```python
# Correr el ejemplo del LaTeX (G=6, V=100, Pc=50, Kj=50)
# H = [0.72, 0.15, 0.91, 0.33, 0.88, 0.47]
# Resultado esperado: x = {1:20, 2:30, 3:20, 4:15, 5:10, 6:5}
# f1 = 2.17, f2 = 13.47
```

### Check 2: Dirección de f1
```python
# Permutación India-first: f1 debe ser < permutación India-last
# Si no: BUG en f1
```

### Check 3: Factibilidad universal
```python
# Generar 10,000 permutaciones aleatorias
# Para CADA una: decode → x → verificar R1-R5
# Tasa de infactibilidad esperada: 0.0%
```

### Check 4: Pareto no-dominancia
```python
# Para cada par (a, b) en el archivo de Pareto:
# Verificar que a no domina a b Y b no domina a a
```

### Check 5: Coherencia LaTeX ↔ Código
```python
# Extraer cada ecuación del references/Fase02_Latex/HarrisFase02.tex
# Verificar que la implementación en Python es IDÉNTICA
# Si el código revela un error en el LaTeX → ACTUALIZAR el LaTeX
# Si el LaTeX revela un error en el código → ACTUALIZAR el código
```

### Check 6: Reproducibilidad
```python
# Correr MOHHO con semilla 42, guardar frente de Pareto
# Correr de nuevo con semilla 42, comparar
# Deben ser IDÉNTICOS
```

---

## 8. FLUJO DE EJECUCIÓN

1. **Primero:** Leer `references/Fase02_Latex/HarrisFase02.tex` completo (modelo matemático) y `references/Fase01_Latex/` (operadores HHO) para absorber todo el contexto
2. **Segundo:** Implementar `config.py`, `data.py`, `problem.py`, `decoder.py` — los cimientos
3. **Tercero:** Implementar tests del decoder y fitness — correr y pasar
4. **Cuarto:** Implementar `hho.py`, `mohho.py` — los operadores y el loop principal
5. **Quinto:** Implementar tests de HHO y Pareto — correr y pasar
6. **Sexto:** Implementar `baseline.py` y `experiment.py`
7. **Séptimo:** Correr experimentos (30 corridas) y generar resultados
8. **Octavo:** Implementar Streamlit app
9. **Noveno:** Generar `references/Fase03_Latex/Fase03Reporte.tex` con figuras reales de `results/`
10. **Décimo:** Correr ciclo de auditoría (Sección 7) — iterar hasta que pase todo
11. **Undécimo:** Actualizar `references/Fase02_Latex/HarrisFase02.tex` con los 6 gaps (G1-G6) si no se han aplicado
12. **Doceavo:** Commit final con mensaje: "Fase 03: implementación MOHHO, resultados y reporte"

---

## 9. RESTRICCIONES ABSOLUTAS

- **NO usar DEAP, PyGMO ni ningún framework de optimización.** Todo from scratch.
- **NO usar matplotlib para diagramas de arquitectura.** Solo para gráficas de resultados (scatter, convergence, heatmap).
- **NO inventar datos.** Todo número en el reporte sale de `results/`.
- **NO dejar tests sin pasar.** El CI mental es: tests pasan → código válido → reporte válido.
- **Formato de números mexicano** en el LaTeX: punto decimal, coma para miles (e.g., 140.000 visas, f1 = 2,17).
- **`\graphicspath{{../results/figures/}{Figures/}}`** en el LaTeX de Fase03 para referenciar figuras generadas.
- **Todas las figuras** se generan en `results/figures/` como PNG (300 dpi) Y se copian a `references/Fase03_Latex/Figures/` para que el LaTeX compile en Overleaf sin dependencias externas.

---

## 10. DEFINICIÓN DE TERMINADO

El proyecto está terminado cuando:

- [ ] `pytest tests/ -v` pasa al 100%
- [ ] Los 6 checks de auditoría (Sección 7) pasan
- [ ] `python src/experiment.py` genera 30 corridas en `results/`
- [ ] `streamlit run app/streamlit_app.py` levanta sin errores y muestra frente de Pareto
- [ ] `references/Fase03_Latex/Fase03Reporte.tex` compila sin errores y contiene datos reales
- [ ] `references/Fase02_Latex/HarrisFase02.tex` tiene los 6 gaps corregidos
- [ ] Todos los valores numéricos del reporte tienen respaldo en `results/`
- [ ] Git log muestra commits incrementales con mensajes en español
