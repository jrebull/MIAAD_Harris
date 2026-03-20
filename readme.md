# Visa Predict AI — MOHHO

Optimización Multiobjetivo de la Asignación de *Green Cards* en Estados Unidos mediante Harris Hawks Optimization (HHO).

**Proyecto Final** — Optimización Inteligente (MIAAD, UACJ)

## Equipo
- Yazmín Ivonne Flores Martínez (261548)
- Javier Augusto Rebull Saucedo (263483)

## Estructura

```
├── src/                  # Código fuente
│   ├── config.py         # Parámetros del problema y algoritmo
│   ├── data.py           # Datos reales de visas (10 países × 5 categorías)
│   ├── problem.py        # Funciones objetivo f1, f2
│   ├── decoder.py        # SPV + Decodificador Greedy
│   ├── hho.py            # Harris Hawks Optimization (6 operadores)
│   ├── mohho.py          # Multi-Objective HHO (archivo Pareto)
│   ├── experiment.py     # Runner de experimentos (30 corridas)
│   └── baseline.py       # Simulación FIFO del sistema actual
├── app/
│   └── streamlit_app.py  # Dashboard interactivo
├── tests/                # Suite de pruebas (pytest)
├── results/              # Resultados generados
│   └── figures/          # Gráficas PNG
└── references/
    ├── Fase01_Latex/     # Reporte Fase 01
    ├── Fase02_Latex/     # Modelo matemático (Fase 02)
    └── Fase03_Latex/     # Reporte de implementación (Fase 03)
```

## Ejecución

```bash
# Instalar dependencias
pip install -r requirements.txt

# Correr tests
python -m pytest tests/ -v

# Ejecutar experimentos (30 corridas × 500 iteraciones)
python -m src.experiment

# Lanzar dashboard
streamlit run app/streamlit_app.py
```
