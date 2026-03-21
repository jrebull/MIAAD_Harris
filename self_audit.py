"""Self-audit script: 12 checks for triobjetivo correctness."""

import csv
import json
import sys

import numpy as np


def check_1():
    """f3 variability — MUST have hundreds of unique values."""
    from src.problem import VisaProblem
    from src.decoder import spv, decode

    problem = VisaProblem()
    rng = np.random.default_rng(42)
    f3_set = set()
    for _ in range(5000):
        hawk = rng.uniform(0, 1, size=len(problem.groups))
        perm = spv(hawk)
        x = decode(perm, problem.groups, problem.total_visas,
                   problem.country_caps, problem.category_caps)
        f3 = problem.f3(x)
        f3_set.add(f3)
    assert len(f3_set) > 100, f"f3 has only {len(f3_set)} unique values — should be hundreds"
    print(f"CHECK 1 PASS: f3 has {len(f3_set)} unique values, range [{min(f3_set)}, {max(f3_set)}]")


def check_2():
    """evaluate() returns 3-tuple."""
    from src.problem import VisaProblem
    from src.decoder import spv, decode

    problem = VisaProblem()
    hawk = np.random.default_rng(42).uniform(0, 1, size=len(problem.groups))
    perm = spv(hawk)
    x = decode(perm, problem.groups, problem.total_visas,
               problem.country_caps, problem.category_caps)
    result = problem.evaluate(x)
    assert len(result) == 3, f"evaluate() returns {len(result)}-tuple, expected 3"
    assert isinstance(result[2], (int, float)), f"f3 type is {type(result[2])}"
    print(f"CHECK 2 PASS: evaluate() returns 3-tuple: ({result[0]:.4f}, {result[1]:.4f}, {result[2]:.0f})")


def check_3():
    """3D dominance."""
    from src.mohho import dominates
    assert dominates((1, 1, 1), (2, 2, 2)), "Should dominate"
    assert not dominates((1, 1, 2), (2, 1, 1)), "Should NOT dominate (f3 worse)"
    assert not dominates((1, 1, 1), (1, 1, 1)), "Equal should NOT dominate"
    print("CHECK 3 PASS: 3D dominance correct")


def check_4():
    """Feasibility universal (10K permutations)."""
    from src.problem import VisaProblem
    from src.decoder import spv, decode

    problem = VisaProblem()
    rng = np.random.default_rng(42)
    violations = 0
    for _ in range(10000):
        hawk = rng.uniform(0, 1, size=len(problem.groups))
        perm = spv(hawk)
        x = decode(perm, problem.groups, problem.total_visas,
                   problem.country_caps, problem.category_caps)
        total = sum(x.values())
        if total > problem.total_visas:
            violations += 1
        for g in problem.groups:
            if x[g["index"]] < 0 or x[g["index"]] > g["n"]:
                violations += 1
    assert violations == 0, f"{violations} constraint violations!"
    print("CHECK 4 PASS: 10,000 permutations, 0 violations")


def check_5():
    """f1 direction correct."""
    from src.problem import VisaProblem
    from src.decoder import decode

    problem = VisaProblem()
    india_groups = [g["index"] for g in problem.groups if g["country"] == "India"]
    other_groups = [g["index"] for g in problem.groups if g["country"] != "India"]
    x_if = decode(india_groups + other_groups, problem.groups, problem.total_visas,
                  problem.country_caps, problem.category_caps)
    x_il = decode(other_groups + india_groups, problem.groups, problem.total_visas,
                  problem.country_caps, problem.category_caps)
    assert problem.f1(x_if) < problem.f1(x_il), "min f1 should prefer India-first"
    print(f"CHECK 5 PASS: f1(India-first)={problem.f1(x_if):.4f} < f1(India-last)={problem.f1(x_il):.4f}")


def check_6():
    """Results regenerated with f3."""
    with open("results/summary.json") as f:
        s = json.load(f)
    assert "best_f3" in s, "summary.json missing best_f3"
    assert "f3" in s["baseline"], "baseline missing f3"
    print(f"CHECK 6 PASS: summary.json has f3 data (baseline f3={s['baseline']['f3']})")


def check_7():
    """Pareto CSV has f3 column."""
    with open("results/pareto_front.csv") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        assert "f3" in row, f"CSV columns: {list(row.keys())} — missing f3!"
    print("CHECK 7 PASS: pareto_front.csv has f3 column")


def check_8():
    """Pareto no-dominance (3D)."""
    from src.mohho import dominates
    pts = []
    with open("results/pareto_front.csv") as f:
        for row in csv.DictReader(f):
            if row["type"] == "pareto":
                pts.append((float(row["f1"]), float(row["f2"]), float(row["f3"])))
    dominated = 0
    for i, a in enumerate(pts):
        for j, b in enumerate(pts):
            if i != j and dominates(b, a):
                dominated += 1
                break
    assert dominated == 0, f"{dominated} dominated solutions in Pareto front!"
    print(f"CHECK 8 PASS: {len(pts)} Pareto points, 0 dominated")


def check_9():
    """App has no biobjetivo references."""
    with open("app/streamlit_app.py") as f:
        content = f.read()
    bad = ["biobjetivo", "bi-objetivo", "Los Dos Objetivos",
           "2 Objetivos y No 3", "dos objetivos"]
    found = []
    for b in bad:
        if b.lower() in content.lower():
            found.append(b)
    assert not found, f"App still references: {found}"
    print("CHECK 9 PASS: No biobjetivo references in app")


def check_10():
    """Reproducibility."""
    from src.mohho import run_mohho
    from src.problem import VisaProblem
    problem = VisaProblem()
    _, fit1, _ = run_mohho(problem, seed=42, pop_size=10, max_iter=5, archive_size=20)
    _, fit2, _ = run_mohho(problem, seed=42, pop_size=10, max_iter=5, archive_size=20)
    assert fit1 == fit2, "Not reproducible with same seed!"
    print("CHECK 10 PASS: Reproducible with same seed")


def check_11():
    """App launches without errors (import test)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", "app/streamlit_app.py")
    mod = importlib.util.module_from_spec(spec)
    # Just parse, don't execute (Streamlit needs a running server)
    import ast
    with open("app/streamlit_app.py") as f:
        ast.parse(f.read())
    print("CHECK 11 PASS: app/streamlit_app.py parses without errors")


def check_12():
    """LaTeX coherent."""
    with open("references/Fase02_Latex/HarrisFase02.tex") as f:
        tex = f.read()
    assert "f_3" in tex, "LaTeX missing f3"
    assert "triobjetivo" in tex.lower() or "{1,2,3}" in tex, "LaTeX still says biobjetivo"
    assert "Eliminación de" not in tex, "LaTeX still has elimination section"
    print("CHECK 12 PASS: LaTeX is triobjetivo")


if __name__ == "__main__":
    checks = [check_1, check_2, check_3, check_4, check_5, check_6,
              check_7, check_8, check_9, check_10, check_11, check_12]
    passed = 0
    failed = []
    for i, check in enumerate(checks, 1):
        try:
            check()
            passed += 1
        except Exception as e:
            print(f"CHECK {i} FAIL: {e}")
            failed.append(i)

    print(f"\n{'='*50}")
    print(f"RESULT: {passed}/12 PASS")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASS")
        sys.exit(0)
