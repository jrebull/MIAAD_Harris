"""Tests for hypervolume computation."""

import pytest
from src.mohho import compute_hypervolume


def test_hv_two_points():
    """Known HV for two points: (1,3) and (2,1) with ref (3,4)."""
    pts = [(1.0, 3.0), (2.0, 1.0)]
    hv = compute_hypervolume(pts, ref_point=(3.0, 4.0))
    assert abs(hv - 4.0) < 1e-10, f"Expected HV=4.0, got {hv}"


def test_hv_three_points():
    """Known HV for three points with ref (5,5)."""
    pts = [(1.0, 4.0), (2.0, 2.0), (4.0, 1.0)]
    hv = compute_hypervolume(pts, ref_point=(5.0, 5.0))
    assert abs(hv - 11.0) < 1e-10, f"Expected HV=11.0, got {hv}"


def test_hv_single_point():
    """HV for a single point."""
    pts = [(2.0, 3.0)]
    hv = compute_hypervolume(pts, ref_point=(5.0, 5.0))
    expected = 3.0 * 2.0  # width=3, height=2
    assert abs(hv - expected) < 1e-10


def test_hv_empty():
    """HV for empty list should be 0."""
    assert compute_hypervolume([]) == 0.0


def test_hv_point_beyond_ref():
    """Points beyond reference should contribute 0."""
    pts = [(6.0, 6.0)]
    hv = compute_hypervolume(pts, ref_point=(5.0, 5.0))
    assert hv == 0.0


def test_hv_dominated_point_ignored():
    """Dominated points should not increase HV."""
    pts_non_dom = [(1.0, 3.0), (2.0, 1.0)]
    hv1 = compute_hypervolume(pts_non_dom, ref_point=(5.0, 5.0))

    pts_with_dom = [(1.0, 3.0), (2.0, 1.0), (2.0, 4.0)]
    hv2 = compute_hypervolume(pts_with_dom, ref_point=(5.0, 5.0))

    assert abs(hv1 - hv2) < 1e-10, "Dominated point should not change HV"


def test_hv_fixed_ref_point():
    """compute_hypervolume with default ref should use HV_REF_POINT."""
    from src.mohho import HV_REF_POINT
    pts = [(8.0, 5.0)]
    hv = compute_hypervolume(pts)
    expected = (HV_REF_POINT[0] - 8.0) * (HV_REF_POINT[1] - 5.0)
    assert abs(hv - expected) < 1e-10
