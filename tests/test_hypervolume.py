"""Tests for 3D hypervolume computation."""

import pytest
from src.mohho import compute_hypervolume, _hv_2d


def test_hv_2d_two_points():
    """Known 2D HV for two points: (1,3) and (2,1) with ref (3,4)."""
    pts = [(1.0, 3.0), (2.0, 1.0)]
    hv = _hv_2d(pts, (3.0, 4.0))
    assert abs(hv - 4.0) < 1e-10, f"Expected HV=4.0, got {hv}"


def test_hv_2d_single_point():
    """2D HV for a single point."""
    pts = [(2.0, 3.0)]
    hv = _hv_2d(pts, (5.0, 5.0))
    expected = 3.0 * 2.0  # width=3, height=2
    assert abs(hv - expected) < 1e-10


def test_hv_3d_single_point():
    """3D HV for a single point = product of distances to ref."""
    pts = [(1.0, 2.0, 3.0)]
    ref = (5.0, 5.0, 10.0)
    hv = compute_hypervolume(pts, ref_point=ref)
    expected = (5.0 - 1.0) * (5.0 - 2.0) * (10.0 - 3.0)
    assert abs(hv - expected) < 1e-10, f"Expected {expected}, got {hv}"


def test_hv_3d_empty():
    """HV for empty list should be 0."""
    assert compute_hypervolume([]) == 0.0


def test_hv_3d_point_beyond_ref():
    """Points beyond reference should contribute 0."""
    pts = [(6.0, 6.0, 60000.0)]
    hv = compute_hypervolume(pts, ref_point=(5.0, 5.0, 50000.0))
    assert hv == 0.0


def test_hv_3d_two_points():
    """Known 3D HV for two non-dominated points."""
    # Point A: (1, 3, 2), Point B: (2, 1, 4), ref: (5, 5, 5)
    pts = [(1.0, 3.0, 2.0), (2.0, 1.0, 4.0)]
    ref = (5.0, 5.0, 5.0)
    hv = compute_hypervolume(pts, ref_point=ref)
    # Slice 1: f1=1 to 2, width=1, 2D front = {(3,2)}, ref=(5,5) -> (5-3)*(5-2)=6 -> 1*6=6
    # Slice 2: f1=2 to 5, width=3, 2D front = {(3,2),(1,4)}, ref=(5,5)
    #   sweep: sort by first coord: (1,4),(3,2) ->
    #     (1,4): h=5-4=1, w=5-1=4 -> 4, prev=4
    #     (3,2): h=4-2=2, w=5-3=2 -> 4, prev=2  -> total 2D = 8
    #   -> 3*8=24
    # Total = 6+24 = 30
    assert abs(hv - 30.0) < 1e-10, f"Expected 30.0, got {hv}"


def test_hv_3d_default_ref():
    """compute_hypervolume with default ref should use HV_REF_POINT."""
    from src.mohho import HV_REF_POINT
    pts = [(8.0, 5.0, 10000.0)]
    hv = compute_hypervolume(pts)
    expected = (HV_REF_POINT[0] - 8.0) * (HV_REF_POINT[1] - 5.0) * (HV_REF_POINT[2] - 10000.0)
    assert abs(hv - expected) < 1e-10
