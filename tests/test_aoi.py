"""Tests for AOI hit testing via controller.point_in_polygon."""

import pytest

from app.controller import point_in_polygon


# Unit square polygon
_SQUARE = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]


def test_centre_inside():
    assert point_in_polygon(0.5, 0.5, _SQUARE) is True


def test_corner_outside():
    assert point_in_polygon(0.0, 0.0, _SQUARE) is False


def test_edge_considered_inside():
    # cv2.pointPolygonTest returns >= 0 for on-boundary
    assert point_in_polygon(0.1, 0.5, _SQUARE) is True


def test_clearly_outside():
    assert point_in_polygon(0.95, 0.95, _SQUARE) is False


def test_degenerate_polygon_returns_false():
    assert point_in_polygon(0.5, 0.5, [(0.0, 0.0)]) is False
    assert point_in_polygon(0.5, 0.5, []) is False
