"""Center selection and real rotated-polygon overlap, before CSV checks."""

import pytest

from app.core.inspection_config import ObbConfig
from app.domain.entities.obb_result import OBB_CLASSES, ObbDetection
from app.domain.services.appearance_classifier import AppearanceClassifier

CENTER = (5.0, 5.0)
CENTRAL = ((3, 3), (7, 3), (7, 7), (3, 7))
SIDE = ((8, 3), (10, 3), (10, 7), (8, 7))


def box(polygon=CENTRAL, *, name="Solder_Paste", confidence=0.8):
    return ObbDetection(name, confidence, polygon)


def classify(*detections, center=CENTER, tolerance=0.05):
    return AppearanceClassifier(
        ObbConfig().class_thresholds, center_tolerance_ratio=tolerance
    ).classify(detections, image_center=center)


@pytest.mark.parametrize("reverse_order", [False, True])
def test_disjoint_neighbors_are_ignored_regardless_of_confidence_or_order(
    reverse_order,
):
    target = box(confidence=0.7)
    neighbors = (box(SIDE, confidence=0.99), box(((0, 0), (2, 0), (2, 2), (0, 2))))
    detections = (target, *neighbors)
    if reverse_order:
        detections = detections[::-1]
    result = classify(*detections)
    assert result.passed and result.reason == ""
    assert result.selected == target
    assert (result.ok_count, result.center_count) == (3, 1)
    assert (result.overlap_count, result.ignored_count) == (0, 2)
    assert result.retained == detections


@pytest.mark.parametrize("detections", [(box(SIDE),), (box(SIDE), box(SIDE))])
def test_only_off_center_solder_paste_is_ng_even_when_confident(detections):
    result = classify(*detections)
    assert not result.passed and result.reason == "missing_center_solder_paste"
    assert result.selected is None and result.center_count == 0


def test_low_confidence_center_cannot_be_replaced_with_confident_neighbor():
    result = classify(box(confidence=0.699), box(SIDE, confidence=0.99))
    assert result.reason == "missing_center_solder_paste"
    assert result.ok_count == 1


@pytest.mark.parametrize(
    "neighbor",
    [
        ((6, 4), (9, 4), (9, 6), (6, 6)),
        ((5.5, 5.5), (6.5, 5.5), (6.5, 6.5), (5.5, 6.5)),
    ],
    ids=["partial-overlap", "nested-away-from-center"],
)
def test_positive_area_overlap_with_center_box_is_ng(neighbor):
    result = classify(box(), box(neighbor))
    assert result.reason == "overlapping_qualified_solder_paste"
    assert result.center_count == 1 and result.overlap_count == 1
    assert result.selected == box()


def test_multiple_boxes_containing_center_are_ambiguous():
    result = classify(box(), box(confidence=0.99))
    assert result.reason == "multiple_qualified_solder_paste"
    assert result.center_count == 2 and result.selected is None


@pytest.mark.parametrize(
    "polygon",
    [
        ((7, 4), (9, 4), (9, 6), (7, 6)),
        ((7, 7), (9, 7), (9, 9), (7, 9)),
    ],
    ids=["touching-edge", "touching-vertex"],
)
def test_boundary_contact_without_intersection_area_is_ignored(polygon):
    result = classify(box(), box(polygon))
    assert result.passed and result.ignored_count == 1


@pytest.mark.parametrize("center", [(3.0, 5.0), (3.0, 3.0)])
def test_center_on_box_boundary_is_included(center):
    assert classify(box(), center=center).passed


@pytest.mark.parametrize("name", OBB_CLASSES[1:])
def test_qualified_ng_anywhere_blocks_even_with_valid_center(name):
    result = classify(box(), box(SIDE), box(SIDE, name=name, confidence=0.25))
    assert not result.passed and result.reason == name
    assert result.center_count == 1 and result.ng_count == 1


def test_unqualified_overlapping_solder_and_ng_are_ignored():
    result = classify(
        box(), box(confidence=0.699), box(name="Bridge", confidence=0.249)
    )
    assert result.passed and result.retained == (box(),)


def test_no_qualified_solder_paste_keeps_missing_detection_reason():
    assert classify().reason == "missing_qualified_solder_paste"


def test_center_selection_uses_rotated_polygon_instead_of_axis_aligned_bounds():
    diamond = ((5, 1), (9, 5), (5, 9), (1, 5))
    assert (
        classify(box(diamond), center=(8.0, 8.0)).reason
        == "missing_center_solder_paste"
    )
    assert classify(box(diamond)).passed


@pytest.mark.parametrize("reverse_winding", [False, True])
def test_rotated_neighbors_with_intersecting_bounds_but_disjoint_polygons_are_ignored(
    reverse_winding,
):
    central = ((1, 2), (2, 1), (9, 8), (8, 9))
    neighbor = ((1, 5), (2, 4), (9, 11), (8, 12))
    if reverse_winding:
        central, neighbor = central[::-1], neighbor[::-1]
    result = classify(box(central), box(neighbor))
    assert result.passed and result.ignored_count == 1


def test_rotated_neighbor_with_real_overlap_is_ng():
    central = ((1, 2), (2, 1), (9, 8), (8, 9))
    neighbor = ((6, 7), (7, 6), (10, 9), (9, 10))
    result = classify(box(central), box(neighbor))
    assert result.reason == "overlapping_qualified_solder_paste"


@pytest.mark.parametrize(
    "polygon",
    [
        ((3, 3), (3, 3), (3, 3), (3, 3)),
        ((3, 3), (4, 4), (5, 5), (6, 6)),
        ((3, 3), (7, 7), (3, 7), (7, 3)),
    ],
    ids=["zero-size", "collinear", "self-crossing"],
)
def test_invalid_solder_geometry_never_passes_or_gets_ignored(polygon):
    assert classify(box(), box(polygon)).reason == "invalid_solder_paste_geometry"


@pytest.mark.parametrize(
    "center", [(0, 1), (-1, 1), (1, float("nan")), (1, float("inf"))]
)
def test_invalid_image_center_cannot_pass(center):
    with pytest.raises(ValueError, match="Image center"):
        classify(box(), center=center)


@pytest.mark.parametrize("left,passed", [(5.49, True), (5.5, True), (5.51, False)])
def test_slightly_offset_box_uses_inclusive_point_to_polygon_distance(left, passed):
    target = box(((left, 4), (7, 4), (7, 6), (left, 6)))
    result = classify(target)
    assert result.passed == passed
    assert result.center_tolerance_px == 0.5
    assert result.center_selection_mode == ("within_tolerance" if passed else "none")
    assert result.selected == (target if passed else None)


@pytest.mark.parametrize("tolerance,passed", [(0, False), (0.05, False), (0.1, True)])
def test_center_tolerance_can_be_disabled_or_adjusted(tolerance, passed):
    target = box(((5.75, 4), (7, 4), (7, 6), (5.75, 6)))
    assert classify(target, tolerance=tolerance).passed == passed


def test_exact_center_takes_priority_over_disjoint_neighbor_within_margin():
    target = box(((4.9, 4.9), (5.1, 4.9), (5.1, 5.1), (4.9, 5.1)))
    neighbor = box(((5.2, 4.8), (5.6, 4.8), (5.6, 5.2), (5.2, 5.2)), confidence=0.99)
    result = classify(neighbor, target)
    assert result.passed and result.selected == target
    assert result.center_selection_mode == "contains_center"
    assert result.center_count == 1 and result.ignored_count == 1


def test_multiple_offset_candidates_remain_ng_instead_of_guessing():
    left = box(((4.3, 4), (4.7, 4), (4.7, 6), (4.3, 6)))
    right = box(((5.3, 4), (5.7, 4), (5.7, 6), (5.3, 6)), confidence=0.99)
    result = classify(left, right)
    assert result.reason == "multiple_qualified_solder_paste"
    assert result.center_selection_mode == "within_tolerance"
    assert result.center_count == 2 and result.selected is None


def test_offset_center_target_still_rejects_overlapping_neighbor():
    target = box(((5.25, 4), (6.5, 4), (6.5, 6), (5.25, 6)))
    neighbor = box(((6, 4), (8, 4), (8, 6), (6, 6)))
    result = classify(target, neighbor)
    assert result.reason == "overlapping_qualified_solder_paste"
    assert result.center_count == 1 and result.overlap_count == 1


def test_offset_target_ignores_far_neighbor_but_qualified_ng_still_blocks():
    target = box(((5.25, 4), (6.5, 4), (6.5, 6), (5.25, 6)))
    assert classify(target, box(SIDE)).passed
    assert classify(target, box(SIDE, name="Bridge")).reason == "Bridge"
    assert (
        classify(box(target.polygon, confidence=0.699)).reason
        == "missing_qualified_solder_paste"
    )


@pytest.mark.parametrize("offset,passed", [(0.3, True), (0.4, False)])
def test_diagonal_gap_uses_euclidean_distance_instead_of_axis_wise_margin(
    offset, passed
):
    target = box(((5 + offset, 5 + offset), (7, 5 + offset), (7, 7), (5 + offset, 7)))
    assert classify(target).passed == passed


@pytest.mark.parametrize("left,passed", [(104, True), (104.01, False)])
def test_tolerance_scales_with_original_shorter_side(left, passed):
    target = box(((left, 30), (120, 30), (120, 50), (left, 50)))
    result = classify(target, center=(100.0, 40.0))
    assert result.center_tolerance_px == 4
    assert result.passed == passed


def test_tolerance_uses_actual_rotated_polygon_boundary():
    diamond = ((5, 1), (9, 5), (5, 9), (1, 5))
    result = classify(box(diamond), center=(7.5, 7.5))
    assert result.passed and result.center_selection_mode == "within_tolerance"
    assert (
        classify(box(diamond), center=(8.0, 8.0)).reason
        == "missing_center_solder_paste"
    )
