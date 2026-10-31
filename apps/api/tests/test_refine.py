"""Board-corner and check-ruler refinement from masks (app/domain/refine.py). Offline and synthetic: a board quad and a
ruler drawn with OpenCV on a 2336x1744 canvas stand in for SAM masks, so these tests pin the geometry and the fallbacks,
not how well SAM segments a real photo."""
import cv2
import numpy as np
import pytest

from app.domain import refine
from app.domain.calibration import CalibrationError, calibrate
from app.domain.refine import refine_board_corners, refine_check_points

W, H = 2336, 1744
SIZE = (W, H)
BOARD = np.array([[110, 95], [2230, 70], [2260, 1650], [80, 1670]], float)  # TL, TR, BR, BL (keystoned)
BOARD_JITTER = np.array([[30, 22], [-28, 18], [35, -30], [-25, -20]], float)  # Spark: 1-2% of the image off


def poly_mask(pts) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(pts)).astype(np.int32)], 1)
    return m


def ruler_mask(p0, p1, thick=8, tick=36, tick_w=7, label=False) -> np.ndarray:
    """A flat-ended line from p0 to p1 with a bar (end tick) across it at each end, centred on the end point."""
    m = np.zeros((H, W), np.uint8)
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-u[1], u[0]])

    def bar(centre, along, across):  # a rectangle: `along` px along the line, `across` px across it
        a, c = u * along / 2, n * across / 2
        cv2.fillPoly(m, [np.round([centre - a - c, centre + a - c, centre + a + c, centre - a + c]).astype(np.int32)], 1)

    bar((p0 + p1) / 2, float(np.linalg.norm(p1 - p0)), thick)
    if tick:
        bar(p0, tick_w, tick)
        bar(p1, tick_w, tick)
    if label:  # the "CHECK 2 m" text: a separate piece under the line
        cv2.rectangle(m, tuple(((p0 + p1) / 2 + n * 60 - [150, 12]).astype(int)),
                      tuple(((p0 + p1) / 2 + n * 60 + [150, 12]).astype(int)), 1, -1)
    return m


def close(a, b, tol):
    return np.abs(np.asarray(a, float) - np.asarray(b, float)).max() <= tol


# ---- board ------------------------------------------------------------------------------------------------------
def test_board_corners_come_from_the_mask_ordered_tl_tr_br_bl_not_from_the_seed():
    seed = BOARD + BOARD_JITTER
    r = refine_board_corners(poly_mask(BOARD), seed, SIZE)
    assert r.source == "refined" and "approxPolyDP" in r.detail
    assert close(r.points, BOARD, 1.5)  # exact corners, in the order TL, TR, BR, BL
    assert np.abs(seed - BOARD).max() > 20  # whereas Spark's seed was 20+ px off
    # the points are in the order TL, TR, BR, BL whatever order the mask's contour runs in
    assert r.points[0][0] < r.points[1][0] and r.points[0][1] < r.points[3][1] and r.points[2][1] > r.points[1][1]


def test_holes_a_door_tab_and_stray_pixels_do_not_move_the_corners():
    m = poly_mask(BOARD)
    cv2.rectangle(m, (40, 700), (100, 900), 1, -1)  # a door marker sticking out of the left edge
    for y in (200, 1000):  # objects on the board that SAM left out of the mask: holes
        cv2.rectangle(m, (400, y), (900, y + 300), 0, -1)
    cv2.circle(m, (1200, 900), 0, 1, -1)  # a stray pixel inside
    cv2.circle(m, (5, 1740), 4, 1, -1)  # and a stray blob outside, not connected to the board
    r = refine_board_corners(m, BOARD + BOARD_JITTER, SIZE)
    assert r.source == "refined" and close(r.points, BOARD, 1.5)


def test_a_rotated_board_keeps_the_screen_order_tl_tr_br_bl():
    c = np.array([1168.0, 872.0])
    rot = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), -20, 1.0)  # 20 degrees clockwise on screen
    rect = np.array([[-800, -550], [800, -550], [800, 550], [-800, 550]], float) + c
    q = cv2.transform(rect.reshape(-1, 1, 2), rot).reshape(4, 2)
    r = refine_board_corners(poly_mask(q), q + 15, SIZE)
    assert r.source == "refined" and close(r.points, q, 1.5)


def test_min_area_rect_is_the_fallback_when_no_quadrilateral_fits_and_is_checked_like_any_result():
    tri = np.array([[100, 100], [2000, 100], [1000, 1500]])
    cnts, _ = cv2.findContours(poly_mask(tri), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    q, method = refine._quad(cnts[0])
    assert method == "minAreaRect" and close(sorted(q.tolist()), sorted([[100, 100], [2000, 100], [2000, 1500], [100, 1500]]), 1)
    r = refine_board_corners(poly_mask(tri), [[100, 100], [2000, 100], [2000, 1500], [100, 1500]], SIZE)
    assert r.source == "seed" and "minAreaRect" in r.detail and "fills" in r.detail  # a triangle is not a rectangle


def _board_seed():
    return (BOARD + BOARD_JITTER).tolist()


def _l_shape():
    m = poly_mask(BOARD)
    cv2.rectangle(m, (1000, 0), (2300, 1000), 0, -1)
    return m


def _disc():
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (1168, 872), 800, 1, -1)
    return m


def _board_far_from_seed():
    return poly_mask(BOARD + np.array([150, 0]))  # 6.4% of the width from where Spark put it


@pytest.mark.parametrize("mask, why", [
    (None, "no mask"),
    (np.zeros((H, W), bool), "empty"),
    (np.ones((H // 2, W // 2), bool), "mask is"),                     # wrong size for the image
    (_l_shape(), "from the Spark seed"),                              # a corner of the L is far from every seed corner
    (_disc(), "fills"),                                               # 4 hull points fit, the area does not
    (_board_far_from_seed(), "from the Spark seed"),
    (np.pad(np.ones((1, 1), np.uint8), ((5, H - 6), (5, W - 6))), "not convex"),  # a single pixel
])
def test_every_failed_refinement_keeps_the_spark_seed_and_says_why(mask, why):
    seed = _board_seed()
    r = refine_board_corners(mask, seed, SIZE)
    assert r.source == "seed" and r.points == [list(map(float, p)) for p in seed] and why in r.detail


def test_a_non_convex_or_mis_ordered_quad_is_rejected():
    dart = np.array([[0, 0], [100, 0], [30, 30], [0, 100]], float)  # one reflex corner
    bowtie = np.array([[0, 0], [100, 100], [100, 0], [0, 100]], float)
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    assert not refine._convex(dart) and not refine._convex(bowtie) and refine._convex(square)
    assert refine._order(square[[2, 0, 3, 1]]).tolist() == square.tolist()  # any input order -> TL, TR, BR, BL


def test_bad_seed_shapes_are_a_programming_error_not_a_silent_fallback():
    with pytest.raises(ValueError):
        refine_board_corners(poly_mask(BOARD), [[0, 0]] * 3, SIZE)
    with pytest.raises(ValueError):
        refine_check_points(None, [[0, 0], [float("nan"), 1]], SIZE)


# ---- check ruler ------------------------------------------------------------------------------------------------
P0, P1 = (700.0, 330.0), (1930.0, 330.0)  # a 1230 px ruler


def test_ruler_ends_are_the_tick_centres_not_the_outer_edge_of_the_ticks():
    seed = [[P0[0] - 28, P0[1] + 21], [P1[0] + 30, P1[1] - 14]]  # Spark: about 1.5% off, and 58 px too long
    r = refine_check_points(ruler_mask(P0, P1), seed, SIZE)
    assert r.source == "refined" and "end ticks" in r.detail
    assert close(r.points, [P0, P1], 0.75)
    plain = refine_check_points(ruler_mask(P0, P1, tick=0, thick=8), seed, SIZE)  # the same line without ticks
    assert plain.source == "refined" and "mask extent" in plain.detail and close(plain.points, [P0, P1], 1.0)
    ticked_extent = np.nonzero(ruler_mask(P0, P1).any(axis=0))[0]
    assert ticked_extent.max() - ticked_extent.min() > P1[0] - P0[0] + 5  # the ticks add 6-7 px to the extent: not used


def test_a_tilted_ruler_with_a_label_piece_in_the_mask():
    a, b = np.array([600.0, 400.0]), np.array([1800.0, 520.0])  # about 5.7 degrees
    r = refine_check_points(ruler_mask(a, b, label=True), [a + [25, 20], b - [18, 24]], SIZE)
    assert r.source == "refined" and "end ticks" in r.detail
    assert close(r.points, [a, b], 1.2)  # the label under the line does not move the ends or the centre line


def test_ruler_points_follow_the_order_of_the_seed():
    r = refine_check_points(ruler_mask(P0, P1), [list(P1), list(P0)], SIZE)  # seed lists the right end first
    assert r.source == "refined" and close(r.points, [P1, P0], 0.75)


def _blob():
    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (700, 300), (1900, 900), 1, -1)  # 1200 x 600
    return m


def _speck():
    m = np.zeros((H, W), np.uint8)
    m[330, 700:705] = 1
    return m


@pytest.mark.parametrize("mask, seed, why", [
    (None, [P0, P1], "no mask"),
    (np.zeros((H, W), bool), [P0, P1], "empty"),
    (np.ones((10, 10), bool), [P0, P1], "mask is"),
    (_speck(), [P0, P1], "pixels"),                                                           # 5 pixels
    (_blob(), [P0, P1], "not a thin line"),                                                   # 1200x600 block
    (ruler_mask(P0, P1), [P0, (P0[0] + 800, P0[1])], "x the Spark seed's"),                  # 1.54x the seed length
    (ruler_mask(P0, P1), [P0, (P0[0] + 1700, P0[1])], "x the Spark seed's"),                 # 0.72x
    (ruler_mask(P0, P1), [(P0[0] + 160, P0[1]), (P1[0] + 160, P1[1])], "from the Spark seed"),  # right length, 6.8% away
    (ruler_mask(P0, P1), [P0, P0], "zero length"),
])
def test_every_failed_ruler_refinement_keeps_the_spark_seed_and_says_why(mask, seed, why):
    r = refine_check_points(mask, seed, SIZE)
    assert r.source == "seed" and r.points == [list(map(float, p)) for p in seed] and why in r.detail


# ---- what it is for ---------------------------------------------------------------------------------------------
def test_refined_points_pass_calibrations_independent_check_where_the_coarse_seeds_fail_it():
    """Board 600x450 cm photographed with keystone; a 200 cm ruler lies on it. Seeds a few % off are rejected by the
    unchanged 3% length check; the same scene refined from masks measures the ruler within 0.5% and the board corners
    are exact. The ruler is never used to place the board: calibrate() still takes it as the independent reference."""
    to_px = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [600, 0], [600, 450], [0, 450]]), BOARD.astype(np.float32))
    ends = cv2.perspectiveTransform(np.float32([[[350, 36]], [[550, 36]]]), to_px).reshape(2, 2)
    seed_board = (BOARD + BOARD_JITTER).tolist()
    seed_ruler = [(ends[0] + [-22, 17]).tolist(), (ends[1] + [40, -12]).tolist()]
    with pytest.raises(CalibrationError, match="independent check length"):
        calibrate(seed_board, 600, 450, seed_ruler, 200.0, SIZE)
    board = refine_board_corners(poly_mask(BOARD), seed_board, SIZE)
    ruler = refine_check_points(ruler_mask(ends[0], ends[1]), seed_ruler, SIZE)
    assert board.source == ruler.source == "refined"
    cal = calibrate(board.points, 600, 450, ruler.points, 200.0, SIZE)
    assert cal.check_error_pct < 0.5
