from itertools import combinations


def no_overlap(fps) -> bool:
    for a, b in combinations(fps, 2):
        if a.polygon.intersection(b.polygon).area > 0.05:
            return False
    return True
