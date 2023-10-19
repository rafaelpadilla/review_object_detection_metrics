from shapely.geometry import Polygon
from shapely import intersection


def trials_with_polygon_intersection():
    points1 = [(0, 0), (0, 1), (1, 1), (1, 0)]
    points2_no_intersect = [(5, 5), (5, 6), (6, 6), (6, 5)]
    poly1 = Polygon(points1)
    poly2_no_intersect = Polygon(points2_no_intersect)
    res_poly = intersection(poly1, poly2_no_intersect)
    if res_poly.is_empty:
        print("No intersection")


if __name__ == "__main__":
    trials_with_polygon_intersection()
