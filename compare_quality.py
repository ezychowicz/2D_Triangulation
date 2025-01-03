import math
from monotonic_division import loadFigure, triangulate
import delaunay
import generate_sun_like_figure
def calculate_alpha(a, b, c):
    cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
    alpha_radians = math.acos(cos_alpha)
    return alpha_radians

def length(A, B):
    x1, y1 = A
    x2, y2 = B
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def result(points, triangles):
    measuresSum = 0
    for triangle in triangles:
        trianglePoints = list(map(lambda idx: points[idx], triangle))
        a, b, c = length(trianglePoints[0], trianglePoints[1]), length(trianglePoints[1], trianglePoints[2]), length(trianglePoints[0], trianglePoints[2])
        measure = min(calculate_alpha(a,b,c), calculate_alpha(b, a, c), calculate_alpha(c, a, b))
        measuresSum += measure
    return measuresSum/len(triangles)*360/(2*math.pi)

if __name__ == '__main__':
    # points = loadFigure("mirroredmountains.json")["points"]
    points = generate_sun_like_figure.snowflake(2, 10)
    print(f"DELAUNAY:{result(points, delaunay.triangulate(points))}")
    print(f"MD:{result(points, triangulate(points))}")
