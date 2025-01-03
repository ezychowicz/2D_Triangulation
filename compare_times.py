import generate_sun_like_figure
import delaunay
import monotonic_division
import time
import numpy as np
import utils.draw_triangulation



for n in [2]:
    print(f"n={n}")
    points = generate_sun_like_figure.snowflake(n, 100)
    start = time.time()
    delaunay.triangulate(points)
    end = time.time()
    print(f"DELAUNAY: {end - start}")
    start = time.time()
    monotonic_division.triangulate(points)
    end = time.time()
    print(f"MD: {end - start}")