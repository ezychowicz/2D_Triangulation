import matplotlib.pyplot as plt
import numpy as np


def drawTriangle(pointsIndices, points):
    trianglePoints = list(map(lambda idx: points[idx], pointsIndices))
    trianglePoints = np.vstack([trianglePoints, trianglePoints[0]]) #dodaj pierwszy punkt na koniec 
    plt.plot(trianglePoints[:, 0], trianglePoints[:, 1], color = 'blue', linewidth = 1)

def draw(points, triangles):
    '''
    param: points - list of points, triangles - list of indices: [(i,j,k): i,j,k indices of points in points (CCW order)]
    '''
    for triangle in triangles:
        drawTriangle(triangle, points)
    X, Y = zip(*points)
    plt.scatter(X, Y, color='orange')
    plt.show()
