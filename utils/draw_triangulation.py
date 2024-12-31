import matplotlib.pyplot as plt
import numpy as np


def drawTriangle(pointsIndices, points):
    trianglePoints = list(map(lambda idx: points[idx], pointsIndices))
    trianglePoints = np.vstack([trianglePoints, trianglePoints[0]]) #dodaj pierwszy punkt na koniec 
    plt.plot(trianglePoints[:, 0], trianglePoints[:, 1], color = 'blue', linewidth = 1)

def drawEdge(i1, i2, points):
    plt.plot([points[i1][0], points[i2][0]], [points[i1][1], points[i2][1]], color = 'black', linewidth = 4)

# def drawEdge(i1, i2, points):
#     # Calculate the start and end points
#     start = points[i1]
#     end = points[i2]
    
#     # Draw an arrow from start to end
#     plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
#              head_width=0.1, head_length=0.2, fc='black', ec='black')

def draw(points, triangles, edges = []):
    '''
    param: points - list of points, triangles - list of indices: [(i,j,k): i,j,k indices of points in points]
    '''
    for triangle in triangles:
        drawTriangle(triangle, points)
    X, Y = zip(*points)
    plt.scatter(X, Y, color='orange')

    if len(edges):
        for i1, i2 in edges:
            drawEdge(i1, i2, points)

    plt.show()