from pathlib import Path
from utils.halfedge import HalfEdge, HalfEdgeMesh, Vertex, Face
from utils import interactive_figure
import json
from sortedcontainers import SortedSet
from functools import cmp_to_key

def loadFigure(dataName = "exportData.json"):
    '''
    Wczytuje określony wielokąt json w folderze "data". 
    '''
    pathToJson = Path(__file__).parent / "data" /dataName
    with open(pathToJson,"r") as jsonFile:
        figure = json.load(jsonFile)
    return figure

class Structures:
    def __init__(self, points):
        self.points = list(map(lambda A: Vertex(A[0], A[1]), points))

    def prepareHalfEdgeMesh(self): 
        ''' 
        Convert CCW Vertex list to HalfEdge DS (One face with edges).
        '''
        mesh = HalfEdgeMesh(vertices = self.points) 
        
        N = len(self.points) 
        for i in range (N):
            mesh.addEdge(self.points[i], self.points[i + 1])
        mesh.addEdge(self.points[N - 1], self.points[0])

        mesh.addFace(mesh.edges) #connect mesh.edges (prev, next) and add face
        return mesh

    def prepareEvents(self):
        return sorted(self.points, key = lambda vertex: (vertex.y, vertex.x))
        
    def cmp(self):
        pass
    
    def prepareSweep():
        sweep = SortedSet(key = cmp_to_key(Structures.cmp))

class Division:
    def __init__(self, polygon, events, sweep):
        self.polygon = polygon
        self.Q = events
        self.T = sweep

    def divide():
        pass

if __name__ == "__main__":
    figure = loadFigure()
    points = figure["points"]
    prepare = Structures(points)
    division = Division(prepare.prepareHalfEdgeMesh(), prepare.prepareEvents(), prepare.prepareSweep())
    
    