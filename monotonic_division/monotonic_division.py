from pathlib import Path
from utils.halfedge import HalfEdge, HalfEdgeMesh, Vertex, Face
from utils import interactive_figure

from sortedcontainers import SortedSet # type: ignore
from functools import cmp_to_key
import matplotlib.pyplot as plt
import json
#na razie zakladam ze nie bedzie punktow o rownych y, trzeba bedzie te jakies rotacje dorobic
savefig = False
EPS = 10**(-10)

def loadFigure(dataName = "exportData.json"):
    '''
    Wczytuje określony wielokąt json w folderze "data". 
    '''
    pathToJson = Path(__file__).parent.parent / "data" /dataName
    with open(pathToJson,"r") as jsonFile:
        figure = json.load(jsonFile)
    return figure

class Structures:
    global EPS
    def __init__(self, points):
        self.points = [Vertex(points[i][0], points[i][1], i) for i in range (len(points))]
        
    def prepareHalfEdgeMesh(self): 
        ''' 
        Convert CCW Vertex list to HalfEdge DS (One face with edges).
        '''
        mesh = HalfEdgeMesh(vertices = self.points) 
        ccwFace = []
        N = len(self.points) 
        for i in range (N - 1):
            e1, e2 = mesh.addEdge(self.points[i], self.points[i + 1])
            ccwFace.append(e1)
        #edge case dla ostatniej krawedzi:       
        edge1, edge2 = HalfEdge(), HalfEdge()
        v1, v2 = self.points[-1], self.points[0]
        edge1.origin = v1
        edge2.origin = v2
        v1.outgoingEdge = edge1 #nie zmienam outgoingEdge dla pierwszego wierzcholka bo juz ma dobra krawedz w CCW
        edge1.twin = edge2
        edge2.twin = edge1
        mesh.edges.append(edge1)
        mesh.edges.append(edge2)
        ccwFace.append(edge1)
        for i, edge in enumerate(mesh.edges):
            if edge.origin.y == edge.twin.origin.y: #tymczasowe rozw. bo jak bedzie duzo punktów współliniowych i to jeszcze niesasiadujacych to chyba nie zadziala
                if edge.origin.x < edge.twin.origin.x:
                    edge.k = (edge.origin.x - edge.twin.origin.x)/((edge.origin.y + EPS) - edge.twin.origin.y)
                    edge.l = edge.origin.x - edge.k*(edge.origin.y + EPS)
                
            edge.k = (edge.origin.x - edge.twin.origin.x)/(edge.origin.y - edge.twin.origin.y)
            edge.l = edge.origin.x - edge.k*edge.origin.y
        
        mesh.addFace(ccwFace) #connect (prev, next) and add face
        
        return mesh

    def prepareEvents(self):
        clf = Classification(self.points) 
        clf.classify() #klasyfikuj punkty (dodaj typy)
        return sorted(self.points, key = lambda vertex: (vertex.y, -vertex.x)) #bedziemy zdarzenia pobierać od końca
        
    def cmp(halfedge1, halfedge2):
        if halfedge1 is halfedge2:
            return 0
        if halfedge1.currX() > halfedge2.currX():
            return 1
        elif halfedge1.currX() < halfedge2.currX():
            return -1
        return 0
    
    def prepareSweep(self):
        sweep = SortedSet(key = cmp_to_key(Structures.cmp))
        return sweep
    

class Classification:
    '''
    Klasa do klasyfikacji wierzchołków. Klasyfikacja odbywa się w classify(), punkty rozdzielane są do self.start, self.end, itd.
    '''
    def __init__(self, points, edges = None, eps = 10**(-12)):
        self.eps = eps
        self.points = points
        self.edges = edges
        self.start = []  #przechowuje indeksy punktów początkowych
        self.end = [] #analog.
        self.merge = [] #...
        self.split = []
        self.regular = []

    def det_sarrus(self, A, B, C):
        return A.x*B.y + A.y*C.x + B.x*C.y - C.x*B.y - B.x*A.y - A.x*C.y

    def orient(self, A, B, C):
        det = self.det_sarrus(A, B, C) 
        if det >= self.eps:
            return -1 #C na lewo od AB 
        elif det <= -self.eps:
            return 1 #C na prawo od AB
        return 0 #współliniowe


    def classify(self): 
        '''
        p "powyżej" q <==> p.y > q.y or (p.y == q.y and p.x < q.x) 
        Zakłada, że moc self.points >= 3, ponadto zakłada że 3 punkty sąsiadujące nie mogą być współliniowe - zgodnie z definicją wielokąta(jako łamanej zamkniętej).
        '''
        N = len(self.points)
        for idx, point in enumerate(self.points):
            prev = self.points[(idx - 1) % N]
            next = self.points[(idx + 1) % N]
            if (prev.y > point.y or (prev.y == point.y and prev.x < point.x)) and (next.y > point.y or (next.y == point.y and next.x < point.x)): 
                orientation = self.orient(prev, point, next)
                if orientation == -1: #kąt < pi
                    self.end.append(idx)
                    point.type = 'E' 
                else:
                    self.merge.append(idx)
                    point.type = 'M'
            elif (prev.y < point.y or (prev.y == point.y and prev.x > point.x)) and (next.y < point.y or (next.y == point.y and next.x > point.x)):
                orientation = self.orient(prev, point, next)
                if orientation == -1: #kąt < pi
                    self.start.append(idx)
                    point.type = 'I' #initial
                else:
                    self.split.append(idx)
                    point.type = 'S'
            else: # w pozostałych przypadkach (czyli gdy oba punkty nie są ani "poniżej" ani "powyżej", słowem - jeden powyżej drugi poniżej
                self.regular.append(idx)
                if prev.y > next.y:
                    point.type = 'RL' #regular left, intP po prawej
                else:
                    point.type = 'RR' #regular right, intP po lewej
    def convert(self, pointSubset):
        if pointSubset:
            pts = list(map(lambda i: self.points[i], pointSubset))
            resX, resY = zip(*pts)
            return resX, resY
        
    def visualizeClassification(self):
        global savefig
        fig, ax = plt.subplots()
        xlim = (0,10)
        ylim = (0,10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
 
        conv = self.convert(self.start)
        if conv is not None:
            X, Y = self.convert(self.start)
            ax.scatter(X, Y, color = 'green', s = 50)
        conv = self.convert(self.end)
        if conv is not None:
            X, Y = self.convert(self.end)
            ax.scatter(X, Y, color = 'red', s = 50)
        conv = self.convert(self.merge)
        if conv is not None:        
            X, Y = self.convert(self.merge)
            ax.scatter(X, Y, color = 'blue', s = 50)
        conv = self.convert(self.split)
        if conv is not None:      
            X, Y = self.convert(self.split)
            ax.scatter(X, Y, color = 'cyan', s = 50)
        conv = self.convert(self.regular)
        if conv is not None:      
            X, Y = self.convert(self.regular)
            ax.scatter(X, Y, color = 'brown', s = 50)
        path = Path(__file__).parent / "obrazki" / "przykladowa_klasyfikacja" 
        if savefig:
            plt.savefig(path, dpi = 300)
        plt.show(block = True)


class Division:
    def __init__(self, polygon, events, sweep):
        self.polygon = polygon #mesh
        self.Q = events
        self.T = sweep
        self.addedDiags = set()

    def handleStartVertex(self, v):
        toAdd = v.outgoingEdge
        toAdd.helper = v
        self.T.add(toAdd)

    def handleEndVertex(self, v):
        prevEdge = v.outgoingEdge.prev
        if prevEdge.helper.type == 'M':
            self.addedDiags.add(self.polygon.addDiagDiv(v, prevEdge.helper))
        self.T.discard(prevEdge)
        
    def handleSplitVertex(self, v):
        e = v.outgoingEdge
        e.helper = v
        self.T.add(e)
        left = self.T[self.T.index(e) - 1] #na lewo od e_{i}, na pewno nalezy do T bo dodalem dopiero
        self.addedDiags.add(self.polygon.addDiagDiv(v, left.helper))
        left.helper = v

    def handleMergeVertex(self, v):
        prevEdge = v.outgoingEdge.prev #prevedge tutaj na pewnonie jest diagonalną bo dopiero po zamieceniu v moze powstac do niego diagonalna. 
        if prevEdge.helper.type == 'M':
            self.addedDiags.add(self.polygon.addDiagDiv(v, prevEdge.helper))
        left = self.T[self.T.index(prevEdge) - 1] #na lewo od e_{i-1}, na pewno w T bo jest przy lewym przbu
        self.T.discard(prevEdge)
        if left.helper.type == 'M':
            self.addedDiags.add(self.polygon.addDiagDiv(v, left.helper))
        left.helper = v
    
    
    def handleRegularVertex(self, v):
        if v.type == 'RL': #intP po prawej    
            prevEdge = v.outgoingEdge.prev
            if prevEdge.helper.type == 'M':
                self.addedDiags.add(self.polygon.addDiagDiv(v, prevEdge.helper))
            self.T.discard(prevEdge)
            v.outgoingEdge.helper = v
            self.T.add(v.outgoingEdge)
        else:
            probe = v.outgoingEdge 
            self.T.add(probe) #puszczamy sonde
            left = self.T[self.T.index(probe) - 1]
            self.T.discard(probe) #wyciagamy sonde
            if left.helper.type == 'M':
                self.addedDiags.add(self.polygon.addDiagDiv(v, left.helper))
            left.helper = v

    def createAdjacentFaces(self, diag, visited):
        if not visited[diag]:
            newface = Face()
            diag.face = newface
            visited[diag] = True
            newface.outerEdge = diag
            p = diag.next
            while p != diag: 
                print(p,diag)
                p.face = newface
                visited[p] = True
                p = p.next
            self.polygon.faces.append(newface)
        if not visited[diag.twin]:
            newface = Face()
            diagTwin = diag.twin
            newface.outerEdge = diagTwin
            diagTwin.face = newface
            visited[diagTwin] = True
            q = diag.next
            while q != diag:
                q.face = newface
                visited[q] = True
                q = q.next
            self.polygon.faces.append(newface)

    def updateFaces(self): 
        '''
        Updating faces has to take place after the algorithm of division, because we cant afford time cost of such an operation during the algorithm.
        updateFaces works similar to unconnected graph traversal. Polygon consists of unconnected cycles (faces). If we want to visit every edge, we have to start traversal at every diagonal and its twin.
        Warning: we shouldn't start at non-diagonal, because if we did so, we would traverse whole initial polygon and every initial edge would have the same face.
        Result: A polygon separated into faces. 
        '''
        if len(self.addedDiags) == 0:
            return
        self.polygon.faces = [] #bo na razie miał jedną ścianę, ale zostanie ona usunięta, bo podzielono ją na kilka mniejszych.
        visited = dict()
        for diag in self.addedDiags:
            visited[diag] = False
            visited[diag.twin] = False
        for diag in self.addedDiags:
            if not visited[diag] or not visited[diag.twin]:
                self.createAdjacentFaces(diag, visited)

    def divide(self):
        handle = {'M': self.handleMergeVertex, 'S': self.handleSplitVertex, 'I': self.handleStartVertex, 'RL': self.handleRegularVertex, 'RR': self.handleRegularVertex, 'E': self.handleEndVertex}
        while self.Q:
            event = self.Q.pop()
            print(event.type)
            HalfEdge.currY = event.y
            func = handle[event.type]
            func(event)
        self.updateFaces()
        return self.polygon


    # def convertToEdgeSet(self):
    #     # visited = dict()
    #     edges = set()
    #     for face in self.polygon.faces:
    #         start = face.outerEdge
    #         p = p.next
    #         edges.add(start)
    #         while p != start:
    #             edges.add(p)
    #             p = p.next  
    #     return edges

    def visualize(self):
        fig, ax = plt.subplots()
        xlim = (0,10)
        ylim = (0,10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        edgeSet = set()
        for edge in self.polygon.edges:
            if edge not in edgeSet and edge.twin not in edgeSet:
                edgeSet.add(edge)
        pts = [(edge.origin, edge.twin.origin) for edge in edgeSet]
        for pair in pts:
            ax.plot((pair[0].x, pair[1].x), (pair[0].y, pair[1].y))
        plt.show()


if __name__ == "__main__":
    figure = loadFigure()
    points = figure["points"]
    X, Y = zip(*points)
    for i in range (len(points)):
        plt.plot([X[i], X[(i + 1)%len(points)]], [Y[i], Y[(i + 1)%len(points)]])
    plt.show()
    prepare = Structures(points)
    division = Division(prepare.prepareHalfEdgeMesh(), prepare.prepareEvents(), prepare.prepareSweep())
    division.divide()
    division.visualize()