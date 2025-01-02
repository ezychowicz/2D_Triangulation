from pathlib import Path
from utils.halfedge import HalfEdge, HalfEdgeMesh, Vertex, Face
from utils import interactive_figure, draw_triangulation
from copy import deepcopy
from sortedcontainers import SortedSet 
from functools import cmp_to_key
import matplotlib.pyplot as plt
import json
import animations
import sys
import generate_sun_like_figure
sys.setrecursionlimit(10**6)

#na razie zakladam ze nie bedzie punktow o rownych y, trzeba bedzie te jakies rotacje dorobic
savefig = False
EPS = 10**(-10)

def det_sarrus(A, B, C):
    return A.x*B.y + A.y*C.x + B.x*C.y - C.x*B.y - B.x*A.y - A.x*C.y

def loadFigure(dataName = "exportData.json"):
    '''
    Wczytuje określony wielokąt json w folderze "data". 
    '''
    pathToJson = Path(__file__).parent / "data" /dataName
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
            self.points[i].outgoingEdge = e1
            self.points[i + 1].outgoingEdge = e2
            ccwFace.append(e1)
        #edge case dla ostatniej krawedzi:       
        edge1, edge2 = HalfEdge(), HalfEdge()
        v1, v2 = self.points[-1], self.points[0]
        edge1.origin = v1
        edge2.origin = v2
        v1.outgoingEdge = edge1 #nie zmieniam outgoingEdge dla pierwszego wierzcholka bo juz ma dobra krawedz w CCW
        edge1.twin = edge2
        edge2.twin = edge1
        mesh.edges.append(edge1)
        mesh.edges.append(edge2)
        ccwFace.append(edge1)
        for i, edge in enumerate(mesh.edges):
            pt1,pt2 = edge.origin, edge.twin.origin
            edge.A = pt2.y - pt1.y
            edge.B = pt1.x - pt2.x
            edge.C = -(edge.A * pt1.x + edge.B * pt1.y)
        
        
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
        det = det_sarrus(A, B, C) 
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
        # self.visualizeClassification()
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
        self.polygonCopy = deepcopy(self.polygon)
        self.dictionary = self.initializeDict()
        self.fig, self.ax = plt.subplots()
        # xlim = (0,10)
        # ylim = (0,10)
        # self.ax.set_xlim(*xlim)
        # self.ax.set_ylim(*ylim)
        self.anim = animations.Animation(self.fig, self.ax, 'test')
        self.frame = 10
    def initializeDict(self):
        d = dict()
        for i,edge in enumerate(self.polygon.edges):
            d[edge] = self.polygonCopy.edges[i] 
        return d
    
    def addToTStructureAnim(self, halfedge):
        self.anim.addAction(self.frame, lambda line = halfedge: self.anim.changeLineColor('lightgreen', self.anim.find_line_index((line.origin.x, line.origin.y), (line.twin.origin.x, line.twin.origin.y))))
        self.frame += 2

    def deleteFromTStructureAnim(self, halfedge):
        self.anim.addAction(self.frame, lambda line = halfedge: self.anim.changeLineColor('blue', self.anim.find_line_index((line.origin.x, line.origin.y), (line.twin.origin.x, line.twin.origin.y))))
        self.frame += 2
        self.removeAsHelperAnim(halfedge.helper)
    def setAsHelperAnim(self, v):
        self.anim.addAction(self.frame, lambda pt = [v.x, v.y]: self.anim.addPoints([pt], color = 'cyan', s= 50))
        self.frame += 2

    def removeAsHelperAnim(self, v):
        if v is None:
            return
        self.anim.addAction(self.frame, lambda pt = [v.x, v.y]: self.anim.deletePoints(self.anim.find_scatter_index([pt])))
        self.frame += 2

    def addDiagonalAnim(self, start, end):
        self.anim.addAction(self.frame, lambda start = (start.x, start.y), end = (end.x, end.y): self.anim.addLine(start,end, color = 'orange', alpha = 0.5))
        self.frame += 2

    def handleStartVertex(self, v):
        toAdd = v.outgoingEdge
        self.removeAsHelperAnim(toAdd.helper)
        toAdd.helper = v
        self.setAsHelperAnim(v)
        
        self.T.add(toAdd)

        self.addToTStructureAnim(toAdd)

    def handleEndVertex(self, v):
        prevEdge = v.outgoingEdge.prev
        if prevEdge.helper is not None and prevEdge.helper.type == 'M':
            prevEdgeHelperD = self.polygonCopy.vertices[prevEdge.helper.id]
            vD = self.polygonCopy.vertices[v.id]
            self.addedDiags.add(self.polygonCopy.addDiagDiv(vD, prevEdgeHelperD)) 

            self.addDiagonalAnim(vD, prevEdgeHelperD)

        self.T.discard(prevEdge)

        self.deleteFromTStructureAnim(prevEdge)

    def handleSplitVertex(self, v):
        e = v.outgoingEdge

        self.removeAsHelperAnim(e.helper)
        e.helper = v
        self.setAsHelperAnim(v)

        self.T.add(e)

        self.addToTStructureAnim(e)
        
        left = self.T[self.T.index(e) - 1] #na lewo od e_{i}, na pewno nalezy do T bo dodalem dopiero
        leftHelperD = self.polygonCopy.vertices[left.helper.id]
        vD = self.polygonCopy.vertices[v.id]
        self.addedDiags.add(self.polygonCopy.addDiagDiv(vD, leftHelperD))

        self.addDiagonalAnim(vD, leftHelperD)

        self.removeAsHelperAnim(left.helper)
        left.helper = v
        self.setAsHelperAnim(v)

    def handleMergeVertex(self, v):
        second = False
        prevEdge = v.outgoingEdge.prev #prevedge tutaj na pewno nie jest diagonalną bo dopiero po zamieceniu v moze powstac do niego diagonalna. ALE sprawdzic nie zaszkodzi
        if prevEdge.helper is not None and prevEdge.helper.type == 'M':
            prevEdgeHelperD = self.polygonCopy.vertices[prevEdge.helper.id]
            vD = self.polygonCopy.vertices[v.id]
            self.addedDiags.add(self.polygonCopy.addDiagDiv(vD, prevEdgeHelperD))

            self.addDiagonalAnim(vD, prevEdgeHelperD)

            second = True 
        left = self.T[self.T.index(prevEdge) - 1] #na lewo od e_{i-1}, na pewno w T bo jest przy lewym przbu
        self.T.discard(prevEdge)

        self.deleteFromTStructureAnim(prevEdge)

        if left.helper.type == 'M':
            leftHelperD = self.polygonCopy.vertices[left.helper.id]
            vD = self.polygonCopy.vertices[v.id]
            self.addedDiags.add(self.polygonCopy.addDiagDiv(vD, leftHelperD, second))

            self.addDiagonalAnim(vD, leftHelperD)

        self.removeAsHelperAnim(left.helper)
        left.helper = v
        self.setAsHelperAnim(v)
    
    def handleRegularVertex(self, v):
        if v.type == 'RL': #intP po prawej    
            prevEdge = v.outgoingEdge.prev
            
            if prevEdge.helper is not None and prevEdge.helper.type == 'M': #prevEdge.helper is not None ROWNOZNACZNE Z: prevEdge nie zostal zakryty przekatna
                prevEdgeHelperD = self.polygonCopy.vertices[prevEdge.helper.id]
                vD = self.polygonCopy.vertices[v.id]                
                self.addedDiags.add(self.polygonCopy.addDiagDiv(vD, prevEdgeHelperD))

                self.addDiagonalAnim(vD, prevEdgeHelperD)

            self.T.discard(prevEdge)

            self.deleteFromTStructureAnim(prevEdge)

            v.outgoingEdge.helper = v
            self.T.add(v.outgoingEdge)

            self.addToTStructureAnim(v.outgoingEdge)

        else:
            probe = v.outgoingEdge 
            self.T.add(probe) #puszczamy sonde
            left = self.T[self.T.index(probe) - 1]
            self.T.discard(probe) #wyciagamy sonde
            if left.helper.type == 'M':
                leftHelperD = self.polygonCopy.vertices[left.helper.id]
                vD = self.polygonCopy.vertices[v.id]
                self.addedDiags.add(self.polygonCopy.addDiagDiv(vD, leftHelperD))

                self.addDiagonalAnim(vD, leftHelperD)

            self.removeAsHelperAnim(left.helper)
            left.helper = v
            self.setAsHelperAnim(v)

    def createAdjacentFaces(self, diag, visited):
        if not visited[diag]:
            newface = Face()
            diag.face = newface
            visited[diag] = True
            newface.outerEdge = diag
            p = diag.next
            while p != diag: 
                p.face = newface
                visited[p] = True
                p = p.next
            self.polygonCopy.faces.append(newface)
        if not visited[diag.twin]:
            newface = Face()
            diagTwin = diag.twin
            newface.outerEdge = diagTwin
            diagTwin.face = newface
            visited[diagTwin] = True
            q = diagTwin.next
            while q != diagTwin:
                q.face = newface
                visited[q] = True
                q = q.next
            self.polygonCopy.faces.append(newface)

    def updateFaces(self): 
        '''
        Updating faces has to take place after the algorithm of division, because we cant afford time cost of such an operation during the algorithm.
        updateFaces works similar to unconnected graph traversal. Polygon consists of unconnected cycles (faces). If we want to visit every edge, we have to start traversal at every diagonal and its twin.
        Warning: we shouldn't start at non-diagonal, because if we did so, we would traverse whole initial polygon and every initial edge would have the same face.
        Result: A polygon separated into faces. 
        '''
        if len(self.addedDiags) == 0:
            return
        self.polygonCopy.faces = [] #bo na razie miał jedną ścianę, ale zostanie ona usunięta, bo podzielono ją na kilka mniejszych.
        visited = dict()
        for diag in self.addedDiags:
            visited[diag] = False
            visited[diag.twin] = False
        for diag in self.addedDiags:
            if not visited[diag] or not visited[diag.twin]:
                self.createAdjacentFaces(diag, visited)

    def divide(self):
        # narysuj wielokąt bez niczego na razie 
        points = list(map(lambda vertex: (vertex.x, vertex.y), self.polygon.vertices))
        X, Y = zip(*points)
        for i in range (len(points)):
            self.ax.plot([X[i], X[(i + 1)%len(points)]], [Y[i], Y[(i + 1)%len(points)]], color = 'blue')
        #
    
        mergeOrSplits = [p for i, p in enumerate(points) if self.polygon.vertices[i].type in ('M', 'S')]
        X, Y = zip(*mergeOrSplits)
        # plt.scatter(X,Y, color = 'red', s =40)
        plt.scatter(X,Y, color = 'orange')
        points = [p for i, p in enumerate(points) if self.polygon.vertices[i].type not in ('M', 'S')]
        
        X, Y = zip(*points)
        self.ax.scatter(X, Y, color = 'orange')
        handle = {'M': self.handleMergeVertex, 'S': self.handleSplitVertex, 'I': self.handleStartVertex, 'RL': self.handleRegularVertex, 'RR': self.handleRegularVertex, 'E': self.handleEndVertex}

        while self.Q:
            event = self.Q.pop()
            self.anim.addAction(self.frame, lambda: self.anim.deleteLine(self.anim.find_axhline_index('red')))
            self.frame += 2
            self.anim.addAction(self.frame, lambda k = event.y: self.anim.addSweepLine(k, color = 'red'))
            self.frame += 2 
            print(event.type)
            HalfEdge.currY = event.y
            func = handle[event.type]
            func(event)
        self.anim.addAction(self.frame, lambda: self.anim.deleteLine(self.anim.find_axhline_index('red')))
        self.frame += 2
        self.updateFaces()
        # self.anim.draw(75)
        return self.polygonCopy

    def visualize(self):
        fig, ax = plt.subplots()
        xlim = (0,10)
        ylim = (0,10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        edgeSet = set()
        for edge in self.polygonCopy.edges:
            if edge not in edgeSet and edge.twin not in edgeSet:
                edgeSet.add(edge)
        pts = [(edge.origin, edge.twin.origin) for edge in edgeSet]
        for pair in pts:
            ax.plot((pair[0].x, pair[1].x), (pair[0].y, pair[1].y))
        plt.show()



class Triangulation:
    '''
    Klasa z algorytmami używanymi do triangulacji. Przyjmuje informacje o wielokącie i określoną tolerancję na zero. Zawiera: algorytm z wykładu (algorithm), funkcje animującą (algoAnimation),
    algorytm z wykładu, ale dodający przekątne zamiast trójkątów (algorithmDiag), wizualizacje wyniku algorithmDiag (VisualizeAlgoDiag) i wizualizacje wyniku algorithm (rysuje trójkąty triangulacji) (VisualizeResult)
    Pozostałe funkcje są pomocnicze.
    '''

    def __init__(self, mesh, fig, ax, anim, frame ,currFace,eps = 10**(-12) ):
        self.mesh = mesh
        self.eps = eps
        self.currFace = currFace
        self.vertices = self.mesh.extractFaceVertices(currFace)
        self.points = list(map(lambda vertex: (vertex.x, vertex.y), self.vertices))
        self.left = [True]*(len(self.vertices))
        self.addedDiags = []
        self.triangles = []
        self.fig, self.ax = fig, ax
        self.anim =anim
        self.frame = frame + 10
    def branches(self): 
        '''
        Dzieli na lewą i prawą gałąź - do prawej należy najniższy punkt, do lewej najwyższy.
        uwaga: leftidxs są od góry a rightidxs od dołu
        '''
        initialPointIdx, maxi = -1, -float('inf')
        endPointIdx, mini = -1, float('inf')
        for i, vert in enumerate(self.vertices):
            if maxi < vert.y:
                maxi = vert.y
                initialPointIdx = i
            if mini > vert.y:
                mini = vert.y
                endPointIdx = i
        i = endPointIdx
        leftIdxs = []
        rightIdxs = []
        while i != initialPointIdx:
            self.left[i] = False
            rightIdxs.append(i)
            i = (i + 1) % len(self.vertices)
            
        while i != endPointIdx:
            leftIdxs.append(i)
            i = (i + 1) % len(self.vertices)

        return leftIdxs, rightIdxs

    def orient(self, A, B, C):
        det = det_sarrus(A, B, C) 
        if det >= self.eps:
            return -1 #C na lewo od AB 
        elif det <= -self.eps:
            return 1 #C na prawo od AB
        return 0 #współliniowe


    def inside(self, idx1, idx2, idx3): 
        '''
        Sprawdza czy trójkąt stworzony z punktów o indeksach idx1, idx2, idx3 jest w środku wielokąta. Najpierw ustawia punkty w kolejności ccw, potem oblicza ich wyznacznik. Gdy wyznacznik > eps: kąt wew. tworzony 
        przez te 3 punkty jest < pi => trójkąt należy do wielokąta.
        '''
        order = sorted([idx1,idx2,idx3]) #posortuj po wystepowaniu w self.points: czyli ccw
        A, B, C = self.vertices[order[0]], self.vertices[order[1]], self.vertices[order[2]]
        det = det_sarrus(A, B, C)
        return det >= self.eps

    def mergeBranches(self):
        leftIdxs, rightIdxs = self.branches()
        idxs = [None]*(len(self.left))
        l,r = 0, len(rightIdxs) - 1
        for i in range (len(idxs)):
            if self.vertices[leftIdxs[l]].y >= self.vertices[rightIdxs[r]].y:
                idxs[i] = leftIdxs[l]
                l += 1
                if l == len(leftIdxs):
                    idxs[i + 1:] = rightIdxs[r::-1]
                    break
            else:
                idxs[i] = rightIdxs[r]
                r -= 1
                if r == -1:
                    idxs[i + 1:] = leftIdxs[l:]
                    break
        return idxs
    
    # def mergeBranches(self):
    #     leftIdxs, rightIdxs = self.branches()
    #     idxs = sorted(leftIdxs + rightIdxs, key = lambda idx: (-self.vertices[idx].y, self.vertices[idx].x))
    #     return idxs
    def addToStackAnim(self,idx):
        self.anim.addAction(self.frame, lambda p1 = self.points[idx]: self.anim.addPoints([p1], color = 'green'))
        self.frame += 1
    def popFromStackAnim(self, idx):
        self.anim.addAction(self.frame, lambda p1 = self.points[idx]: self.anim.deletePoints(self.anim.find_scatter_index([p1])))
        self.frame += 1
    def algorithmTriangles(self):
        '''
        GŁÓWNY ALGORYTM TRIANGULACJI. Tworzy trójkąty i zapisuje je do listy w kolejności ccw.
        '''
        
        N = len(self.vertices)
        self.branches()
        idxs = self.mergeBranches() #posortowana wzgledem y lista indeksów
        
        self.anim.addAction(self.frame, lambda points = self.points: self.anim.fill_polygon_ccw(points))
       
        self.frame += 1

        stack = [idxs[0], idxs[1]]
        for idx in range (2, N):
            if self.left[idxs[idx]] != self.left[stack[-1]]: #jeśli łańcuchy różne
                first = stack.pop()
                u = stack.pop()

                self.popFromStackAnim(first)
                self.popFromStackAnim(u)
                self.anim.addAction(self.frame, lambda p1 = self.points[idxs[idx]], p2 = self.points[u], p3 = self.points[first]: self.anim.addPoints([p1, p2, p3], color = 'red'))
                self.frame += 1
                self.anim.addAction(self.frame, lambda p1 = self.points[idxs[idx]], p2 = self.points[u], p3 = self.points[first]: self.anim.addTriangleLines(p1, p2, p3, color = 'red'))
                self.frame += 1
                while stack:
                    u,v = stack.pop(), u

                    self.popFromStackAnim(u)
            
                    self.anim.addAction(self.frame, lambda p1 = self.points[idxs[idx]], p2 = self.points[u], p3 = self.points[v]: self.anim.addPoints([p1, p2, p3], color = 'red'))
                    self.frame += 1
                    self.anim.addAction(self.frame, lambda p1 = self.points[idxs[idx]], p2 = self.points[u], p3 = self.points[v]: self.anim.addTriangleLines(p1,p2,p3, color = 'red'))
                    self.frame += 1
                stack.append(first)
                stack.append(idxs[idx])

                self.addToStackAnim(first)
                self.addToStackAnim(idxs[idx])
                
            else:
                flag = False
                v = stack.pop() #ten po stronie idxsa
                u = stack.pop() 
                self.popFromStackAnim(v)
                
                self.popFromStackAnim(u)
                
                while self.inside(idxs[idx], v, u):
                    self.anim.addAction(self.frame, lambda p1 = self.points[idxs[idx]], p2 = self.points[u], p3 = self.points[v]: self.anim.addPoints([p1, p2, p3], color = 'red'))
                    self.frame += 1
                    self.anim.addAction(self.frame, lambda p1 = self.points[idxs[idx]], p2 = self.points[u], p3 = self.points[v]: self.anim.addTriangleLines(p1,p2,p3, color = 'red'))
                    self.frame += 1
                    if not stack: #jesli skonczyl sie stack to nic nadmiarowego nie usunelismy
                        flag = True
                        break
                    u, v = stack.pop(), u

                    self.popFromStackAnim(u)
                stack.append(u) 
                self.addToStackAnim(u)

                if not flag: 
                    stack.append(v) #jesli w ostatniej iteracji udało się dodać trojkat, nie dodawaj ostatniego elementu stosu, bo właśnie go "uwieziliśmy" odcinkiem i->u. Jesli sie nie udalo to dodaj zeby wrocic do stanu z przed usuwania
                    
                    self.addToStackAnim(v)
                
                stack.append(idxs[idx])
                self.addToStackAnim(idxs[idx])
                
        self.frame += 1
        while stack:
            u = stack.pop()
            self.popFromStackAnim(u)
            
            self.anim.addAction(self.frame, lambda p1 = self.points[u]: self.anim.addPoints([p1], color = 'red'))
            self.frame += 1
        


def triangulate(points):
    '''
    param: list of points  
    return: list of tuples: [(i,j,k): i,j,k indices of points from points in CCW order]
    '''
    prepare = Structures(points)
    division = Division(prepare.prepareHalfEdgeMesh(), prepare.prepareEvents(), prepare.prepareSweep())
    division.divide()
    allTriangles = []
    prevframe = division.frame + 1
    for face in division.polygonCopy.faces:
        trian = Triangulation(division.polygonCopy, division.fig, division.ax, division.anim, prevframe, currFace=face)
        if len(trian.vertices) == 3:
            allTriangles += [(trian.vertices[0].id, trian.vertices[1].id, trian.vertices[2].id)]
            trian.anim.addAction(trian.frame, lambda p1 = trian.points[0], p2 = trian.points[1], p3 = trian.points[2]: trian.anim.addPoints([p1, p2, p3], color = 'red'))
            trian.frame += 1
            trian.anim.addAction(trian.frame, lambda p1 = trian.points[0], p2 = trian.points[1], p3 = trian.points[2]: trian.anim.addTriangleLines(p1,p2,p3, color = 'red'))
            trian.frame += 1
            trian.anim.addAction(trian.frame, lambda points = trian.points: trian.anim.fill_polygon_ccw(points))
            trian.frame += 1
        else:
            trian.algorithmTriangles()
            #zamieniaj indeksy wewnetrzne algorytmu na indeksy globalne (id z Vertex) za pomocą listy trian.vertices gdzie indeks globalny dla i = trian.vertices[i].id 
            allTriangles += list(map(lambda innerIdxTuple: tuple(map(lambda innerIdx: trian.vertices[innerIdx].id, innerIdxTuple)), trian.triangles))
        prevframe = trian.frame + 1
    print(len(allTriangles)) 
    division.anim.draw(150)
    return allTriangles
                
if __name__ == "__main__":
    figure = loadFigure("exportData.json")
    points = figure["points"]
    points = generate_sun_like_figure.generate(5,10,10)
    triangulate(points)
    

