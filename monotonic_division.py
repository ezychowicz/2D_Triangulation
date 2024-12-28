from pathlib import Path
from utils.halfedge import HalfEdge, HalfEdgeMesh, Vertex, Face
from utils import interactive_figure

from sortedcontainers import SortedSet 
from functools import cmp_to_key
import matplotlib.pyplot as plt
import json

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
            if edge.origin.y == edge.twin.origin.y: #tymczasowe rozw. bo jak bedzie duzo punktów współliniowych i to jeszcze niesasiadujacych to chyba nie zadziala
                if edge.origin.x < edge.twin.origin.x:
                    edge.k = (edge.origin.x - edge.twin.origin.x)/((edge.origin.y + EPS) - edge.twin.origin.y)
                    edge.l = edge.origin.x - edge.k*(edge.origin.y + EPS)
                else:
                    edge.k = (edge.twin.origin.x - edge.origin.x)/((edge.twin.origin.y + EPS) - edge.origin.y)
                    edge.l = edge.twin.origin.x - edge.k*(edge.twin.origin.y + EPS) 
            else:
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
        if  prevEdge.helper.type == 'M':
            self.addedDiags.add(self.polygon.addDiagDiv(v, prevEdge.helper))
        self.T.discard(prevEdge)
        
    def handleSplitVertex(self, v):
        e = v.outgoingEdge
        e.helper = v
        self.T.add(e)
        left = self.T[self.T.index(e) - 1] #na lewo od e_{i}, na pewno nalezy do T bo dodalem dopiero
        self.addedDiags.add(self.polygon.addDiagDiv(v, left.helper))
        left.helper = v
        print(id(left))

    def handleMergeVertex(self, v):
        second = False
        prevEdge = v.outgoingEdge.prev #prevedge tutaj na pewno nie jest diagonalną bo dopiero po zamieceniu v moze powstac do niego diagonalna. ALE sprawdzic nie zaszkodzi
        if  prevEdge.helper.type == 'M':
            self.addedDiags.add(self.polygon.addDiagDiv(v, prevEdge.helper, ))
            second = True 
        left = self.T[self.T.index(prevEdge) - 1] #na lewo od e_{i-1}, na pewno w T bo jest przy lewym przbu
        self.T.discard(prevEdge)
        if left.helper.type == 'M':
            self.addedDiags.add(self.polygon.addDiagDiv(v, left.helper, second))
        left.helper = v
        
    
    def handleRegularVertex(self, v):
        if v.type == 'RL': #intP po prawej    
            prevEdge = v.outgoingEdge.prev
            if  prevEdge.helper.type == 'M': #prevEdge.helper is not None ROWNOZNACZNE Z: prevEdge nie zostal zakryty przekatna
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
            q = diagTwin.next
            while q != diagTwin:
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



class Triangulation:
    '''
    Klasa z algorytmami używanymi do triangulacji. Przyjmuje informacje o wielokącie i określoną tolerancję na zero. Zawiera: algorytm z wykładu (algorithm), funkcje animującą (algoAnimation),
    algorytm z wykładu, ale dodający przekątne zamiast trójkątów (algorithmDiag), wizualizacje wyniku algorithmDiag (VisualizeAlgoDiag) i wizualizacje wyniku algorithm (rysuje trójkąty triangulacji) (VisualizeResult)
    Pozostałe funkcje są pomocnicze.
    '''

    def __init__(self, mesh, currFace, eps = 10**(-12)):
        self.mesh = mesh
        self.eps = eps
        self.currFace = currFace
        self.vertices = self.mesh.extractFaceVertices(currFace)
        self.left = [True]*(len(self.vertices))
        self.addedDiags = []
    def branches(self): 
        '''
        Dzieli na lewą i prawą gałąź - do prawej należy najniższy punkt, do lewej najwyższy.
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
    
    # def updateMesh(self, idx1, idx2): #dodaje przekatna i aktualizuje faces
    #     if idx1 > idx2: #chcemy posortowane bo indeksy sa ccw, a dodajac przekatna chcemy to zrobic w dobrym kierunku
    #         idx1, idx2 = idx2, idx1
    #     diag = self.mesh.addDiagDiv(self.vertices[idx1], self.vertices[idx2])
    #     newFace1 = Face()
    #     # diag = self.vertices[idx1].outgoingEdge
    #     newFace1.outerEdge = diag 
    #     newFace2 = Face() 
    #     # diagTwin = self.vertices[idx2].outgoingEdge
    #     diagTwin = diag.twin
    #     newFace2.outerEdge = diagTwin
    #     for i in range (3):
    #         diag.face = newFace1
    #         diag = diag.next

    #         diagTwin.face = newFace2
    #         diagTwin = diagTwin.next
    #     self.mesh.faces.append(newFace1)
    #     self.mesh.faces.append(newFace2)

    def algorithm(self):
        '''
        GŁÓWNY ALGORYTM TRIANGULACJI. Tworzy trójkąty i zapisuje je do listy w kolejności ccw.
        '''
        N = len(self.vertices)
        self.branches()
        idxs = self.mergeBranches() #posortowana wzgledem y lista indeksów
        
        stack = [idxs[0], idxs[1]]
        for idx in range (2, N):
            if self.left[idxs[idx]] != self.left[stack[-1]]: #jeśli łańcuchy różne
                first = stack.pop()
                u = stack.pop()
                if abs(idxs[idx] - first) != N - 1 and abs(idxs[idx] - first) != 1:
                    # self.addedDiags.append((idxs[idx], first))
                    self.addedDiags.append((self.vertices[idxs[idx]], self.vertices[first]))
                    # self.updateMesh(idxs[idx], first)
                while stack:
                    u,v = stack.pop(), u
                    if abs(idxs[idx] - v) != N - 1 and abs(idxs[idx] - v) != 1:  #żeby nie dodawał gdy to jest bok
                        # self.addedDiags.append((idxs[idx], v))
                        self.addedDiags.append((self.vertices[idxs[idx]], self.vertices[v]))
                        # self.updateMesh(idxs[idx], v)
                stack.append(first)
                stack.append(idxs[idx])
            else:
                flag = False
                v = stack.pop() 
                u = stack.pop() 
                while self.inside(idxs[idx], v, u): 
                    if abs(idxs[idx] - u) != N - 1 and abs(idxs[idx] - u) != 1:
                        # self.addedDiags.append((idxs[idx], u))
                        self.addedDiags.append((self.vertices[idxs[idx]], self.vertices[u]))
                        # self.updateMesh(idxs[idx], u)
                    if not stack: #jesli skonczyl sie stack to nic nadmiarowego nie usunelismy
                        flag = True
                        break
                    u, v = stack.pop(), u 

                stack.append(u) 
                if not flag: 
                    stack.append(v) 
                stack.append(idxs[idx])

def visualize(mesh, addedDiags):
    fig, ax = plt.subplots()
    xlim = (0,10)
    ylim = (0,10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    edgeSet = set()
    for edge in mesh.edges:
        if edge not in edgeSet and edge.twin not in edgeSet:
            edgeSet.add(edge)
    pts = [(edge.origin, edge.twin.origin) for edge in edgeSet]
    pts += addedDiags
    for pair in pts:
        ax.plot((pair[0].x, pair[1].x), (pair[0].y, pair[1].y))
    plt.show()


if __name__ == "__main__":
    figure = loadFigure("mirroredmountains.json")
    points = figure["points"]
    X, Y = zip(*points)
    for i in range (len(points)):
        plt.plot([X[i], X[(i + 1)%len(points)]], [Y[i], Y[(i + 1)%len(points)]])
    plt.show()
    prepare = Structures(points)
    division = Division(prepare.prepareHalfEdgeMesh(), prepare.prepareEvents(), prepare.prepareSweep())
    division.divide()
    division.visualize()
    facesIdxToRemove = set()
    allDiags = []
    for i, face in enumerate(division.polygon.faces[::]):
        print(i, face)
        trian = Triangulation(division.polygon, currFace=face)
        if len(trian.vertices) == 3:
            continue
        else:
            trian.algorithm()
            allDiags += trian.addedDiags
            division.polygon.faces.remove(face)
            print(division.polygon)
            print(division.polygon.faces)
    visualize(division.polygon, allDiags)

    division.polygon.faces = list(filter(lambda faceIdx: faceIdx not in facesIdxToRemove, division.polygon.faces))
    