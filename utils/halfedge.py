class Vertex:
    def __init__(self, x, y, id = None):
        self.x = x  # Współrzędna x wierzchołka
        self.y = y  # Współrzędna y wierzchołka
        self.id = id
        self.outgoingEdge = None  # Wskaźnik na jedną z wychodzących krawędzi (HalfEdge)
        self.type = None #typ punktu (terminal itd.)
   
    # def __eq__(self, other):
    #     return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"Vertex({self.x}, {self.y})"

def orient(A, B, C):
    det = det_sarrus(A, B, C) 
    if det > 0:
        return -1 #C na lewo od AB 
    elif det < 0:
        return 1 #C na prawo od AB
    return 0 #współliniowe
def det_sarrus(A, B, C):
    return A.x*B.y + A.y*C.x + B.x*C.y - C.x*B.y - B.x*A.y - A.x*C.y
class HalfEdge:
    currY = -1
    def __init__(self):
        self.origin = None  # Wskaźnik na początkowy wierzchołek (Vertex)
        self.twin = None  # Wskaźnik na drugą połowę tej samej krawędzi (HalfEdge)
        self.next = None  # Wskaźnik na następną krawędź w tej samej ścianie (HalfEdge)
        self.prev = None  # Wskaźnik na poprzednią krawędź w tej samej ścianie (HalfEdge)
        self.face = None  # Wskaźnik na ścianę, do której należy krawędź (Face)
        self.A = None #Ax + By + C = 0
        self.B = None 
        self.C = None 
        self.helper = None
        self.CCW = True
        
    def currX(self):
        if self.A == 0:
            return self.origin.x #zwroc x poczatkowy - pozwoli na rozroznienie miedzy segmentami na tym samym y=k
        else:
            return (-self.B*HalfEdge.currY - self.C)/self.A  
    
    def __repr__(self):
        return f"HalfEdge(origin={self.origin})"

    def __eq__(self, other):
        if not isinstance(other, HalfEdge):
            return False
        return self.origin.id == other.origin.id and self.twin.origin.id == other.twin.origin.id

    def __hash__(self):
        return hash((self.origin.id))
        
class Face:
    def __init__(self):
        self.outerEdge = None  # Wskaźnik na jedną z otaczających krawędzi (HalfEdge)

    def __repr__(self):
        return f"Face(outer_edge={self.outerEdge})"


class Graph:
    def __init__(self, vertices, edges):
        self.vertices = vertices  # Lista wierzchołków (Vertex objects)
        self.edges = edges        # Lista krawędzi (tuples: (Vertex1, Vertex2))
        self.G = self.createAdjacencyList()

    def createAdjacencyList(self):
        adjacency_list = [[] for _ in range(len(self.vertices))]

        for edge in self.edges:
            v1, v2 = edge  # Rozpakowanie krawędzi (Vertex1, Vertex2)
            adjacency_list[v1.id].append(v2.id)  
            adjacency_list[v2.id].append(v1.id)  
        
        return adjacency_list
    
    def addDiagDiv(self, v1ID, v2ID):
        self.G[v1ID].append(v2ID)
        self.G[v2ID].append(v1ID)
       
    



class HalfEdgeMesh:
    def __init__(self, vertices = [], faces = [], edges = []):
        self.vertices = vertices  # Lista wszystkich wierzchołków
        self.edges = edges  # Lista wszystkich krawędzi (HalfEdge)
        self.faces = faces  # Lista wszystkich ścian (Face)
    def copy_coefficients(self, original_edge, new_edge):
        """
        Funkcja kopiująca współczynniki A, B, C z oryginalnej krawędzi do nowej krawędzi.
        """
        new_edge.A = original_edge.A
        new_edge.B = original_edge.B
        new_edge.C = original_edge.C
    def deepcopy(self):
        # Tworzymy nowe wierzchołki, kopiując dane każdego wierzchołka
        new_vertices = [Vertex(v.x, v.y, v.id) for v in self.vertices]

        # Mapowanie krawędzi (oryginał -> nowa krawędź)
        edge_map = {}
        new_edges = [None] * len(self.edges)  # Tworzymy listę krawędzi o takiej samej długości
        edge = self.faces[0].outerEdge
        start = edge
        i = 0
        while True:
            # Tworzymy nową krawędź
            if edge not in edge_map:
                new_edge = HalfEdge()
            else:
                new_edge = edge_map[edge]
            # Nowy wierzchołek dla tej krawędzi
            new_edge.origin = new_vertices[edge.origin.id]  # Powiązanie z nowym wierzchołkiem
            self.copy_coefficients(edge, new_edge)
            # Zachowujemy krawędź w odpowiednim miejscu w nowej liście
            new_edges[i] = new_edge
            edge_map[edge] = new_edge  # Mapujemy oryginalną krawędź na nową krawędź

            # Krawędź bliźniacza
            if edge.twin:
                if edge.twin not in edge_map:  # Tylko kopiujemy, jeśli jeszcze nie kopiowaliśmy
                    new_edge.twin = HalfEdge()
                    edge_map[edge.twin] = new_edge.twin
                new_edge.twin = edge_map[edge.twin]
            # Krawędź następna
            if edge.next:
                if edge.next not in edge_map:
                    new_edge.next = HalfEdge()
                    edge_map[edge.next] = new_edge.next
                new_edge.next = edge_map[edge.next]
            # Krawędź poprzednia
            if edge.prev:
                if edge.prev not in edge_map:
                    new_edge.prev = HalfEdge()
                    edge_map[edge.prev] = new_edge.prev
                new_edge.prev = edge_map[edge.prev]
                
            edge = edge.next
            if edge == start:
                break
            i += 1
        # Przypisujemy nowe krawędzie jako `outgoingEdge` w odpowiednich wierzchołkach
        for v in self.vertices:
            # Ustawiamy krawędź wychodzącą w nowym wierzchołku
            new_vertices[v.id].outgoingEdge = edge_map.get(v.outgoingEdge)
            new_vertices[v.id].type = v.type
        # Tworzymy nowe ściany, kopiując odpowiednie krawędzie
        new_faces = []
        for face in self.faces:
            new_face = Face()
            if face.outerEdge:
                new_face.outerEdge = edge_map.get(face.outerEdge)
            new_faces.append(new_face)

        # Zwracamy nową, głęboko skopiowaną siatkę HalfEdge
        new_mesh = HalfEdgeMesh(vertices=new_vertices, faces=new_faces, edges=new_edges)
        return new_mesh
    def addVertex(self, x, y):
        vertex = Vertex(x, y)
        self.vertices.append(vertex)
        return vertex

    def addEdge(self, v1, v2):
        """
        Dodaje krawędź między wierzchołkami v1 i v2.
        Tworzy dwie HalfEdge: jedną dla kierunku v1 -> v2 i drugą dla v2 -> v1.
        """
        edge1 = HalfEdge()
        edge2 = HalfEdge()

        # Powiązanie krawędzi z wierzchołkami
        edge1.origin = v1
        edge2.origin = v2
        # v1.outgoingEdge = edge1  #to nie moze sie dziac bo w handleach potrzebuje zeby v.outgoingi byly nieprzekatnymi
        # v2.outgoingEdge = edge2

        # Powiązanie krawędzi jako twin
        edge1.twin = edge2
        edge2.twin = edge1

        # Dodanie krawędzi do listy
        self.edges.append(edge1)
        self.edges.append(edge2)

        return edge1, edge2

    def addFace(self, edgeCycle):
        """
        Tworzy ścianę (Face) otoczoną przez cykl krawędzi edge_cycle.
        edge_cycle: lista obiektów HalfEdge tworzących zamknięty obwód.
        """
        face = Face()
        self.faces.append(face)

        # Powiązanie krawędzi z nową ścianą
        for i, edge in enumerate(edgeCycle):
            edge.face = face
            edge.next = edgeCycle[(i + 1) % len(edgeCycle)]  # Następna krawędź w cyklu
            edge.prev = edgeCycle[(i - 1) % len(edgeCycle)]  # Poprzednia krawędź w cyklu
        # Powiązanie ściany z jedną z jej krawędzi
        face.outerEdge = edgeCycle[0]

        return face
    
    def CCW(self, A, BC):
        B, C = BC.origin, BC.twin.origin
        return  orient(A, B, C) == -1 #jesli C na lewo od AB to BC jest CCW 
            
        
    def addDiagDiv(self, v1, v2, second = False):
        '''
        Dodaje przekątną między punktami, rozdziela face na dwie nowe ściany.
        Lemat: Zanim dwa wierzchołki się połączą, mają maksymalnie jedną wspólną ścianę. 
        NIE MODYFIKUJE FACES, bo zlozoność bylaby liniowa
        modyfikuje tylko krawedzie w sumie, da sie przeiterowac po wszystkich krawedziach majac tylko to (i chyba 
        oznaczyc sciany tez)
        '''
        #v1
        if v1.type == 'M':
            if second:
                v1PrevEdge = v1.outgoingEdge.twin
                v1NextEdge = v1.outgoingEdge.twin.next
            else:
                v1PrevEdge = v1.outgoingEdge.prev
                v1NextEdge = v1.outgoingEdge
        else:
            v1PrevEdge = v1.outgoingEdge.prev
            v1NextEdge = v1.outgoingEdge
        
        #v2
        if self.CCW(v1, v2.outgoingEdge):
            v2PrevEdge = v2.outgoingEdge.prev
            v2NextEdge = v2.outgoingEdge 
        else:
            v2PrevEdge = v2.outgoingEdge.twin
            v2NextEdge = v2.outgoingEdge.twin.next
       

        new, newTwin = self.addEdge(v1,v2)
        
        pt1,pt2 = new.origin, new.twin.origin
        new.A = pt2.y - pt1.y
        new.B = pt1.x - pt2.x
        new.C = -(new.A * pt1.x + new.B * pt1.y)
        newTwin.A, newTwin.B, newTwin.C = new.A, new.B, new.C

        new.prev = v1PrevEdge
        new.next = v2NextEdge
        newTwin.prev = v2PrevEdge
        newTwin.next = v1NextEdge

        v1PrevEdge.next = new
        v2NextEdge.prev = new

        v1NextEdge.prev = newTwin
        v2PrevEdge.next = newTwin

        v1.outgoingEdge = new
        if v2.type != "S":
            v2.outgoingEdge = newTwin
        return new
    # def addDiagDiv(self, v1, v2, second = False):
    #     '''
    #     Dodaje przekątną między punktami, rozdziela face na dwie nowe ściany.
    #     Lemat: Zanim dwa wierzchołki się połączą, mają maksymalnie jedną wspólną ścianę. 
    #     NIE MODYFIKUJE FACES, bo zlozoność bylaby liniowa
    #     modyfikuje tylko krawedzie w sumie, da sie przeiterowac po wszystkich krawedziach majac tylko to (i chyba 
    #     oznaczyc sciany tez)
    #     '''
    #     # if v2.type == 'S':
    #     #     v1PrevEdge = v1.outgoingEdge.prev
    #     #     v1NextEdge = v1.outgoingEdge
    #     #     if  v1.x > v2.x: #jestesmy po prawej, intP po prawej od outgoing, trzeba odwrocic zeby bylo CCW
    #     #         v2PrevEdge = v2.outgoingEdge.twin
    #     #         v2NextEdge = v2.outgoingEdge.twin.next
    #     #     else:
    #     #         v2PrevEdge = v2.outgoingEdge.prev
    #     #         v2NextEdge = v2.outgoingEdge
    #     # else:
    #     #     if not v1.outgoingEdge.CCW:
    #     #         v1PrevEdge = v1.outgoingEdge.twin
    #     #         v1NextEdge = v1.outgoingEdge.twin.next
    #     #     else:
    #     #         v1PrevEdge = v1.outgoingEdge.prev
    #     #         v1NextEdge = v1.outgoingEdge
    #     #     if not v2.outgoingEdge.CCW:
    #     #         v2PrevEdge = v2.outgoingEdge.twin
    #     #         v2NextEdge = v2.outgoingEdge.twin.next
    #     #     else:
    #     #         v2PrevEdge = v2.outgoingEdge.prev
    #     #         v2NextEdge = v2.outgoingEdge
     
    #     new, newTwin = self.addEdge(v1,v2)
        
    #     pt1,pt2 = new.origin, new.twin.origin
    #     new.A = pt2.y - pt1.y
    #     new.B = pt1.x - pt2.x
    #     new.C = -(new.A * pt1.x + new.B * pt1.y)
    #     newTwin.A, newTwin.B, newTwin.C = new.A, new.B, new.C

    #     new.prev = v1PrevEdge
    #     new.next = v2NextEdge
    #     newTwin.prev = v2PrevEdge
    #     newTwin.next = v1NextEdge

    #     v1PrevEdge.next = new
    #     v2NextEdge.prev = new

    #     v1NextEdge.prev = newTwin
    #     v2PrevEdge.next = newTwin

    #     v1.outgoingEdge = new
    #     if v2.type != "S":
    #         v2.outgoingEdge = newTwin
    #     #     if v1.type == 'M' or v1.type == 'RL' or v1.type == 'RR':
    #     #         if v1.x < v2.x:
    #     #             v1.outgoingEdge.CCW = False
    #     #         else:
    #     #             v2.outgoingEdge.CCW = False
        # return new

    # def better(self, e1, e2):
    #     assert(e1.origin == e2.origin)
    #     A = e1.origin
    #     B = e1.twin.origin
    #     C = e2.twin.origin
    #     if orient(A,B,C) == -1:
    #         #bierzemy C
    #         return e2
    #     return e1
    
    # def update_coef(self, new, newTwin):
    #     pt1,pt2 = new.origin, new.twin.origin
    #     new.A = pt2.y - pt1.y
    #     new.B = pt1.x - pt2.x
    #     new.C = -(new.A * pt1.x + new.B * pt1.y)
    #     newTwin.A, newTwin.B, newTwin.C = new.A, new.B, new.C

    # def addDiagDiv(self, v1, v2):
    #     v1v2, v2v1 = self.addEdge(v1,v2)
    #     # if v1.type == 'M':
    #     if v1.outgoingEdge.twin.next is not None:
    #         v1better = self.better(v1.outgoingEdge, v1.outgoingEdge.twin.next)
    #     else:
    #         v1better = v1.outgoingEdge
    #     if v2.outgoingEdge.twin.next is not None:
    #         v2better = self.better(v2.outgoingEdge, v2.outgoingEdge.twin.next)
    #     else:
    #         v2better = v2.outgoingEdge
    #     v2v1.next = v1better
    #     v2v1.prev = v2better.prev
    #     v1v2.next = v2better
    #     v1v2.prev = v1better.prev
    #     v2v1.next.prev = v2v1
    #     v2v1.prev.next = v2v1
    #     v1v2.next.prev = v1v2
    #     v1v2.prev.next = v1v2
    #     v1.outgoingEdge = v1v2
    #     if v2.type != 'S':
    #         v2.outgoingEdge = v2v1
    #     self.update_coef(v1v2,v2v1)
    #     return v1v2


    def extractFaceVertices(self, face):
        vertices = []
        start = face.outerEdge
        vertices.append(start.origin)
        p = start.next
        while p != start:
            vertices.append(p.origin)
            p = p.next
        return vertices
        
    def __repr__(self):
        return f"HalfEdgeMesh(vertices={len(self.vertices)}, edges={len(self.edges)}, faces={len(self.faces)})"


# Przykład użycia
if __name__ == "__main__":
    # Tworzymy siatkę
    mesh = HalfEdgeMesh()

    # Dodajemy wierzchołki
    v1 = mesh.addVertex(0, 0)
    v2 = mesh.addVertex(1, 0)
    v3 = mesh.addVertex(1, 1)
    v4 = mesh.addVertex(0, 1)

    # Dodajemy krawędzie
    e1, e2 = mesh.addEdge(v1, v2)
    e3, e4 = mesh.addEdge(v2, v3)
    e5, e6 = mesh.addEdge(v3, v4)
    e7, e8 = mesh.addEdge(v4, v1)

    # Tworzymy ścianę (Face)
    face = mesh.addFace([e1, e3, e5, e7])

    print(mesh)
    print("Vertices:", mesh.vertices)
    print("Edges:", mesh.edges)
    print("Faces:", mesh.faces)
