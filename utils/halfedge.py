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
