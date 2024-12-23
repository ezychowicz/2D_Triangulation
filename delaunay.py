import random


class Vector:
  def __init__(self, x, y):
    self.x = x
    self.y = y    

def sgn(value):
  if value < 0:
    return -1
  elif value > 0:
    return 1
  else:
    return 0

def orientation(v1, v2, v3):
  det = (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x)
  return sgn(det)

def circleTest(v1, v2, v3, v4):
  det = (
    (v1.x**2 + v1.y**2) * (v2.x - v4.x) +
    (v2.x**2 + v2.y**2) * (v4.x - v1.x) +
    (v4.x**2 + v4.y**2) * (v1.x - v2.x) -
    (v1.x**2 + v1.y**2) * (v3.x - v4.x) -
    (v3.x**2 + v3.y**2) * (v4.x - v1.x) -
    (v4.x**2 + v4.y**2) * (v1.x - v3.x)
  )
  return sgn(det)

class Face:
  def __init__(self, edge):    
    self.vertex1 = edge.vertex
    self.vertex2 = edge.next.vertex
    self.vertex3 = edge.next.next.vertex
    self.children = []

  def addChild(self, face):
    self.children.append(face)

  def contains(self, point, vertices):    
    return orientation(vertices[self.vertex1], vertices[self.vertex2], point) == orientation(vertices[self.vertex2], vertices[self.vertex3], point) == orientation(vertices[self.vertex3], vertices[self.vertex1], point) == -1

class HalfEdge:
  def __init__(self, twin, next, vertex, face = None):
    self.twin = twin
    self.next = next
    self.vertex = vertex
    self.face = face

  def flip(self):
    prev1 = self.next.next
    next1 = self.next
    prev2 = self.twin.next.next
    next2 = self.twin.next

    prev1.next = next2
    prev2.next = next1

    newEdge1 = HalfEdge(None, prev2, prev1.vertex)
    newEdge2 = HalfEdge(newEdge1, prev1, prev2.vertex)
    newEdge1.twin = newEdge2

    next1.next = newEdge1
    next2.next = newEdge2

    return newEdge1  


class Mesh:

  def __init__(self, vertices):
    self.vertices = vertices
    self.faces = []

    e3 = HalfEdge(None, None, Vector(-1000, -1000))
    e2 = HalfEdge(None, e3, Vector(1000, -1000))
    e1 = HalfEdge(None, e2, Vector(0, 1000))
    e3.next = e1

    self.faces.append(Face(e1))
    self.root = 0

  def flipEdge(self, edge):
    face1 = edge.face
    face2 = edge.twin.face

    newEdge = edge.flip()

    self.faces.append(Face(newEdge))
    newEdge.face = len(self.faces) - 1

    self.faces.append(Face(newEdge.twin))
    newEdge.twin.face = len(self.faces) - 1

    face1.addChild(newEdge.face)
    face1.addChild(newEdge.twin.face)
    face2.addChild(newEdge.face)
    face2.addChild(newEdge.twin.face)


  def isEdgeLegal(self, edge):
    return circleTest(self.vertices[edge.vertex], self.vertices[edge.next.vertex], self.vertices[edge.next.next.vertex], self.vertices[edge.twin.next.vertex]) < 0      

  def legalizeEdge(self, edge):
    if not self.isEdgeLegal(edge):
      e1 = edge.next
      e2 = e1.next

      self.flipEdge(edge)

      if e1.twin:
        self.legalizeEdge(e1.twin)
      if e2.twin:
        self.legalizeEdge(e2.twin)

  def addVertexToFaceAndLegalize(self, face, vertexIndex):
    pass

  def contains(self, face, vertex):
    return self.faces[face].contains(self.vertices[vertex], self.vertices)

  def locate(self, vertex):
    face = self.root
    while self.faces[face].children:
      for child in self.faces[face].children:
        if self.contains(child, vertex):
          face = child
          break
    return face

  def toTriangleList(self):

    ans = []

    queue = []
    l = 0

    queue.push(self.root)

    visited = [False] * len(self.faces)

    while l < len(queue):
      face = queue[l]
      l += 1
      visited[face] = True

      ans.append((self.faces[face].v1, self.faces[face].v2, self.faces[face].v3))

      for child in self.faces[face]:
        if not visited[child]:
          queue.append(child)

    return ans


def delaunayNaive(points):
  mesh = Mesh()

  for p in points:
    for face in mesh.faces:
      if mesh.contains(face, p):
        mesh.addVertexToFaceAndLegalize(face, p)

  return mesh.toTriangleList()

def delaunay(points):
  mesh = Mesh()

  indices = [i for i in range(len(points))]
  random.shuffle(indices)

  for i in indices:
    p = points[i]
    face = mesh.locate(p)
    mesh.addVertexToFaceAndLegalize(face, p)

  return mesh.toTriangleList()

def constrained(points):
  pass

# sources:
# Some great thing about non-constrained delaunay https://ianthehenry.com/posts/delaunay/#fnref:2
# constrained but in type script https://tchayen.com/constrained-delaunay-triangulation-from-a-paper
# there is also chapter in 'the' book
# https://cp-algorithms.com/geometry/delaunay.html
# it seems there are few different options divida and conqure or online idk which one to use
# http://www.geom.uiuc.edu/~samuelp/del_project.html
# divide and conquer seems great dunno about constraining tho