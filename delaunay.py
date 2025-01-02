import random
import math
import matplotlib.pyplot as plt
from collections import deque

import animations

class Vector:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __repr__(self):
    return f"Vector({self.x:.2f}, {self.y:.2f})"

def sgn(value):
  if value < 0:
    return -1
  elif value > 0:
    return 1
  else:
    return 0

def orientationDet(v1, v2, v3):
  return (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x)

def orientation(v1, v2, v3):
  det = orientationDet(v1, v2, v3)
  return -sgn(det)

# checks if point p lies on segment a->b, uses Eps for tolerance
def onsegment(a, b, p, Eps):
  det = orientationDet(a, b, p)
  if abs(det) > Eps:
    return False

  return p.x >= min(a.x, b.x) and p.x <= max(a.x, b.x) and p.y >= min(a.y, b.y) and p.y <= max(a.y, b.y)

# returns:
# -1 when circumcircle (a, b c) doesn't contain p
#  0 when p lies on circumcircle (a, b c)
#  1 when p lies inside circumcircle (a, b c)
def circumcircleTest(a, b, c, p):
  import numpy as np

  matrix = np.array([
      [a.x, a.y, a.x**2 + a.y**2, 1],
      [b.x, b.y, b.x**2 + b.y**2, 1],
      [c.x, c.y, c.x**2 + c.y**2, 1],
      [p.x, p.y, p.x**2 + p.y**2, 1],
  ])

  return sgn(np.linalg.det(matrix))

# checks if segment a->b and c-> intersect
def segmentsIntersect(a, b, c, d):
  oa = orientation(c, d, a)
  ob = orientation(c, d, b)
  oc = orientation(a, b, c)
  od = orientation(a, b, d)
  return oa * ob < 0 and oc * od < 0

# checks if quad is convex
def quadConvex(a, b, c, d):
  return orientation(a, b, c) == orientation(b, c, d) == orientation(c, d, a) == orientation(d, a, b) == -1

# finds trinagle that contains all points, else returns some arbitrary triangle (used for interactive mode)
def getBoundingTriangle(points):
  if len(points) == 0:
    return Vector(-1000, -1000), Vector(1000, -1000), Vector(0, 1000)
  minx = points[0].x
  maxx = points[0].x
  miny = points[0].x
  maxy = points[0].x
  for p in points:
    minx = min(minx, p.x)
    maxx = max(maxx, p.x)
    miny = min(miny, p.y)
    maxy = max(maxy, p.y)

  cx = (minx + maxx) / 2
  cy = (miny + maxy) / 2
  w = maxx - minx
  h = maxy - miny
  a = (math.sqrt(3) * 2 / 3) * h + w

  pad = a * 0.1 + 1

  return Vector(cx - a / 2 - pad, miny - pad), Vector(cx + a / 2 + pad, miny - pad), Vector(cx, miny + a * math.sqrt(3) / 2 + pad)

class Face:
  def __init__(self, edge):
    self.edge = edge
    self.vertex1 = edge.vertex
    self.vertex2 = edge.next.vertex
    self.vertex3 = edge.next.next.vertex
    self.children = []

  def addChild(self, face):
    self.children.append(face)

  # checks if face contains given point
  def contains(self, point, vertices):
    return orientation(vertices[self.vertex1], vertices[self.vertex2], point) <= 0 and orientation(vertices[self.vertex2], vertices[self.vertex3], point) <= 0 and orientation(vertices[self.vertex3], vertices[self.vertex1], point) <= 0

class HalfEdge:
  def __init__(self, twin, next, vertex, face = None):
    self.twin = twin
    self.next = next
    self.vertex = vertex
    self.face = face

  #  flips edge
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

  # splits face - new vertex lies inside the face
  def splitFace(self, newVertex):
    e1 = self
    e2 = e1.next
    e3 = e2.next

    a1 = HalfEdge(None, None, e2.vertex)
    b1 = HalfEdge(None, e1, newVertex)
    a1.next = b1
    e1.next = a1

    a2 = HalfEdge(None, None, e3.vertex)
    b2 = HalfEdge(None, e2, newVertex)
    a2.next = b2
    e2.next = a2

    a3 = HalfEdge(None, None, e1.vertex)
    b3 = HalfEdge(None, e3, newVertex)
    a3.next = b3
    e3.next = a3

    a1.twin = b2
    b2.twin = a1

    b1.twin = a3
    a3.twin = b1

    a2.twin = b3
    b3.twin = a2

    return (e1, e2, e3)

  # splits edge - when new vertex lies on edge
  def splitEdge(self, newVertex):
    e = self
    e1 = e.next
    e2 = e1.next
    e3 = e.twin.next
    e4 = e3.next

    b2 = HalfEdge(None, e2, newVertex)
    a2 = HalfEdge(None, b2, e.vertex)

    b4 = HalfEdge(None, e4, newVertex)
    a4 = HalfEdge(None, b4, e.twin.vertex)

    a1 = HalfEdge(None, e1, newVertex)
    b1 = HalfEdge(None, a1, e2.vertex)

    a3 = HalfEdge(None, e3, newVertex)
    b3 = HalfEdge(None, a3, e4.vertex)

    b1.twin = b2
    b2.twin = b1

    b4.twin = b3
    b3.twin = b4

    a1.twin = a4
    a4.twin = a1

    a2.twin = a3
    a3.twin = a2

    e4.next = a4
    e1.next = b1
    e2.next = a2
    e3.next = b3

    return (e1, e2, e3, e4)




class DelaunayAnimation:
  def __init__(self, points = [], on = False):
    self.on = on
    self.skip = False
    fig, ax = plt.subplots()
    self.anim = animations.Animation(fig, ax, "idk")
    self.points = points
    self.legalizing = None
    self.frame = 0
    self.constrained = set()

  def gp(self, p):
    return (self.points[p].x, self.points[p].y)

  def nextFrame(self):
    if not self.skip:
      self.frame += 1

  def addLine(self, p1, p2, c):    
    p1, p2 = min(p1, p2), max(p1, p2)
    if (p1, p2) in self.constrained:
      return
    self.anim.addAction(self.frame, lambda : self.anim.addLine(self.gp(p1), self.gp(p2), color=c))

  def deleteLine(self, p1, p2):
    p1, p2 = min(p1, p2), max(p1, p2)
    self.anim.addAction(self.frame, lambda : self.anim.deleteLine(self.anim.find_line_index(self.gp(p1), self.gp(p2))))

  def setLineColor(self, p1, p2, color):
    p1, p2 = min(p1, p2), max(p1, p2)
    if (p1, p2) in self.constrained:
      return
    self.anim.addAction(self.frame, lambda : self.anim.changeLineColor(color, self.anim.find_line_index(self.gp(p1), self.gp(p2))))

  def addTriangle(self, e1, e2, e3):
    self.addLine(e1.vertex, e2.vertex, 'blue')
    self.addLine(e2.vertex, e3.vertex, 'blue')
    self.addLine(e3.vertex, e1.vertex, 'blue')

  def triangleOut(self, edge):
    self.nextFrame()
    self.setLineColor(edge.vertex, edge.next.vertex, 'cyan')
    self.setLineColor(edge.next.vertex, edge.next.next.vertex, 'cyan')
    self.setLineColor(edge.next.next.vertex, edge.next.next.next.vertex, 'cyan')

  def constrain(self, p1, p2):
    if not self.on:
      return
    self.nextFrame()
    p1, p2 = min(p1, p2), max(p1, p2)    
    self.addLine(p1, p2, 'black')
    self.constrained.add((p1, p2))

  def flip(self, e1, e2):
    if not self.on:
      return
    self.nextFrame()
    self.setLineColor(e1.vertex, e1.next.vertex, 'red')
    self.nextFrame()
    self.deleteLine(e1.vertex, e1.next.vertex)
    self.addLine(e2.vertex, e2.next.vertex, 'green')
    self.nextFrame()
    self.setLineColor(e2.vertex, e2.next.vertex, 'blue')

  def addedPoint(self, p):
    if not self.on:
      return
    self.nextFrame()
    self.anim.addAction(self.frame, lambda : self.anim.addPoints([self.gp(p)], color = 'orange'))

  def locatedTriangle(self, edge):
    if not self.on:
      return
    self.nextFrame()
    p1 = edge.vertex
    p2 = edge.next.vertex
    p3 = edge.next.next.vertex
    self.setLineColor(p1, p2, 'red')
    self.setLineColor(p2, p3, 'red')
    self.setLineColor(p3, p1, 'red')
    self.nextFrame()
    self.setLineColor(p1, p2, 'blue')
    self.setLineColor(p2, p3, 'blue')
    self.setLineColor(p3, p1, 'blue')

  def addedToEdge(self, e1, e2, e3, e4):
    if not self.on:
      return
    self.nextFrame()
    self.addLine(e1.next.vertex, e1.next.next.vertex, 'blue')
    self.addLine(e3.next.vertex, e3.next.next.vertex, 'blue')

  def addedToFace(self, e1, e2, e3):
    if not self.on:
      return
    self.nextFrame()
    self.addLine(e1.next.vertex, e1.next.next.vertex, 'blue')
    self.addLine(e2.next.vertex, e2.next.next.vertex, 'blue')
    self.addLine(e3.next.vertex, e3.next.next.vertex, 'blue')

  def addInner(self, edge):
    if not self.on:
      return
    pass

class Mesh:

  def __init__(self, vertices, anim = DelaunayAnimation()):
    self.NoDelaunay = False
    self.anim = anim
    self.anim.skip = True

    self.OnSegmentEpsilon = 10**-6

    self.vertices = vertices
    self.faces = []
    self.verticesEdges = []

    p1, p2, p3 = getBoundingTriangle(vertices)
    vertices.append(p1)
    vertices.append(p2)
    vertices.append(p3)

    self.supertriangleIndices = [len(vertices) - 3, len(vertices) - 2, len(vertices) - 1]

    e3 = HalfEdge(None, None, len(vertices) - 1, 0)
    e2 = HalfEdge(None, e3, len(vertices) - 2, 0)
    e1 = HalfEdge(None, e2, len(vertices) - 3, 0)
    e3.next = e1

    self.addEdgeToSet(e1)
    self.addEdgeToSet(e2)
    self.addEdgeToSet(e3)

    self.faces.append(Face(e1))
    self.root = 0

    self.constrained = set()

    self.anim.addedPoint(e1.vertex)
    self.anim.addedPoint(e2.vertex)
    self.anim.addedPoint(e3.vertex)
    self.anim.addTriangle(e1, e2, e3)
    #self.anim.addedEdge(e1)
    #self.anim.addedEdge(e2)
    #self.anim.addedEdge(e3)

  def removeEdgeFromSet(self, edge):
    self.verticesEdges[edge.vertex].remove(edge)

  def addEdgeToSet(self, edge):
    while len(self.verticesEdges) <= edge.vertex:
      self.verticesEdges.append(set())
    self.verticesEdges[edge.vertex].add(edge)

  # checks if edge is present in triangulation
  def edgeAlreadyInTriangulation(self, index1, index2):
    return index2 in self.verticesEdges[index1]

  # flips quadritelar diagonal
  def flipEdge(self, edge):
    self.removeEdgeFromSet(edge)
    if edge.twin:
      self.removeEdgeFromSet(edge.twin)

    face1 = edge.face
    face2 = edge.twin.face

    newEdge = edge.flip()

    self.addEdgeToSet(newEdge)
    if newEdge.twin:
      self.addEdgeToSet(newEdge.twin)

    self.faces.append(Face(newEdge))
    newEdge.face = len(self.faces) - 1
    newEdge.next.face = len(self.faces) - 1
    newEdge.next.next.face = len(self.faces) - 1

    self.faces.append(Face(newEdge.twin))
    newEdge.twin.face = len(self.faces) - 1
    newEdge.twin.next.face = len(self.faces) - 1
    newEdge.twin.next.next.face = len(self.faces) - 1

    self.faces[face1].addChild(newEdge.face)
    self.faces[face1].addChild(newEdge.twin.face)
    self.faces[face2].addChild(newEdge.face)
    self.faces[face2].addChild(newEdge.twin.face)

    self.anim.flip(edge, newEdge)
    return newEdge

  # check if edge is legal
  def isEdgeLegal(self, edge):    
    if edge.vertex in self.supertriangleIndices and edge.next.vertex in self.supertriangleIndices:
      return True
    
    if edge.vertex in self.supertriangleIndices or edge.next.vertex in self.supertriangleIndices:
      return not quadConvex(self.vertices[edge.vertex], self.vertices[edge.twin.next.next.vertex], self.vertices[edge.twin.vertex], self.vertices[edge.next.next.vertex])
    
    if edge.next.next in self.supertriangleIndices or edge.twin.next.next in self.supertriangleIndices:
      return True    
    
    return self.NoDelaunay or circumcircleTest(self.vertices[edge.vertex], self.vertices[edge.next.vertex], self.vertices[edge.next.next.vertex], self.vertices[edge.twin.next.next.vertex]) <= 0



  # makes edge meet delaunay criteria
  def legalizeEdge(self, edge):
    if not self.isEdgeLegal(edge):
      e1 = edge.next
      e2 = e1.next

      newEdge = self.flipEdge(edge)

      if e1.twin:
        self.legalizeEdge(e1.twin)
      if e2.twin:
        self.legalizeEdge(e2.twin)

  # adds vertex on edge and keeps delaunay properties
  def addVertexToEdgeAndLegalize(self, edge, vertexIndex):
    face1 = edge.face
    face2 = edge.twin.face

    e1, e2, e3, e4 = edge.splitEdge(vertexIndex)

    if edge.twin:
      self.removeEdgeFromSet(edge.twin)
    self.removeEdgeFromSet(edge)
    self.addEdgeToSet(e1.next)
    self.addEdgeToSet(e1.next.next)
    self.addEdgeToSet(e2.next)
    self.addEdgeToSet(e2.next.next)
    self.addEdgeToSet(e3.next)
    self.addEdgeToSet(e3.next.next)
    self.addEdgeToSet(e4.next)
    self.addEdgeToSet(e4.next.next)

    self.faces.append(Face(e1))
    self.faces.append(Face(e2))
    self.faces.append(Face(e3))
    self.faces.append(Face(e4))

    e1.face = len(self.faces) - 4
    e1.next.face = len(self.faces) - 4
    e1.next.next.face = len(self.faces) - 4

    e2.face = len(self.faces) - 3
    e2.next.face = len(self.faces) - 3
    e2.next.next.face = len(self.faces) - 3

    e3.face = len(self.faces) - 2
    e3.next.face = len(self.faces) - 2
    e3.next.next.face = len(self.faces) - 2

    e4.face = len(self.faces) - 1
    e4.next.face = len(self.faces) - 1
    e4.next.next.face = len(self.faces) - 1

    self.faces[face1].children.append(len(self.faces) - 4)
    self.faces[face1].children.append(len(self.faces) - 3)

    self.faces[face2].children.append(len(self.faces) - 2)
    self.faces[face2].children.append(len(self.faces) - 1)

    self.anim.addedToEdge(e1, e2, e3, e4)

    if e1.twin:
      self.legalizeEdge(e1.twin)
    if e2.twin:
      self.legalizeEdge(e2.twin)
    if e3.twin:
      self.legalizeEdge(e3.twin)
    if e4.twin:
      self.legalizeEdge(e4.twin)

  # adds vertex inside face and keeps delaunay properties
  def addVertexToFaceAndLegalize(self, face, vertexIndex):
    e1, e2, e3 = self.faces[face].edge.splitFace(vertexIndex)

    self.addEdgeToSet(e1.next)
    self.addEdgeToSet(e1.next.next)
    self.addEdgeToSet(e2.next)
    self.addEdgeToSet(e2.next.next)
    self.addEdgeToSet(e3.next)
    self.addEdgeToSet(e3.next.next)

    self.faces.append(Face(e1))
    self.faces.append(Face(e2))
    self.faces.append(Face(e3))

    e1.face = len(self.faces) - 3
    e1.next.face = len(self.faces) - 3
    e1.next.next.face = len(self.faces) - 3

    e2.face = len(self.faces) - 2
    e2.next.face = len(self.faces) - 2
    e2.next.next.face = len(self.faces) - 2

    e3.face = len(self.faces) - 1
    e3.next.face = len(self.faces) - 1
    e3.next.next.face = len(self.faces) - 1

    self.faces[face].children.append(len(self.faces) - 3)
    self.faces[face].children.append(len(self.faces) - 2)
    self.faces[face].children.append(len(self.faces) - 1)

    self.anim.addedToFace(e1, e2, e3)

    if e1.twin:
      self.legalizeEdge(e1.twin)
    if e2.twin:
      self.legalizeEdge(e2.twin)
    if e3.twin:
      self.legalizeEdge(e3.twin)

  # adds vertex to face and keep delaunay properties
  def addVertexAndLegalize(self, face, vertexIndex):
    self.anim.addedPoint(vertexIndex)

    edge = self.faces[face].edge
    if onsegment(self.vertices[edge.vertex], self.vertices[edge.next.vertex], self.vertices[vertexIndex], self.OnSegmentEpsilon):
      self.addVertexToEdgeAndLegalize(edge, vertexIndex)
    elif onsegment(self.vertices[edge.next.vertex], self.vertices[edge.next.next.vertex], self.vertices[vertexIndex], self.OnSegmentEpsilon):
      self.addVertexToEdgeAndLegalize(edge.next, vertexIndex)
    elif onsegment(self.vertices[edge.next.next.vertex], self.vertices[edge.vertex], self.vertices[vertexIndex], self.OnSegmentEpsilon):
      self.addVertexToEdgeAndLegalize(edge.next.next, vertexIndex)
    else:
      self.addVertexToFaceAndLegalize(face, vertexIndex)

  # check if face contains vertex
  def faceContains(self, face, vertex):
    return self.faces[face].contains(vertex, self.vertices)

  # returns face which contains given vertex or None if none of the faces contains it
  def locate(self, vertex):
    face = self.root

    if not self.faces[face].contains(vertex, self.vertices):
      return None
    
    ITERATIONS = 0

    while self.faces[face].children:
      ITERATIONS += 1
      next = None
      for child in self.faces[face].children:
        if self.faceContains(child, vertex):
          next = child
          break
      if next == None:
        assert(False)
      face = next

    #print(ITERATIONS)
    self.anim.locatedTriangle(self.faces[face].edge)
    return face

  # checks if segments intersect, uses indices instead of vertices
  def segmentsIntersect(self, index1, index2, index3, index4):
    return segmentsIntersect(self.vertices[index1], self.vertices[index2], self.vertices[index3], self.vertices[index4])

  # returns deque of edges that intersect segment index1 -> index2
  def findIntersectingEdges(self, index1, index2):
    edge = None
    for e in self.verticesEdges[index1]:
        if self.segmentsIntersect(e.next.vertex, e.next.next.vertex, index1, index2):
          edge = e.next
          break
    if edge == None:
      return None

    ans = deque()
    ans.append(edge)
    while True:
      edge = edge.twin
      if self.segmentsIntersect(edge.next.vertex, edge.next.next.vertex, index1, index2):
        edge = edge.next
      elif self.segmentsIntersect(edge.next.next.vertex, edge.vertex, index1, index2):
        edge = edge.next.next
      else:
        break
      if edge not in self.verticesEdges[index2]:
        ans.append(edge)
      else:
        break
    return ans

  # constrains edge
  def constrainEdge(self, index1, index2):
    self.anim.skip = False
    self.anim.constrain(index1, index2)

    self.constrained.add((index1, index2))
    index1, index2 = (min(index1, index2), max(index1, index2))
    if self.edgeAlreadyInTriangulation(index1, index2):
      return

    intersecting = self.findIntersectingEdges(index1, index2)
    newEdges = []
  
    # recover constrained edge
    while intersecting:
      e = intersecting.popleft()
      if quadConvex(self.vertices[e.next.vertex], self.vertices[e.next.next.vertex], self.vertices[e.vertex], self.vertices[e.twin.next.next.vertex]):
        e = self.flipEdge(e)
        i1 = e.vertex
        i2 = e.next.vertex
        if i1 > i2:
          i1, i2 = i2, i1
        if i1 != index1 and i2 != index2 and segmentsIntersect(self.vertices[e.vertex], self.vertices[e.next.vertex], self.vertices[index1], self.vertices[index2]):
          intersecting.append(e)
        elif (min(e.vertex, e.next.vertex), max(e.vertex, e.next.vertex)) != (index1, index2):
          newEdges.append(e)
      else:
        intersecting.append(e)

    # restore delaunay
    done = False
    while not done:
      done = True
      for i in range(len(newEdges)):
        if not self.isEdgeLegal(newEdges[i]):
          newEdges[i] = self.flipEdge(newEdges[i])
          done = False

  # finds inner edges - inside constrained polygons
  def findInner(self):
    queue = deque()
    visited = [False] * len(self.faces)

    l = list(self.constrained)

    for i1, i2 in self.constrained:
      for e in self.verticesEdges[i1]:
        if e.next.vertex == i2:
          queue.append(e.face)
          visited[e.face] = True
          break

    while queue:
      face = queue.popleft()
      visited[face] = True
      self.anim.triangleOut(self.faces[face].edge)

      edges = [self.faces[face].edge, self.faces[face].edge.next, self.faces[face].edge.next.next]

      for e in edges:
        if (e.vertex, e.next.vertex) not in self.constrained and e.twin and not visited[e.twin.face]:
          queue.append(e.twin.face)

    return visited


  # checks if face contains super triangle vertex
  def usesSuperTriangle(self, face):
    return ( len( [v for v in [self.faces[face].vertex1, self.faces[face].vertex2, self.faces[face].vertex3] if v in [len(self.vertices) - 1, len(self.vertices) - 2, len(self.vertices) - 3]]) > 0 )

  # returns triangulation as triples of vertex indices for each triangle
  # filterSuperTriangle: if set to true triangles that contain super triangle vertices won't be outputed
  # removeOuter: if set to true removes triangles outside of constrained edges - constrained edges must form closed
  # polygons or else it will delete everything, also assumes that interior is on left side of edge
  def toTriangleList(self, filterSuperTriangle, removeOuter = False):
    ans = []
    inside = self.findInner()

    for face in range(len(self.faces)):
      if len(self.faces[face].children) == 0:
        superTriangleFilter = not filterSuperTriangle or not self.usesSuperTriangle(face)
        outerFilter = not removeOuter or inside[face]
        if superTriangleFilter and outerFilter:
          ans.append((self.faces[face].vertex1, self.faces[face].vertex2, self.faces[face].vertex3))

    return ans


def dt(points, constrains = [], shuffle = False):
  mesh = Mesh(points)

  indices = [i for i in range(0, len(points) - 3)]
  if shuffle:
    random.shuffle(indices)
  already = set()

  for i in indices:
    p = points[i]
    if (p.x, p.y) not in already:
      already.add((p.x, p.y))
      face = mesh.locate(p)
      mesh.addVertexAndLegalize(face, i)

  return mesh.toTriangleList(True, False)

def cdt(points, constrains = [], shuffle = False, anim = DelaunayAnimation()):
  mesh = Mesh(points, anim)

  indices = [i for i in range(0, len(points) - 3)]
  if shuffle:
    random.shuffle(indices)
  already = set()

  for i in indices:
    p = points[i]
    if (p.x, p.y) not in already:
      already.add((p.x, p.y))
      face = mesh.locate(p)
      mesh.addVertexAndLegalize(face, i)

  for (i1, i2) in constrains:
    mesh.constrainEdge(i1, i2)

  return mesh.toTriangleList(True, len(constrains) > 0)

def triangulate(points):
  return cdt(points, [ (i, (i + 1) % len(points)) for i in range(len(points))], True)