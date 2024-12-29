import random
import math
import matplotlib.pyplot as plt
from collections import deque




# TODO
# - refactor a bit?
# - test on artem-ogre tests
# literka a (odwrotna)
# gitara
# cdt

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

def onsegment(a, b, p, Eps):
  det = orientationDet(a, b, p)
  if abs(det) > Eps:
    return False

  return p.x >= min(a.x, b.x) and p.x <= max(a.x, b.x) and p.y >= min(a.y, b.y) and p.y <= max(a.y, b.y)

def circleTest(a, b, c, p):
  import numpy as np

  matrix = np.array([
      [a.x, a.y, a.x**2 + a.y**2, 1],
      [b.x, b.y, b.x**2 + b.y**2, 1],
      [c.x, c.y, c.x**2 + c.y**2, 1],
      [p.x, p.y, p.x**2 + p.y**2, 1],
  ])

  return sgn(np.linalg.det(matrix))

def segmentsIntersect(a, b, c, d):
  oa = orientation(c, d, a)
  ob = orientation(c, d, b)
  oc = orientation(a, b, c)
  od = orientation(a, b, d)
  return oa * ob < 0 and oc * od < 0

def quadConvex(a, b, c, d):
  return orientation(a, b, c) == orientation(b, c, d) == orientation(c, d, a) == orientation(d, a, b) == -1

class Face:
  def __init__(self, edge):
    self.edge = edge
    self.vertex1 = edge.vertex
    self.vertex2 = edge.next.vertex
    self.vertex3 = edge.next.next.vertex
    self.children = []

  def addChild(self, face):
    self.children.append(face)

  def contains(self, point, vertices):
    return orientation(vertices[self.vertex1], vertices[self.vertex2], point) <= 0 and orientation(vertices[self.vertex2], vertices[self.vertex3], point) <= 0 and orientation(vertices[self.vertex3], vertices[self.vertex1], point) <= 0

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

class Mesh:

  def __init__(self, vertices):
    self.OnSegmentEpsilon = 10**-6

    self.vertices = vertices
    self.faces = []
    self.edgeExists = set()
    self.verticesEdges = []

    p1, p2, p3 = getBoundingTriangle(vertices)
    vertices.append(p1)
    vertices.append(p2)
    vertices.append(p3)

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

  def normalizedEdgeForSet(self, edge):
    return ( min(edge.vertex, edge.next.vertex), max(edge.vertex, edge.next.vertex) )

  def removeEdgeFromSet(self, edge):
    self.edgeExists.remove((edge.vertex, edge.next.vertex))
    self.verticesEdges[edge.vertex].remove(edge)

  def addEdgeToSet(self, edge):
    self.edgeExists.add((edge.vertex, edge.next.vertex))
    while len(self.verticesEdges) <= edge.vertex:
      self.verticesEdges.append(set())
    self.verticesEdges[edge.vertex].add(edge)

  def edgeAlreadyInTriangulation(self, index1, index2):
    return (min(index1, index2), max(index1, index2)) in self.edgeExists

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

    return newEdge

  def isEdgeLegal(self, edge):
    return circleTest(self.vertices[edge.vertex], self.vertices[edge.next.vertex], self.vertices[edge.next.next.vertex], self.vertices[edge.twin.next.next.vertex]) <= 0

  def legalizeEdge(self, edge):
    if not self.isEdgeLegal(edge):
      e1 = edge.next
      e2 = e1.next

      self.flipEdge(edge)

      if e1.twin:
        self.legalizeEdge(e1.twin)
      if e2.twin:
        self.legalizeEdge(e2.twin)

  def addVertexToEdgeAndLegalize(self, edge, vertexIndex):
    face1 = edge.face
    face2 = edge.twin.face

    e1, e2, e3, e4 = edge.splitEdge(vertexIndex)

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

    if e1.twin:
      self.legalizeEdge(e1.twin)
    if e2.twin:
      self.legalizeEdge(e2.twin)
    if e3.twin:
      self.legalizeEdge(e3.twin)
    if e4.twin:
      self.legalizeEdge(e4.twin)

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

    if e1.twin:
      self.legalizeEdge(e1.twin)
    if e2.twin:
      self.legalizeEdge(e2.twin)
    if e3.twin:
      self.legalizeEdge(e3.twin)

  def addVertexAndLegalize(self, face, vertexIndex):
    edge = self.faces[face].edge
    if onsegment(self.vertices[edge.vertex], self.vertices[edge.next.vertex], self.vertices[vertexIndex], self.OnSegmentEpsilon):
      self.addVertexToEdgeAndLegalize(edge, vertexIndex)
    elif onsegment(self.vertices[edge.next.vertex], self.vertices[edge.next.next.vertex], self.vertices[vertexIndex], self.OnSegmentEpsilon):
      self.addVertexToEdgeAndLegalize(edge.next, vertexIndex)
    elif onsegment(self.vertices[edge.next.next.vertex], self.vertices[edge.vertex], self.vertices[vertexIndex], self.OnSegmentEpsilon):
      self.addVertexToEdgeAndLegalize(edge.next.next, vertexIndex)
    else:
      self.addVertexToFaceAndLegalize(face, vertexIndex)


  def contains(self, face, vertex):
    return self.faces[face].contains(vertex, self.vertices)

  def locate(self, vertex):
    face = self.root

    if not self.faces[face].contains(vertex, self.vertices):
      return None

    while self.faces[face].children:
      next = None
      for child in self.faces[face].children:
        if self.contains(child, vertex):
          next = child
          break
      if next == None:
        assert(False)
      face = next

    return face

  def segmentsIntersect(self, index1, index2, index3, index4):
    return segmentsIntersect(self.vertices[index1], self.vertices[index2], self.vertices[index3], self.vertices[index4])

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

  def constrainEdge(self, index1, index2):
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
        elif self.normalizedEdgeForSet(e) != (index1, index2):
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

    # return visited

    while queue:
      face = queue.popleft()
      visited[face] = True

      edges = [self.faces[face].edge, self.faces[face].edge.next, self.faces[face].edge.next.next]

      for e in edges:
        if (e.vertex, e.next.vertex) not in self.constrained and e.twin and not visited[e.twin.face]:
          queue.append(e.twin.face)


    return visited


  def toTriangleList(self, filterSuperTriangle, removeOuter = False):
    ans = []

    queue = []
    l = 0

    queue.append(self.root)

    visited = [False] * len(self.faces)

    inside = self.findInner()

    for face in range(len(self.faces)):
      if len(self.faces[face].children) == 0:
        superTriangleFilter = not filterSuperTriangle or not ( len( [v for v in [self.faces[face].vertex1, self.faces[face].vertex2, self.faces[face].vertex3] if v in [len(self.vertices) - 1, len(self.vertices) - 2, len(self.vertices) - 3]]) > 0 )
        outerFilter = not removeOuter or inside[face]
        if superTriangleFilter and outerFilter:
          ans.append((self.faces[face].vertex1, self.faces[face].vertex2, self.faces[face].vertex3))

    return ans


def cdt(points, constrains = []):
  mesh = Mesh(points)

  indices = [i for i in range(0, len(points) - 3)]
  #random.shuffle(indices)
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


def triangulatePolygon(points):
  return cdt(points, [ (i, (i + 1) % len(points)) for i in range(len(points))])

def textScatter(xs, ys, **kwargs):
  # Create a scatter plot
  scatter = plt.scatter(xs, ys, **kwargs)

  # Add numbering to each point
  for i, (xi, yi) in enumerate(zip(xs, ys)):
    plt.text(xi, yi, str(i), fontsize=12, ha='right', va='bottom')

  return scatter

def interactive():
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_xlim([-2000, 2000])
  ax.set_ylim([-2000, 2000])

  mesh = Mesh([])


  def onclick(event):
    nonlocal mesh

    prevPos = Vector(event.xdata, event.ydata)

    if  event.inaxes:
      if event.button == 1:
        mesh.vertices.append(Vector(event.xdata, event.ydata))
        l = mesh.locate(Vector(event.xdata, event.ydata))
        if l != None:
          mesh.addVertexAndLegalize(l, len(mesh.vertices) - 1)

  def onmoved(event):
    nonlocal mesh
    #plt.scatter([p.x for p in mesh.vertices], [p.y for p in mesh.vertices], color = 'orange')

    #print(mesh.vertices)
    plt.clf()
    textScatter([p.x for p in mesh.vertices], [p.y for p in mesh.vertices], color = 'orange')
    t = mesh.toTriangleList(False)

    for (i1, i2, i3) in t:
      txs = [ mesh.vertices[i].x for i in [i1, i2, i3, i1] ]
      tys = [ mesh.vertices[i].y for i in [i1, i2, i3, i1] ]
      plt.plot(txs, tys, color = 'blue')

    if  event.inaxes:
      plt.scatter([event.xdata], [event.ydata])
      l = mesh.locate(Vector(event.xdata, event.ydata))
    else:
      l = None
    if l != None:
      vs = [ mesh.vertices[i] for i in [mesh.faces[l].vertex1, mesh.faces[l].vertex2, mesh.faces[l].vertex3, mesh.faces[l].vertex1] ]
      plt.plot([p.x for p in vs], [p.y for p in vs], color = 'red')
    fig.canvas.draw()

  prevValue = None

  def on_key(event):
    nonlocal prevValue
    if event.key.isdigit():
      digit_value = int(event.key)
      if prevValue == None:
        prevValue = digit_value
      else:
        mesh.constrainEdge(prevValue, digit_value)
        prevValue = None

  fig.canvas.mpl_connect('key_press_event', on_key)

  cid = fig.canvas.mpl_connect('button_press_event', onclick)
  cid = fig.canvas.mpl_connect('motion_notify_event', onmoved)
  plt.show()

def genRandomOnCircle(n, radius):
  angles = [random.random() * 2 * math.pi for _ in range(n)]
  angles.sort()
  xs = [ math.cos(angles[i]) * radius for i in range(n) ]
  ys = [ math.sin(angles[i]) * radius for i in range(n) ]
  return xs, ys

def ogreTest():
  n, m = input().split()
  n = int(n)
  m = int(m)

  points = []
  for i in range(n):
    x, y = input().split()
    x = float(x)
    y = float(y)
    points.append(Vector(x, y))

  constrains = []
  for i in range(m):
    a, b = input().split()
    a = int(a)
    b = int(b)
    constrains.append((a, b))
    #constrains.append((b, a))

  t = cdt(points, constrains)

  xs = [ p.x for p in points ]
  ys = [ p.y for p in points ]
  plt.scatter(xs, ys)
  print(len(t))
  for (i1, i2, i3) in t:
    txs = [ xs[i1], xs[i2], xs[i3], xs[i1] ]
    tys = [ ys[i1], ys[i2], ys[i3], ys[i1] ]
    plt.plot(txs, tys)

  for (i1, i2) in constrains:
    txs = [ xs[i1], xs[i2] ]
    tys = [ ys[i1], ys[i2] ]
    plt.plot(txs, tys, color = 'black')
  plt.show()


if __name__ == "__main__":
  # ogreTest()
  # quit(0)
  #testForCrash()
  interactive()
  #quit(0)
  #points = [Vector(-42.31, 17.54), Vector(-277.81, -231.21), Vector(-295.61, -797.51), Vector(403.87, -824.06), Vector(673.50, -633.73), Vector(372.49, -128.59), Vector(137.48, -372.77)]
  n = int(input())
  points = []
  for i in range(n):
    x, y = input().split()
    x = float(x)
    y = float(y)
    points.append(Vector(x, y))
  #points = list(reversed(points))
  # xs, ys = genRandomOnCircle(10000, 10)
  # points = [ Vector(xs[i], ys[i]) for i in range(len(xs))]
  t = triangulatePolygon(points)
  #t = delaunay(points)
  # points.pop()
  # points.pop()
  # points.pop()
  xs = [ p.x for p in points ]
  ys = [ p.y for p in points ]
  textScatter(xs, ys)
  plt.show()
  plt.scatter(xs, ys)
  print(len(t))
  for (i1, i2, i3) in t:
    txs = [ xs[i1], xs[i2], xs[i3], xs[i1] ]
    tys = [ ys[i1], ys[i2], ys[i3], ys[i1] ]
    plt.plot(txs, tys)
  plt.show()





# sources:
# Some great thing about non-constrained delaunay https://ianthehenry.com/posts/delaunay/#fnref:2
# constrained but in type script https://tchayen.com/constrained-delaunay-triangulation-from-a-paper
# there is also chapter in 'the' book
# https://cp-algorithms.com/geometry/delaunay.html
# it seems there are few different options divida and conqure or online idk which one to use
# http://www.geom.uiuc.edu/~samuelp/del_project.html
# divide and conquer seems great dunno about constraining tho
# https://web.archive.org/web/20210506140628if_/https://www.newcastle.edu.au/__data/assets/pdf_file/0019/22519/23_A-fast-algortithm-for-generating-constrained-Delaunay-triangulations.pdf