import numpy as np
import math
import matplotlib.pyplot as plt

# np.random.seed(155)


def sortPair(a,b):
    if a == b:
        b += 10**(-10)
    return (a,b) if a < b else (b,a)
def generate(r1, r2, n):
    fi1 = sorted([np.random.uniform(0, 2*math.pi) for _ in range(2*n)])
    resCCW = [(r1 * math.cos(fi), r1 * math.sin(fi)) if i%2 == 0 else (r2*math.cos(fi), r2*math.sin(fi)) for i,fi in enumerate(fi1)]
    return resCCW
 

class Vector2D:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __add__(self, other):
    return Vector2D(self.x + other.x, self.y + other.y)

  def __sub__(self, other):
    return Vector2D(self.x - other.x, self.y - other.y)

  def __mul__(self, scalar):
    return Vector2D(self.x * scalar, self.y * scalar)

  def __truediv__(self, scalar):
    return Vector2D(self.x / scalar, self.y / scalar)

  def dot(self, other):
    return self.x * other.x + self.y * other.y

  def magnitude(self):
    return (self.x**2 + self.y**2) ** 0.5

  def normalize(self):
    mag = self.magnitude()
    if mag == 0:
      return Vector2D(0, 0)

  def rotateRight(self):
    self.x, self.y = self.y, -self.x

def genRegularPolygon(n, radius, offset = 0):
  step = (2 * math.pi / n)
  xs = [ math.cos(step * i + offset) * radius for i in range(n) ]
  ys = [ math.sin(step * i + offset) * radius for i in range(n) ]
  return xs, ys

def snowflake(iterations, radius):
  xs, ys = genRegularPolygon(3, radius)
  points = [ Vector2D(xs[i], ys[i]) for i in range(len(xs)) ]

  for i in range(iterations):
    newPoints = []
    for j in range(len(points)):
      next = (j + 1) % len(points)

      p1 = points[j]
      p5 = points[next]
      vect = p5 - p1
      vect /= 3.
      p2 = p1 + vect
      perp = Vector2D(vect.x, vect.y)
      perp.rotateRight()
      perp *= math.sqrt(3) / 2
      p3 = p2 + vect / 2. + perp
      p4 =  p2 + vect
      newPoints.append(p1)
      newPoints.append(p2)
      newPoints.append(p3)
      newPoints.append(p4)
    points = newPoints
  return list(zip([p.x for p in points], [p.y for p in points]))