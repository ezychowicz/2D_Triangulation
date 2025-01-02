import numpy as np
import math
import matplotlib.pyplot as plt

# np.random.seed(155)


def sortPair(a,b):
    if a == b:
        b += 10**(-10)
    return (a,b) if a < b else (b,a)
def generate(r1, r2, n):
    # fi1 = sorted([np.random.uniform(0, 2*math.pi) for _ in range(n)])
    # inner = [(r1 * math.cos(fi), r1 * math.sin(fi)) for fi in fi1]
    # fi2 = sorted([np.random.uniform(*sortPair(fi1[i], fi1[(i + 1)%n])) for i in range (n)])
    # outer = [(r2*math.cos(fi), r2*math.sin(fi)) for fi in fi2]
    fi1 = sorted([np.random.uniform(0, 2*math.pi) for _ in range(2*n)])
    resCCW = [(r1 * math.cos(fi), r1 * math.sin(fi)) if i%2 == 0 else (r2*math.cos(fi), r2*math.sin(fi)) for i,fi in enumerate(fi1)]
    return resCCW
 
def visualize(points):
    """
    Wizualizuje zbiór punktów i łączy kolejne punkty liniami.
    
    :param points: Lista punktów [(x1, y1), (x2, y2), ...].
    """
    # Rozdzielenie listy punktów na współrzędne x i y
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    
    # Dodanie pierwszego punktu na koniec, aby zamknąć wielokąt
    x.append(points[0][0])
    y.append(points[0][1])
    
    # Rysowanie punktów
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, marker='o', linestyle='-', color='b', label="Linia")
    plt.scatter(x[:-1], y[:-1], color='red', zorder=5, label="Punkty")  # Punkty
    
    # Ustawienia osi
    plt.axis('equal')  # Równe proporcje na osiach
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title("Wizualizacja punktów")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# visualize(generate(1,6,10))

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