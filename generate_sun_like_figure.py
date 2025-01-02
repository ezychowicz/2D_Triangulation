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