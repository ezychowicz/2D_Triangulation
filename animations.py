import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.collections
from pathlib import Path
class Animation:
    '''
    Klasa zawierające funkcje do tworzenia animacji. 
    
    W self.actions przechowuje się akcje jakie chce się wykonać w danej klatce (klucz=frame->wartość:akcja).
    
    Funkcje find_* służą do szukania odpowiedniego obiektu w obiektach osi self.ax.
    
    addLine i addPoints służą do dodawania obiektu na rysunek odpowiednio: linii i obiektu scatter
    addAction służy do dodawania akcji do self.actions.
    
    Funkcje delete* usuwają obiekt z list self.ax.collections/self.ax.lines
    
    update jest funkcją używaną przez wywołanie animacji. Przekazując numer klatki, funkcja wykonuje
    zmiany zapisane w self.actions dla tej klatki, po czym zwraca zaktualizowane listy obiektów.
    '''
    def __init__(self, fig, ax, setName):
        self.fig = fig
        self.ax = ax
        self.actions = {}
        self.setName = setName
    def find_scatter_index(self, points):
        if len(points) == 0:
            return
        pointsX, pointsY = zip(*points) 
        for index, collection in enumerate(self.ax.collections):
            if isinstance(collection, matplotlib.collections.PathCollection):
                scatter_offsets = collection.get_offsets()
                scatterX, scatterY = zip(*scatter_offsets)
                if list(scatterX) == list(pointsX) and list(scatterY) == list(pointsY):
                    return index 

        return None  

    def find_line_index(self, A, B):        
        for index, collection in enumerate(self.ax.lines):
            if isinstance(collection, plt.Line2D):
                if list(collection.get_xdata()) == [A[0], B[0]] and list(collection.get_ydata()) == [A[1], B[1]]:
                    return index                

    def addLine(self, A, B, **kwargs):
        line = self.ax.plot([A[0], B[0]], [A[1], B[1]], **kwargs) #,
        return line 
    
    def addTriangleLines(self, A, B, C, **kwargs):
        line1 = self.addLine(A, B, **kwargs)
        line2 = self.addLine(B, C, **kwargs)
        line3 = self.addLine(C, A, **kwargs)
        return line1, line2, line3

    def addAction(self, frame, action):
        if not frame in self.actions:
            self.actions[frame] = []
        self.actions[frame].append(action)

    def addPoints(self, points, **kwargs):
        if len(points) == 0:
            return
        pointsX, pointsY = zip(*points)
        scatter = self.ax.scatter(pointsX, pointsY, **kwargs)
        return scatter  

    def changeLineColor(self, color, index, alpha=None):
        if index is None:
            return        
        if alpha != None:
            self.ax.lines[index].set_alpha(alpha)
        self.ax.lines[index].set_color(color)

    def changePointsColor(self, color, index):
        if index == None:
            return
        self.ax.collections[index].set_facecolor(color)

    def changePointsSize(self, size, index):
        if index == None:
            return
        self.ax.collections[index].set_sizes([size])

    def deletePoints(self, index):
        if index == None:
            return
        self.ax.collections[index].remove()


    def deleteLine(self, index):
        if index == None:
            return
        self.ax.lines[index].remove()

    def update(self, frame):        
        if frame in self.actions:            
            for a in self.actions[frame]:
                a()
                #self.actions[frame]()  
            del self.actions[frame]

        return self.ax.collections + self.ax.lines  

    def draw(self, frameTime,saveGIF = False):
        framesCnt = max(self.actions.keys())
        ani = animation.FuncAnimation(self.fig, self.update, frames=framesCnt + 100, interval=frameTime, repeat=False,blit=False)
        if not saveGIF:
            plt.show(block = True)
        if saveGIF:
            path = Path(__file__).parent/f"{self.setName}.gif" 
            ani.save(path, writer = 'pillow', fps = 1000/frameTime)
        