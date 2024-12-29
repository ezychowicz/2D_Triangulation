import matplotlib.pyplot as plt
import numpy as np
import json
import time
from pathlib import Path


def export_json_path(points, edges, path_to_json):
    figure = {"points": points, "edges": edges}
    with open(path_to_json, 'w') as json_file:
        json.dump(figure, json_file, indent = 4)

def export_json_triangulation_path(points, triangles, path_to_json):
    figure = {"points": points, "triangles": triangles}
    with open(path_to_json, 'w') as json_file:
        json.dump(figure, json_file, indent = 4)

def export_json(points, edges):
    data_dir = Path(__file__).parent.parent / "data"
    path_to_json  = data_dir / "exportData.json" #tworzy ścieżkę do jsona pobierając ścieżkę do tego pliku
    export_json_path(path_to_json)


class Commit:
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.points = []
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        eps = 0.2
        if event.inaxes is None:
            return
        new_cord = [event.xdata, event.ydata]
        #print(new_cord)
        if event.button == 1:
            self.points.append(new_cord)
        if event.button == 3:
            mini = float('inf')
            to_remove = None
            for idx, point in enumerate(self.points):
                old_cord = point
                if self.dist(old_cord, new_cord) < eps and self.dist(old_cord, new_cord) < mini:
                    to_remove = idx
                    mini = self.dist(old_cord, new_cord)
            if to_remove is not None:
                self.points.pop(to_remove)

    def dist(self,A,B):
        return (A[0] - B[0])**2 + (A[1] - B[1])**2

    def push_changes(self):
        plt.sca(self.fig.axes[0])
        for collection in self.ax.collections:
            collection.remove()
        if self.points:
            X, Y = zip(*self.points)
            plt.scatter(X, Y, color = 'blue', s = 10, marker = 'o')


def graphing(xlim,ylim):
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    commit = Commit(fig, ax)
    finish = False
    print("Aby przerwać dodawanie punktów należy w konsoli wpisać CTRL+C.")
    while not finish:
        try:
            commit.push_changes()
            plt.pause(0.4)
        except(KeyboardInterrupt):
            print("Wymuszono zatrzymanie. Aby zakończyć działanie należy zamknąć okno wykresu.")
            finish = True
    plt.show(block = True)

    N = len(commit.points)
    edges = []
    for i in range (1,N):
        edges.append((i - 1, i))
    edges.append((N - 1, 0))
    export_json(commit.points, edges)

if __name__ == '__main__':
    graphing((0,10), (0,10))