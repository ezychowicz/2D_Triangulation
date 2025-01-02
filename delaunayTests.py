import json
import time
import os
import delaunay
from utils import interactive_figure, draw_triangulation
import matplotlib.pyplot as plt
import generate_sun_like_figure

TestsPath = "delaunayTests/"



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

  mesh = delaunay.Mesh([])

  def onclick(event):
    nonlocal mesh

    prevPos = delaunay.Vector(event.xdata, event.ydata)

    if  event.inaxes:
      if event.button == 1:
        mesh.vertices.append(delaunay.Vector(event.xdata, event.ydata))
        l = mesh.locate(delaunay.Vector(event.xdata, event.ydata))
        if l != None:
          mesh.addVertexAndLegalize(l, len(mesh.vertices) - 1)

  def onmoved(event):
    nonlocal mesh
    plt.clf()
    textScatter([p.x for p in mesh.vertices], [p.y for p in mesh.vertices], color = 'orange')
    t = mesh.toTriangleList(False)

    for (i1, i2, i3) in t:
      txs = [ mesh.vertices[i].x for i in [i1, i2, i3, i1] ]
      tys = [ mesh.vertices[i].y for i in [i1, i2, i3, i1] ]
      plt.plot(txs, tys, color = 'blue')

    if  event.inaxes:
      plt.scatter([event.xdata], [event.ydata])
      l = mesh.locate(delaunay.Vector(event.xdata, event.ydata))
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



def test(inpath, outpath):
  with open(inpath,"r") as jsonFile:
    figure = json.load(jsonFile)

  with open(outpath,"r") as jsonFile2:
    ansfigure = json.load(jsonFile2)

  points = figure["points"]
  constrains = figure["edges"]

  got = delaunay.cdt([delaunay.Vector(x, y) for x, y in points], constrains)  
  correct = [ tuple(v) for v in ansfigure["triangles"]  ]

  got.sort()
  correct.sort()  

  for i in range(len(got)):
    if not got[i] in correct and not (got[i][1], got[i][2], got[i][0]) in correct and not (got[i][2], got[i][0], got[i][1]) in correct:
      return False
  return True



def tests():
  all = True
  directory = os.fsencode(TestsPath)
  for file in os.listdir(directory):
    file_path = os.path.join(directory, file)
    if not os.path.isfile(file_path):
      continue
    filename = os.fsdecode(file)

    if test(TestsPath + "/" + filename, TestsPath + "ans/" + filename):
      print("Test: " + filename + " " + "OK")
    else:
      all = False
      print("Test: " + filename + " " + "WRONG ANSWER!")
  if all:
    print("ALL TESTS PASSED")
  else:
    print("WRONG ANSWERS WERE GIVEN")

def inputPolygon():
  interactive_figure.graphing((-10, 10), (-10, 10))
  print("XD?")

def specificTest():
  path = input()

  with open(path,"r") as jsonFile:
    figure = json.load(jsonFile)

  points = figure["points"]
  constrains = figure["edges"]

  pointsv = [delaunay.Vector(x, y) for x, y in points]
  anim = delaunay.DelaunayAnimation(pointsv, True)
  #anim = delaunay.DelaunayAnimation()
  ans = delaunay.cdt(pointsv, constrains, False, anim)
  anim.anim.draw(100)

  draw_triangulation.draw(points, ans, constrains)

def specificTestNoConstrains():
  path = input()

  with open(path,"r") as jsonFile:
    figure = json.load(jsonFile)

  points = figure["points"]  

  pointsv = [delaunay.Vector(x, y) for x, y in points]  
  ans = delaunay.dt(pointsv, False)  
  pointsv.pop()
  pointsv.pop()
  pointsv.pop()
  draw_triangulation.draw([(v.x, v.y) for v in pointsv], ans)

  #interactive_figure.export_json_triangulation_path(points, ans, "tmpans2.json")

def sunLikeThing():

  points = generate_sun_like_figure.snowflake(4, 1)

  start = time.time()
  ans = delaunay.triangulate(points)
  end = time.time()
  print(end - start)
  
  #draw_triangulation.draw([(v.x, v.y) for v in pointsv], ans)

if __name__ == "__main__":
  print(
    "[A] - interactive, \n\
    [B] - tests, \n\
    [C] - input polygon, get triangulation and save it, \n\
    [D] [Path] loads test from path and shows results, \n\
    [E] [Path] loads test from path and shows results no constrains, \n\
    [F] sun like thing")
  action = str(input())

  if len(action) != 1:
    print("Invalid action")
    quit(0)

  index = ord(action) - ord('A')

  functions = [interactive, tests, inputPolygon, specificTest, specificTestNoConstrains, sunLikeThing]
  functions[index]()