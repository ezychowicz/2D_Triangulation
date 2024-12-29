import json
import os
import delaunay
from utils import interactive_figure, draw_triangulation

TestsPath = "delaunayTests/"

def interactive():
  pass

def test(inpath, outpath):
  with open(inpath,"r") as jsonFile:
    figure = json.load(jsonFile)

  with open(outpath,"r") as jsonFile2:
    ansfigure = json.load(jsonFile2)  

  points = figure["points"]
  constrains = figure["edges"]

  got = delaunay.cdt([delaunay.Vector(x, y) for x, y in points], constrains)
  correct = ansfigure["triangles"]

  for i in range(len(got)):
    if got[i] != tuple(correct[i]):
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
  pass

def specificTest():
  path = input()

  with open(path,"r") as jsonFile:
    figure = json.load(jsonFile)  

  points = figure["points"]
  constrains = figure["edges"]

  ans = delaunay.cdt([delaunay.Vector(x, y) for x, y in points], constrains)  

  draw_triangulation.draw(points, ans, constrains)

  interactive_figure.export_json_triangulation_path(points, ans, "tmpans2.json")

if __name__ == "__main__":
  print("[A] - interactive, \n[B] - tests, \n[C] - input polygon, get triangulation and save it, \n[D] [Path] loads test from path and shows results")
  action = str(input())

  if len(action) != 1:
    print("Invalid action")
    quit(0)

  index = ord(action) - ord('A')

  functions = [interactive, tests, inputPolygon, specificTest]
  functions[index]()