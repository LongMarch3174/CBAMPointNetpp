#!/usr/bin/env python3
# visualize_with_vtk.py

import numpy as np
import vtk
import matplotlib.pyplot as plt

def load_and_normalize(txt_path):
    # 读取 x y z label
    data = np.loadtxt(txt_path)
    xyz   = data[:, :3]
    labels = data[:, 3].astype(int)
    # 中心化
    centroid = xyz.mean(axis=0)
    xyz = xyz - centroid
    # 缩放到单位球
    scale = np.max(np.linalg.norm(xyz, axis=1))
    xyz = xyz / scale
    return xyz, labels

def create_vtk_point_cloud(xyz, labels):
    # 1. vtkPoints
    points = vtk.vtkPoints()
    for x, y, z in xyz:
        points.InsertNextPoint(x, y, z)

    # 2. 颜色数组（UnsignedChar 0–255）
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")
    # 用 matplotlib 的 tab20 生成调色板
    cmap = plt.get_cmap("tab20")
    for lbl in labels:
        r, g, b, _ = cmap(lbl % 20)
        colors.InsertNextTuple3(int(r*255), int(g*255), int(b*255))

    # 3. 将点和颜色放入 PolyData
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.GetPointData().SetScalars(colors)

    # 4. 顶点图元化（VertexGlyphFilter）
    glyph = vtk.vtkVertexGlyphFilter()
    glyph.SetInputData(poly)
    glyph.Update()

    return glyph.GetOutput()

def visualize(polydata):
    # Mapper 和 Actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(2)  # 点大小

    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)

    # Render Window
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(renderer)
    renWin.SetSize(800, 600)
    renWin.SetWindowName("VTK Point Cloud Visualization")

    # Interactor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()
    renWin.Render()
    iren.Start()

if __name__ == "__main__":
    TXT_PATH = "./test/scene/area_scene_50.txt"
    xyz, labels = load_and_normalize(TXT_PATH)
    polydata = create_vtk_point_cloud(xyz, labels)
    visualize(polydata)

