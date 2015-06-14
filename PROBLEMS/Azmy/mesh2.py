# Test default region -> region_name mapping (str)

material_map = {"1" : "M1", "2" : "M2"}

vacuum_boundaries = [1]
reflective_boundaries = [2]

from dolfin.cpp.mesh import RectangleMesh, MeshFunctionSizet, SubDomain
from dolfin import DOLFIN_EPS

class M1(SubDomain):
  def inside(self, x, on_boundary):
    return x[0] < 5.0 + DOLFIN_EPS and x[1] < 5.0 + DOLFIN_EPS

class Reflective(SubDomain):
  def inside(self, x, on_boundary):
    return (x[0] < DOLFIN_EPS and on_boundary) or (x[1] < DOLFIN_EPS and on_boundary)

mesh = RectangleMesh(0, 0, 10, 10, 32, 32)

regions = MeshFunctionSizet(mesh, mesh.topology().dim(), 2)
M1().mark(regions, 1)

boundaries = MeshFunctionSizet(mesh, mesh.topology().dim() - 1, 1)
Reflective().mark(boundaries, 2)