# Test loading of boundary conditions from core.dat

region_map = {1 : "reg1", 2 : "reg2"}
material_map = {"reg1" : "M1", "reg2" : "M2"}
boundary_map = {1 : "vacuum", 2 : "reflective"}

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