import sys

try:
  import dolfin
  from dolfin.cpp.mesh import Mesh, MeshQuality
except ImportError:
  raise ImportError('Problem with Dolfin Installation')

if len(sys.argv) != 2:
  print "Correct usage: python check_mesh_quality.py path_to_mesh_file.xml"
  sys.exit(-1)
  
mesh = Mesh(sys.argv[1])
print str(mesh)

mq = MeshQuality.radius_ratio_matplotlib_histogram(mesh)
exec mq