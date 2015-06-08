import os.path, sys
from math import pi 


try:
  from dolfin.compilemodules.compilemodule import compile_extension_module
  
  src_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "cpp")

  with open(os.path.join(src_dir, "OrdinatesData.h"), "r") as header:
    ordinates_ext_module = compile_extension_module(header.read(),
                                                    include_dirs=[".", src_dir],
                                                    source_directory=src_dir,
                                                    sources=["OrdinatesData.cpp"])
except IOError as e:
  print "Cannot open source files for extension module."
  print "Details: ERR#{0}: {1}".format(e.errno, e.strerror)
  sys.exit(-1)
  
quadrature = ordinates_ext_module.OrdinatesData(4, 3, "lgvalues.txt")
print
print "S{}, {}D, {} directions".format(quadrature.get_N(), quadrature.get_D(), quadrature.get_M())
print
print "----------------  XI  ---------------------"
print quadrature.get_xi()
print "---------------  ETA  ---------------------"
print quadrature.get_eta()
print "----------------  MU  ---------------------"
print quadrature.get_mu()
print "-------------- WEIGHTS  -------------------"
pw = quadrature.get_pw()
print pw
print pw.sum() / (4*pi)

for m in range(quadrature.get_M()):
  print m, quadrature.get_ordinate(m)
print





try:
  with open(os.path.join(src_dir, "AngularTensors.h"), "r") as header:
    tensors_ext_module = compile_extension_module(header.read(),
                                                  include_dirs=[".", src_dir],
                                                  source_directory=src_dir,
                                                  sources=["AngularTensors.cpp"])
except IOError as e:
  print "Cannot open source files for extension module."
  print "Details: ERR#{0}: {1}".format(e.errno, e.strerror)
  sys.exit(-1)
  
import numpy as np
tensors = tensors_ext_module.AngularTensors(quadrature, 0)

np.set_printoptions(threshold=1000, linewidth=200)

Q = np.reshape(tensors.Q(), tensors.shape_Q())
print Q
print
Qt = np.reshape(tensors.Qt(), tensors.shape_Qt())
print Qt
print
G = np.reshape(tensors.G(), tensors.shape_G())
print G
print
T = np.reshape(tensors.T(), tensors.shape_T())
print T
print
