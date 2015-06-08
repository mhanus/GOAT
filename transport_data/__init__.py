import os

_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "cpp"))

try:
  from dolfin.cpp.common import dolfin_error
  from dolfin.compilemodules.compilemodule import compile_extension_module
except Exception as e:
  print "Invalid DOLFIN installation:"
  raise e

try:
  with open(os.path.join(_src_dir, "OrdinatesData.h"), "r") as header:
    ordinates_ext_module = compile_extension_module(header.read(),
                                                    include_dirs=[".", _src_dir],
                                                    source_directory=_src_dir,
                                                    sources=["OrdinatesData.cpp"])
except Exception as e:
  dolfin_error(__file__,
               "initialize transport data",
               "Cannot compile extension module for ordinates ({})".format(e))


try:
  with open(os.path.join(_src_dir, "AngularTensors.h"), "r") as header:
    angular_tensors_ext_module = compile_extension_module(header.read(),
                                                          include_dirs=[".", _src_dir],
                                                          source_directory=_src_dir,
                                                          sources=["AngularTensors.cpp"])
except Exception as e:
  dolfin_error(__file__,
               "initialze transport data",
               "Cannot compile extension module for angular tensors ({})".format(e))

quadrature_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "lgvalues.txt"))