__author__ = 'Milan'

import os

try:
  from dolfin.compilemodules.compilemodule import compile_extension_module

  _src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "cpp", "PETSc"))

  with open(os.path.join(_src_dir, "PETSc_utils.h"), "r") as header:
    backend_ext_module = compile_extension_module(header.read(),
                                                  include_dirs=[".", _src_dir],
                                                  source_directory=_src_dir,
                                                  sources=["PETSc_utils.cpp"])

except EnvironmentError as e:
  print "Cannot open source files for PETSc extension module: {}".format(e)
  raise e
except Exception as e:
  print "Cannot initialize PETSc extension module"
  raise e