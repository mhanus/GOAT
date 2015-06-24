from collections import deque
from dolfin.cpp.common import info
import dolfin
from common import MPI, comm

import re
__stripper = re.compile(r"\s+=\s+")

__author__ = 'mhanus'

def set_default_dolfin_parameters():
  #parameters['num_threads'] = 6
  dolfin.parameters['form_compiler']['optimize'] = True
  dolfin.parameters['form_compiler']['cpp_optimize'] = True
  dolfin.parameters['form_compiler']['cpp_optimize_flags'] = '-O3'
  dolfin.parameters["refinement_algorithm"] = "plaza_with_parent_facets"
  #dolfin.parameters['form_compiler']['representation'] = 'quadrature'
  #dolfin.parameters["form_compiler"]["quadrature_degree"] = 2


# noinspection PyArgumentList
def add_solver_parameters():
  import problem_data
  import discretization_modules.generic_discretization as discretization
  import flux_modules.elliptic_sn_flux_module as flux_module

  dolfin.parameters.add(problem_data.get_parameters())
  dolfin.parameters.add(discretization.get_parameters())
  dolfin.parameters.add(flux_module.get_parameters())

def print_info():
  if MPI.rank(comm) == 0:
    info(dolfin.parameters, True)

def load_algebraic_solver_parameters(algebraic_solver_params_file):
  """
  Load PETSC / SLEPc parameters from file.

  :param str algebraic_solver_params_file:  Path to the file with the parameters.
  :return: String that can be processed by Dolfin's parameters system.
  :rtype: str
  """

  alg_solver_args = ""

  if algebraic_solver_params_file:
    try:
      with open (algebraic_solver_params_file, "r") as algebraic_solver_params_file:
        for l in algebraic_solver_params_file:
          s = l.strip().split('#')[0]
          if s:
            if s == "switch_mat":
              alg_solver_args += " --flux_module.eigensolver.switch_mat=True"
            else:
              alg_solver_args += " --petsc." + __stripper.sub("=", s)
    except Exception as e:
      raise RuntimeError("Solver parameters could not be read from "+algebraic_solver_params_file + ".\n" + str(e))

  return alg_solver_args

class InvalidFileFormat(Exception):
  def __init__(self):
    super(InvalidFileFormat, self).__init__("Invalid file format.")

def load_olver_parameters(solver_params_file):
  """
  Load coupled solver parameters from file.

  :param str solver_params_file:  Path to the file with the parameters.
  :return: String that can be processed by Dolfin's parameters system.
  :rtype: str
  """

  cpl_solver_args = ""

  if solver_params_file:
    try:
      with open (solver_params_file, "r") as coupled_solver_params_file:
        modules = deque()

        for l in coupled_solver_params_file:
          s = l.strip().split('#')[0]
          if s:
            # Find submodule level
            for i,c in enumerate(s):
              if c == "[":
                if s[-1-i] != "]":
                  raise InvalidFileFormat
              else:
                break

            if i > 0:
              # Get module name
              module = s[i:-i].strip().lower()

              # Add it to the appropriate level in the submodules structure
              nlvl = len(modules)
              if i == nlvl + 1:
                modules.append(module)
              elif i == nlvl:
                modules[-1] = module
              elif i < nlvl:
                for j in xrange(nlvl-i):
                  modules.pop()
                modules[-1] = module
              else:
                raise InvalidFileFormat

            else:
              # Actual parameter
              cpl_solver_args += " --{}.{}".format(".".join(modules), __stripper.sub("=", s))

    except Exception as e:
      raise RuntimeError("Solver parameters could not be read from "+solver_params_file + ".\n" + str(e))

  return cpl_solver_args
