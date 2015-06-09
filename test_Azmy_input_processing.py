from dolfin.cpp.function import Function
from common import *
#set_log_level(DEBUG)

#------------------------------------------------------------------------------#
#                       LOAD CMDLINE ARGUMENTS                                 #
#------------------------------------------------------------------------------#

from argparse import ArgumentParser

parser = ArgumentParser(description='Description.')
parser.add_argument('problem_name',
                    help="Problem name. Defines the subfolder in the 'problems' folder serving as the parent folder "
                         "for all problem data and results.")
parser.add_argument('-m', '--mesh', type=str, default="",
                    help='Mesh base filename. If not specified, mesh_base_filename = problem_name. If Python module '
                         '<mesh_base_filename>.py is found in the problem folder, mesh data is imported from it. '
                         'Otherwise, mesh data is read from <mesh_base_filename>.xml, '
                         '<mesh_base_filename>_physical_region.xml, <mesh_base_filename>_facet_region.xml and optional '
                         'reg/mat/bnd_names.txt.')
parser.add_argument('-pap', '--print_actual_parameters', action="store_true",
                    help='Print actually set control parameters.')
parser.add_argument('-v', '--verbosity', type=int, choices=range(6), default=0,
                    help='Output verbosity.')
parser.add_argument('-N', '--SN_order', type=check_sn_order, default=0,
                    help='SN order (positive even number <= 32; if 0, diffusion will be used).')

args, additional_args = parser.parse_known_args()

if args.problem_name != 'Azmy':
  warning("For the testing purposes, forcing problem_name 'Azmy'")
  args.problem_name = 'Azmy'

#------------------------------------------------------------------------------#
#                   SETUP DOLFIN PARAMETERS DATABASE                           #
#------------------------------------------------------------------------------#

import parameters_processing

parameters_processing.set_default_dolfin_parameters()
parameters_processing.add_solver_parameters()

# parse PETSc/SLEPc and additional eigensolver parameters
# NOTE: this must be performed before the first PETScObject is initialized, otherwise
#       PETSc/SLEPc parameters will not be used
if additional_args:
  additional_args.insert(0, 'dummy')  # 0-th argv, i.e. 'name of the program'
  try:
    dolfin.parameters.parse(additional_args)
  except Exception as e:
    coupled_solver_error(__file__,
                         "setup Dolfin",
                         "Wrong PETSc/SLEPc/eigensolver arguments:\n{}\n{}".format(additional_args, e))

if args.print_actual_parameters:
  parameters_processing.print_info()

#------------------------------------------------------------------------------#
#                              IMPORT MODULES                                  #
#------------------------------------------------------------------------------#

from discretization_modules.discretizations import SNDiscretization as Discretization
from problem_data import ProblemData

# =============================================  S T A R T  ============================================================


#=======================================================================================================================
# TEST 0 - basic test with GMSH mesh files (specified on command line)
#
t_init = dolfin_common.Timer("Init 0 ")

PRD = ProblemData(args.problem_name, args.mesh, args.verbosity)
DD = Discretization(PRD, args.SN_order, args.verbosity)

t_init.stop()

init_timings_table = dolfin_common.timings(True)
print_timings( init_timings_table, args.verbosity > 0,
               os.path.join(PRD.out_folder, "timings0.txt"), os.path.join(PRD.out_folder, "timings0.tex") )

PRD.core.info()
PRD.bc.info()

assert PRD.G == 1
assert PRD.scattering_order == 1

DD.print_diagnostics()
DD.visualize_mesh_data()

St_fun = Function(DD.V0)
D_fun = Function(DD.V0)
Ss_fun = Function(DD.V0)
C_fun = Function(DD.V0)
Q_fun = Function(DD.V0)

PRD.get('St', D_fun, vis=True)
PRD.get('D', St_fun, vis=True)
PRD.get('Ss', Ss_fun, vis=True)
PRD.get('C', C_fun, vis=True)
PRD.get('Q', Q_fun, vis=True)

# Assert D is computed correctly

# Local
assert D_fun.vector().array() == 1/St_fun.vector().array()

# Global
D0 = D_fun.vector().gather_on_zero()
St0 = St_fun.vector().gather_on_zero()
assert D0 == 1/St0

# Assert C is computed correctly

# Local
Ss_array = Ss_fun.vector().array()
St_array = St_fun.vector().array()
assert C_fun.vector().array() == Ss_array/(4*numpy.pi * St_array(St_array - Ss_array))

# Assert that no fission is present

assert PRD.get('nSf', None, vis=True) == False

# Assert correct source distribution

Q_array = Q_fun.vector().array()
M2_dofs = DD.local_cell_dof_map(numpy.in1d(DD.cell_regions, PRD.matname_reg_map['M2']))
assert M2_dofs.size == 0 or numpy.all(Q_array[M2_dofs] == 0)

M1_dofs = DD.local_cell_dof_map(numpy.in1d(DD.cell_regions, PRD.matname_reg_map['M1']))
assert M2_dofs.size == 0 or numpy.all(Q_array[M1_dofs] == 1/(4*numpy.pi))

#=======================================================================================================================
# TEST 1-3 - test mesh data imported from mesh modules
#

def test_mesh_module(idx):
  t_init = dolfin_common.Timer("Init {}".format(idx))

  PRD = ProblemData(args.problem_name, "mesh{}".format(idx), args.verbosity)
  DD = Discretization(PRD, args.SN_order, args.verbosity)

  t_init.stop()

  init_timings_table = dolfin_common.timings(True)
  print_timings( init_timings_table, args.verbosity > 0,
                 os.path.join(PRD.out_folder, "timings{}.txt".format(idx)))

  PRD.core.info()
  PRD.bc.info()

  assert PRD.G == 1
  assert PRD.scattering_order == 1

  DD.print_diagnostics()
  DD.visualize_mesh_data()

  St_fun = Function(DD.V0)
  D_fun = Function(DD.V0)
  Ss_fun = Function(DD.V0)
  C_fun = Function(DD.V0)
  Q_fun = Function(DD.V0)

  PRD.get('St', D_fun, vis=True)
  PRD.get('D', St_fun, vis=True)
  PRD.get('Ss', Ss_fun, vis=True)
  PRD.get('C', C_fun, vis=True)
  PRD.get('Q', Q_fun, vis=True)

  assert PRD.get('nSf', None, vis=True) == False

  # Assert correct source distribution

  Q_array = Q_fun.vector().array()
  M2_dofs = DD.local_cell_dof_map(numpy.in1d(DD.cell_regions, PRD.matname_reg_map['M2']))
  assert M2_dofs.size == 0 or numpy.all(Q_array[M2_dofs] == 0)

  M1_dofs = DD.local_cell_dof_map(numpy.in1d(DD.cell_regions, PRD.matname_reg_map['M1']))
  assert M2_dofs.size == 0 or numpy.all(Q_array[M1_dofs] == 1/(4*numpy.pi))

  return PRD.bc.vacuum_boundaries, PRD.bc.reflective_boundaries, PRD.bc.incoming_fluxes

#-----------------------------------------------------------------------------------------------------------------------

vac, ref, inc = test_mesh_module(1)
assert (vac, ref, inc == test_mesh_module(2))
assert (vac, ref, inc == test_mesh_module(3))
