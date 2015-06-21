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
parser.add_argument('-c', '--core', type=str, default="",
                    help='Optional core specification file. Specifies boundary condition types, power normalization '
                         'for keff eigenvalue calcs. and axial layering for some axial output quantities. Not all '
                         'parts are needed - e.g. only boundary conditions may be specified in the file.')
parser.add_argument('-m', '--mesh', type=str, default="",
                    help='Mesh base filename. If not specified, mesh_base_filename = problem_name. If Python module '
                         '<mesh_base_filename>.py is found in the problem folder, mesh data is imported from it. '
                         'Otherwise, mesh data is read from <mesh_base_filename>.xml, '
                         '<mesh_base_filename>_physical_region.xml, <mesh_base_filename>_facet_region.xml and optional '
                         '{reg,mat,bnd}_names.txt.')
parser.add_argument('-pap', '--print_actual_parameters', action="store_true",
                    help='Print actually set control parameters.')
parser.add_argument('-v', '--verbosity', type=int, choices=range(6), default=0,
                    help='Output verbosity.')
parser.add_argument('-N', '--SN_order', type=check_sn_order, default=0,
                    help='SN order (positive even number <= 32.')

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
print "======================"
print "=       TEST 0       ="
print "======================"
t_init = dolfin_common.Timer("Init 0 ")

PD = ProblemData(args.problem_name, args.core, args.mesh, args.verbosity)
DD = Discretization(PD, args.SN_order, args.verbosity)
PD.distribute_material_data(DD.cell_regions, DD.M)
PD.distribute_boundary_fluxes()

t_init.stop()

init_timings_table = dolfin_common.timings(True)
print_timings( init_timings_table, args.verbosity > 0,
               os.path.join(PD.out_folder, "timings0.txt"), os.path.join(PD.out_folder, "timings0.tex") )

PD.core.info()
PD.bc.info()

assert PD.G == 1
assert PD.scattering_order == 1

DD.print_diagnostics()
DD.visualize_mesh_data()

St_fun = Function(DD.V0)
D_fun = Function(DD.V0)
Ss_fun = Function(DD.V0)
C_fun = Function(DD.V0)
Q_fun = Function(DD.V0)

PD.get_xs('St', St_fun, vis=True)
PD.get_xs('D', D_fun, vis=True)
PD.get_xs('Ss', Ss_fun, vis=True)
PD.get_xs('C', C_fun, vis=True)
PD.get_Q(Q_fun, numpy.random.randint(0,DD.M), vis=True)

# Assert D is computed correctly

# Local
assert numpy.allclose(D_fun.vector().array(), 1/St_fun.vector().array())

# Global
D0 = D_fun.vector().gather_on_zero()
St0 = St_fun.vector().gather_on_zero()
assert numpy.allclose(D0,1/St0)

# Assert C is computed correctly (no scattering in the material data => Ss_array = [0,...,0], C_array = [0,...,0])

# Local
Ss_array = Ss_fun.vector().array()
St_array = St_fun.vector().array()
assert numpy.allclose(C_fun.vector().array(), Ss_array / (4*numpy.pi * St_array * (St_array - Ss_array)))

# Assert that no fission is present

assert PD.get_xs('nSf', None, vis=True) == False

# Assert correct source distribution

Q_array = Q_fun.vector().array()
M2_dofs = DD.local_cell_dof_map[numpy.in1d(DD.cell_regions, PD.matname_reg_map['M2'])]
# noinspection PyTypeChecker
assert M2_dofs.size == 0 or numpy.all(Q_array[M2_dofs] == 0)

M1_dofs = DD.local_cell_dof_map[numpy.in1d(DD.cell_regions, PD.matname_reg_map['M1'])]
# noinspection PyTypeChecker
assert M2_dofs.size == 0 or numpy.allclose(Q_array[M1_dofs], 1/(4*numpy.pi))

#=======================================================================================================================
# TEST 1-4 - test mesh data imported from mesh modules
#

# noinspection PyTypeChecker
def test_mesh_module(idx, core_spec=""):
  print "======================"
  print "=       TEST {}      =".format(idx)
  print "======================"
  t_init = dolfin_common.Timer("Init {}".format(idx))

  pd = ProblemData(args.problem_name, core_spec, "mesh{}".format(idx), args.verbosity)
  dd = Discretization(pd, args.SN_order, args.verbosity)
  pd.distribute_material_data(dd.cell_regions, dd.M)
  pd.distribute_boundary_fluxes()

  t_init.stop()

  init_timings_table = dolfin_common.timings(True)
  print_timings( init_timings_table, args.verbosity > 0,
                 os.path.join(pd.out_folder, "timings{}.txt".format(idx)))

  pd.core.info()
  pd.bc.info()

  assert pd.G == 1
  assert pd.scattering_order == 1

  dd.print_diagnostics()
  dd.visualize_mesh_data()

  St = Function(dd.V0)
  D = Function(dd.V0)
  Ss = Function(dd.V0)
  C = Function(dd.V0)
  Q = Function(dd.V0)

  pd.get_xs('St', St, vis=True)
  pd.get_xs('D', D, vis=True)
  pd.get_xs('Ss', Ss, vis=True)
  pd.get_xs('C', C, vis=True)
  pd.get_Q(Q, numpy.random.randint(0,dd.M), vis=True)

  assert pd.get_xs('nSf', None, vis=True) == False

  # Assert correct source distribution

  Q_array = Q.vector().array()
  M2_dofs = dd.local_cell_dof_map[numpy.in1d(dd.cell_regions, pd.matname_reg_map['M2'])]
  assert M2_dofs.size == 0 or numpy.all(Q_array[M2_dofs] == 0)

  M1_dofs = dd.local_cell_dof_map[numpy.in1d(dd.cell_regions, pd.matname_reg_map['M1'])]
  assert M1_dofs.size == 0 or numpy.allclose(Q_array[M1_dofs], 1/(4*numpy.pi))

  return pd.bc

#-----------------------------------------------------------------------------------------------------------------------

bc1 = test_mesh_module(1)
bc2 = test_mesh_module(2)
bc3 = test_mesh_module(3, "core.dat")
bc4 = test_mesh_module(4)
bc5 = test_mesh_module(5, "core-outf.dat")

assert bc2.vacuum_boundaries == bc1.vacuum_boundaries and \
       bc2.reflective_boundaries == bc1.reflective_boundaries and \
       bc2.incoming_fluxes == bc1.incoming_fluxes
assert bc3.vacuum_boundaries == bc1.vacuum_boundaries and \
       bc3.reflective_boundaries == bc1.reflective_boundaries and \
       bc3.incoming_fluxes == bc1.incoming_fluxes
assert bc4.all_vacuum()

#=======================================================================================================================
# TEST 5 - test adjoint mesh data
#

idx = 5

print "======================"
print "=       TEST {}      =".format(idx)
print "======================"
t_init = dolfin_common.Timer("Init {}".format(idx))

pd = ProblemData(args.problem_name, "core-outf.dat", "mesh{}".format(idx), args.verbosity)
dd = Discretization(pd, args.SN_order, args.verbosity)
pd.distribute_material_data(dd.cell_regions, dd.M)
pd.distribute_boundary_fluxes()

t_init.stop()

init_timings_table = dolfin_common.timings(True)
print_timings( init_timings_table, args.verbosity > 0,
               os.path.join(pd.out_folder, "timings{}.txt".format(idx)))

pd.core.info()
pd.bc.info()

assert pd.G == 1
assert pd.scattering_order == 1

dd.print_diagnostics()
dd.visualize_mesh_data()

Q = Function(dd.V0)
Qa = Function(dd.V0)

pd.get_Q(Q, numpy.random.randint(0,dd.M), adjoint=False, vis=True)
pd.get_Q(Qa, numpy.random.randint(0,dd.M), adjoint=True, vis=True)

# Assert correct source distribution

Q_array = Q.vector().array()
Qa_array = Qa.vector().array()

M2_dofs = dd.local_cell_dof_map[numpy.in1d(dd.cell_regions, pd.matname_reg_map['M2'])]
assert M2_dofs.size == 0 or numpy.all(Q_array[M2_dofs] == 0)
assert M2_dofs.size == 0 or numpy.all(Qa_array[M2_dofs] == 0)

M1_dofs = dd.local_cell_dof_map[numpy.in1d(dd.cell_regions, pd.matname_reg_map['M1'])]
assert M1_dofs.size == 0 or numpy.allclose(Q_array[M1_dofs], 1/(4*numpy.pi))
assert M1_dofs.size == 0 or numpy.all(Qa_array[M1_dofs] == 0)

D_dofs = dd.local_cell_dof_map[numpy.in1d(dd.cell_regions, pd.matname_reg_map['D'])]
assert D_dofs.size == 0 or numpy.allclose(Qa_array[D_dofs], 1.9/(4*numpy.pi))
assert D_dofs.size == 0 or numpy.all(Q_array[D_dofs] == 0)

F = Function(dd.V0)
G = Function(dd.V0)

pd.get_FG(F, G, numpy.random.randint(0,dd.M), vis=True)

# Assert correct source distribution

F_array = F.vector().array()
G_array = G.vector().array()

assert numpy.all(F_array == Q_array + Qa_array)
assert numpy.all(G_array == Q_array - Qa_array)