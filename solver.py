"""
Created on 7.5.2014

@author: mhanus
"""
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
                         '<mesh_base_filename>_physical_region.xml, <mesh_base_filename>_facet_region.xml and '
                         'reg/mat/bnd_names.txt.')
parser.add_argument('-pap', '--print_actual_parameters', action="store_true",
                    help='Print actually set control parameters.')
parser.add_argument('-v', '--verbosity', type=int, choices=range(6), default=0,
                    help='Output verbosity.')
parser.add_argument('-N', '--SN_order', type=check_sn_order, default=0,
                    help='SN order (positive even number <= 32; if 0, diffusion will be used).')

args, additional_args = parser.parse_known_args()

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
from flux_modules.elliptic_sn_flux_module import EllipticSNFluxModule as FluxModule
from problem_data import ProblemData


# =============================================  S T A R T  ============================================================


#=======================================================================================================================
# MODULES INITIALIZATION - discretization, problem specs, materials, T/H feedback, flux solver
#
t_init = dolfin_common.Timer("! Complete initialization phase")

if args.verbosity > 1: print "Process {0}: initializing...".format(comm.rank)


# Core calculation modules

PRD = ProblemData(args.problem_name, args.mesh, args.verbosity)
DD = Discretization(PRD, args.SN_order, args.verbosity)
FM = FluxModule(PRD, DD, args.verbosity)

t_init.stop()


#=======================================================================================================================
# SOLUTION
#

init_timings_table = dolfin_common.timings(True)
if args.verbosity > 0:
  print pid+"\n\n"+init_timings_table.str(True)+"\n"



#=======================================================================================================================
# FINAL TIMINGS
#
sln_timings_table = dolfin_common.timings(True)

sln_timings_table_str = pid + "\n\n" + sln_timings_table.str(True)
sln_timings_table_str = comm.gather(sln_timings_table_str, root=0)
sln_timings_table_tex = pid + "\n\n" + sln_timings_table.str_latex()
sln_timings_table_tex = comm.gather(sln_timings_table_tex, root=0)

init_timings_table_str = pid + "\n\n" + init_timings_table.str(True)
init_timings_table_str = comm.gather(init_timings_table_str, root=0)
init_timings_table_tex = pid + "\n\n" + init_timings_table.str_latex()
init_timings_table_tex = comm.gather(init_timings_table_tex, root=0)

if comm.rank == 0:
  sln_timings_table_str = "\n___________________________________________________________\n".join(sln_timings_table_str)
  sln_timings_table_tex = "\n___________________________________________________________\n".join(sln_timings_table_tex)
  init_timings_table_str= "\n___________________________________________________________\n".join(init_timings_table_str)
  init_timings_table_tex= "\n___________________________________________________________\n".join(init_timings_table_tex)

  print sln_timings_table_str

  try:
    with open(os.path.join(PRD.folder, "timings.txt"), "wt") as f:
      print>>f, init_timings_table_str
      print>>f, sln_timings_table_str
    with open(os.path.join(PRD.folder, "timings.tex"), "wt") as f:
      print>>f, init_timings_table_tex
      print>>f, sln_timings_table_tex
  except Exception as e:
    print "Writing final results failed: {}".format(e)