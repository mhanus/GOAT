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
t_init = dolfin_common.Timer("- Complete initialization phase")

if args.verbosity > 1: print "Process {}: initializing...".format(MPI.rank(comm))

# Core calculation modules

PD = ProblemData(args.problem_name, args.core, args.mesh, args.verbosity)
DD = Discretization(PD, args.SN_order, args.verbosity)
FM = FluxModule(PD, DD, args.verbosity)

t_init.stop()

init_timings_table = dolfin_common.timings(True)
print_timings(init_timings_table, args.verbosity > 0)

#=======================================================================================================================
# SOLUTION
#

FM.solve()
FM.print_results()
FM.visualize()

#=======================================================================================================================
# FINAL TIMINGS
#
sln_timings_table = dolfin_common.timings(True)
print_timings( init_timings_table,
               False,
               os.path.join(PD.out_folder, "init_timings.txt"), os.path.join(PD.out_folder, "init_timings.tex") )
print_timings( sln_timings_table,
               args.verbosity > 0,
               os.path.join(PD.out_folder, "sln_timings.txt"), os.path.join(PD.out_folder, "sln_timings.tex") )