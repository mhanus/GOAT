"""
Created on 7.5.2014

@author: mhanus
"""
# TODO: try-except for dolfin imports

# TODO: Add capability of saving results every n-th iteration of an outer (adaptivity) loop

import os, numpy

from dolfin.cpp.io import File
from dolfin.functions import TestFunction, TrialFunction
from dolfin.cpp.common import Timer, warning, MPI
from dolfin import Function, interpolate, assemble, norm, parameters
from dolfin.cpp.la import PETScMatrix, PETScVector, as_backend_type, PETScOptions
from dolfin.cpp.common import Parameters
from dolfin import solve as dolfin_solve
from ufl import dx

from common import  comm, print0

if MPI.rank(comm) == 0:
  from scipy.io import savemat

from flux_modules import backend_ext_module



# noinspection PyArgumentList
def get_parameters():
  """
  Create a record in the Dolfin parameters database that stores generic runtime parameters for flux modules.
  :return: Parameters
  """
  params = Parameters("flux_module")

  params.add(backend_ext_module.GeneralizedEigenSolver.default_parameters())
  params["eigensolver"].add("adaptive_eig_tol_start", 1e-3)

  params.add(
    Parameters(
      "visualization",
      flux = 1,
      cell_powers = 0
    )
  )
  params.add(
    Parameters(
      "saving",
      results = 0,
      debug = 0,
      algebraic_system = 0
    )
  )
  return params

#------------------------------------------------------------------------------#
#                                MATRIX SAVING                                 #
#------------------------------------------------------------------------------#

def __coo_rep_on_zero_internal(COO, rows_glob=None, cols_glob=None, vals_glob=None, sym=False):
  # TODO: Remove the following two lines once we find out how to make mpi4py work in conjunction with Dolfin
  if MPI.size(comm) > 1:
    return rows_glob, cols_glob, vals_glob
  
  rows = COO.get_rows()  
  cols = COO.get_cols()
  vals = COO.get_vals()
  local_nnz = COO.get_local_nnz()
    
  recv_counts = comm.allgather(local_nnz)
  
  if MPI.rank(comm) == 0 and not (rows_glob and cols_glob and vals_glob):
    nnz = int(numpy.sum(recv_counts))
    rows_glob = numpy.empty(nnz, dtype=rows.dtype)
    cols_glob = numpy.empty(nnz, dtype=cols.dtype)
    vals_glob = numpy.empty(nnz, dtype=vals.dtype)

  # TODO: this is just a stub so that the following doesn't mess code inspection
  MPI_type = lambda x : x
  MPI_IDX_T = MPI_type(rows)
  MPI_VAL_T = MPI_type(vals)
  
  recv_displs = None
  comm.Gatherv(sendbuf=[rows, MPI_IDX_T], 
               recvbuf=[rows_glob, (recv_counts, recv_displs), MPI_IDX_T], 
               root=0)  
  comm.Gatherv(sendbuf=[cols, MPI_IDX_T], 
               recvbuf=[cols_glob, (recv_counts, recv_displs), MPI_IDX_T], 
               root=0)
  comm.Gatherv(sendbuf=[vals, MPI_VAL_T], 
               recvbuf=[vals_glob, (recv_counts, recv_displs), MPI_VAL_T], 
               root=0)

  return rows_glob, cols_glob, vals_glob

#@profile
def coo_rep_on_zero(A, rows_glob=None, cols_glob=None, vals_glob=None, sym=False):
  """ COO representation of matrix A on rank 0.

    :return:
      rank 0:        rows, cols, vals
      rank 1,2,... : None, None, None
    :rtype: (ndarray, ndarray, ndarray)
  """
  
  timer = Timer("COO representation")

  # noinspection PyBroadException
  try:  # DOLFIN 1.4+
    Acomp = PETScMatrix()
    A.copy().compressed(Acomp)
  except:
    Acomp = A   # Don't compress
      
  COO = backend_ext_module.COO(Acomp)

  return __coo_rep_on_zero_internal(COO, rows_glob, cols_glob, vals_glob, sym)


#------------------------------------------------------------------------------#
#                             MAIN FLUX MODULE                                 #
#------------------------------------------------------------------------------#

class FluxModule(object):
  """
  Module responsible for calculating scalar fluxes and keff eigenvalue
  """

  def __init__(self, PD, DD, verbosity):
    """
    Constructor
    :param ProblemData PD: Problem information and various mesh-region <-> xs-material mappings
    :param Discretization DD: Discretization data
    :param int verbosity: Verbosity level.
    """

    super(FluxModule, self).__init__()

    self.verb = verbosity
    self.print_prefix = ""

    self.mat_file_name = dict()

    self.PD = PD
    self.DD = DD
    self.BC = PD.bc

    try:
      self.fixed_source_problem = PD.fixed_source_problem
      self.eigenproblem = PD.eigenproblem
    except AttributeError:
      PD.distribute_material_data(DD.cell_regions, DD.M)

    self.A = PETScMatrix()

    # unused in case of an eigenvalue problem
    self.Q = PETScVector()

    # used only for saving the algebraic system
    self.rows_A = None
    self.cols_A = None
    self.vals_A = None
    self.rows_B = None
    self.cols_B = None
    self.vals_B = None

    if self.fixed_source_problem:
      self.src_F = Function(self.DD.V0)
      self.src_G = Function(self.DD.V0)
      self.vals_Q = None

    # multigroup scalar fluxes
    self.phi_mg = []
    for g in range(self.DD.G):
      phig = Function(self.DD.Vphi1)
      phig.rename("phi","phi_g{}".format(g))
      self.phi_mg.append(phig)

    self.sln = Function(DD.V)
    self.sln_vec = as_backend_type(self.sln.vector())
    self.local_sln_size = self.sln_vec.local_size()

    # auxiliary function for storing various DG(0) quantities (cross sections, group-integrated reaction rates, etc.)
    self.R = Function(self.DD.V0)

    # fission spectrum
    if 'chi' in self.PD.used_xs:
      self.chi = Function(self.DD.V0)
    else:
      self.chi = None

    if 'eSf' in self.PD.used_xs:
      self.E = numpy.zeros(self.DD.local_ndof0)
    else:
      self.E = None

    self.up_to_date = {"flux" : False, "cell_powers" : False}

    self.bnd_matrix_form = None

    self.parameters = parameters["flux_module"]

    if self.eigenproblem:
      assert self.parameters.has_parameter_set("eigensolver")
      self.eigen_params = self.parameters["eigensolver"]
      self.adaptive_eig_tol_end = 0

      self.B = PETScMatrix()
      self.keff = 1
      self.prev_keff = self.keff
      self.set_initial_approximation(numpy.random.random(self.local_sln_size))
      self.update_phi()

    self.u = TrialFunction(self.DD.V)
    self.v = TestFunction(self.DD.V)

    self.v0 = TestFunction(self.DD.V0)
    self.phig = Function(self.DD.Vphi1) # single group scalar flux
    self.cell_RR_form = self.R * self.phig * self.v0 * dx
    self._cell_RRg_vector = PETScVector()

    self.vis_folder = os.path.join(self.PD.out_folder, "FLUX")

    self.vis_files = dict()
    var = "cell_powers"
    self.vis_files[var] = File(os.path.join(self.vis_folder, var+".pvd"), "compressed")
    var = "flux"
    self.vis_files[var] = [
      File(os.path.join(self.vis_folder, "{}_g{}.pvd".format(var, g)), "compressed") for g in range(self.DD.G)
    ]

    variables = self.parameters["saving"].iterkeys()
    self.save_folder = { k : os.path.join(self.PD.out_folder, k.upper()) for k in variables }

  # noinspection PyTypeChecker
  def save_algebraic_system(self, mat_file_name=None, it=0):
    if not mat_file_name: mat_file_name = {}

    try:
      should_save = divmod(it, self.parameters["saving"]["algebraic_system"])[1] == 0
    except ZeroDivisionError:
      should_save = False

    if not should_save:
      return

    if self.verb > 1: print0("Saving the created matrices.")

    timer = Timer("COO representation + matrix saving")

    self.rows_A, self.cols_A, self.vals_A = coo_rep_on_zero(self.A, self.rows_A, self.cols_A, self.vals_A)

    if self.eigenproblem:
      self.rows_B, self.cols_B, self.vals_B = coo_rep_on_zero(self.B, self.rows_B, self.cols_B, self.vals_B)
    elif self.fixed_source_problem:
      self.vals_Q = self.Q.gather_on_zero()

    if MPI.rank(comm) == 0:
      if not mat_file_name:
        mat_file_name['A'] = 'A'
        if self.eigenproblem: mat_file_name['B'] = 'B'
        if self.fixed_source_problem: mat_file_name['Q'] = 'Q'

      for k,v in mat_file_name.iteritems():
        mat_file_name[k] = os.path.join(self.save_folder["algebraic_system"], v)

      if self.verb > 2: print0( self.print_prefix + "  Saving A to " + mat_file_name['A']+'.mat' )
      savemat(mat_file_name['A']+'.mat',
              { 'rows':numpy.asarray(self.rows_A, dtype='d'),
                'cols':numpy.asarray(self.cols_A, dtype='d'),
                'vals':self.vals_A },
              do_compression=True)

      if self.eigenproblem:
        if self.verb > 2: print0( self.print_prefix + "  Saving B to " + mat_file_name['B']+'.mat' )
        savemat(mat_file_name['B']+'.mat',
                { 'rows':numpy.asarray(self.rows_B, dtype='d'),
                  'cols':numpy.asarray(self.cols_B, dtype='d'),
                  'vals':self.vals_B },
                do_compression=True)

      elif self.fixed_source_problem:
        if self.verb > 2: print0( self.print_prefix + "  Saving Q to " + mat_file_name['Q']+'.mat' )
        savemat(mat_file_name['Q']+'.mat',
                { 'vals':self.vals_Q },
                do_compression=True)

  def update(self, it):
    # Handle adaptive eigensolver convergence tolerance strengthening

    if self.eigen_params["adaptive_eig_tol_start"] > 0:
      if self.adaptive_eig_tol_end == 0:
        # First iteration - set the adaptive tolerance strengthening
        if self.verb > 0:
          s = "Eigenvalue tolerance will be decreased adaptively from {} to {} during adaptivity iterations"
          print0(s.format(self.eigen_params["adaptive_eig_tol_start"], self.eigen_params["tol"]))

        self.adaptive_eig_tol_end = self.eigen_params["tol"]
        self.eigen_params["tol"] = self.eigen_params["adaptive_eig_tol_start"]
      else:
        # Subsequent iterations - strengthen the eigensolver convergence tolerance each adaptivity step until
        # adaptive_eig_tol_end is reached
        if self.eigen_params["tol"] > self.adaptive_eig_tol_end:
          self.eigen_params["tol"] /= 10

  def update_phi(self):
    raise NotImplementedError("Abstract scalar flux update method -- must be overriden in specific flux modules.")

  def get_dg0_fluxes(self):
    """
    Get flux interpolated at DG(0) dofs.
    :return: List of `G` arrays of group-fluxes at local DG(0) dofs.
    :rtype: list[ndarray]
    """
    dg0_fluxes = list()

    for g in xrange(self.DD.G):
      dg0_fluxes.append(interpolate(self.phig, self.DD.V0).vector().get_local())

    return dg0_fluxes

  def visualize(self, it=0):
    var = "cell_powers"
    try:
      should_vis = divmod(it, self.parameters["visualization"][var])[1] == 0
    except ZeroDivisionError:
      should_vis = False

    if should_vis:
      if not self.up_to_date[var]:
        self.calculate_cell_powers()
      else:
        qfun = Function(self.DD.V0)
        qfun.vector()[:] = self.E
        qfun.rename("q", "Power")
        self.vis_files["cell_powers"] << (qfun, float(it))

    var = "flux"
    try:
      should_vis = divmod(it, self.parameters["visualization"][var])[1] == 0
    except ZeroDivisionError:
      should_vis = False

    if should_vis:
      if not self.up_to_date[var]:
        self.update_phi()

      for g in xrange(self.DD.G):
        self.vis_files["flux"][g] << (self.phi_mg[g], float(it))

  def print_results(self):
    if self.verb > 2:
      if self.eigenproblem:
        print0(self.print_prefix + "keff = {}".format(self.keff))
        print0(self.print_prefix + "Residual norm: {}".format(self.residual_norm()))

  def eigenvalue_residual_norm(self, norm_type='l2'):
    r = PETScVector()
    y = PETScVector()

    self.B.mult(self.sln_vec, r)
    self.A.mult(self.sln_vec, y)
    r.apply("insert")
    y.apply("insert")

    r -= self.keff * y

    return norm(r, norm_type)

  def fixed_source_residual_norm(self, norm_type='l2'):
    y = PETScVector()
    self.A.mult(self.sln_vec, y)
    y.apply("insert")
    return norm(self.Q - y, norm_type)

  def residual_norm(self, norm_type='l2'):
    if self.eigenproblem:
      return self.eigenvalue_residual_norm(norm_type)
    else:
      return self.fixed_source_residual_norm(norm_type)

  # FIXME: It doesn't work to define eigensolver only once in the c'tor (we use pointers to matrices, so it should...).
  def solve_keff(self, it=0):
    assert self.A
    assert self.B
    assert self.sln_vec

    self.prev_keff = self.keff

    eigensolver = backend_ext_module.GeneralizedEigenSolver(self.A, self.B)

    if eigensolver.parameters["adaptive_shifting"]:
      eigensolver.set_shift_AB(1. / self.keff)

    if eigensolver.parameters["inner_solver_adaptive_tol_multiplier"] > 0:
      PETScOptions.set("st_ksp_atol",
                       eigensolver.parameters["inner_solver_adaptive_tol_multiplier"] * self.eigenvalue_residual_norm())

    if self.verb > 1: print0(self.print_prefix + "Solving ({})...".format(eigensolver.get_actual_problem_description()))
    solution_timer = Timer("Solver")

    eigensolver.set_initial_space(self.sln_vec)

    eigensolver.solve()

    self.keff = 1. / eigensolver.get_first_eigenpair_AB(self.sln_vec)

    # This is needed in parallel (why not in serial?)
    self.sln.vector()[:] = self.sln_vec.array()

    solution_timer.stop()

    if MPI.rank(comm) == 0:
      if self.verb > 1:
        print "\n" + self.print_prefix + "keff = {}\n".format(self.keff)

      try:
        should_save = divmod(it, self.parameters["saving"]["results"])[1] == 0
      except ZeroDivisionError:
        should_save = False

      if should_save:
        savemat(os.path.join(self.save_folder["results"], "eigensolver_out.mat"),
                {'x': self.sln_vec.array(), 'keff': self.keff})

  def solve_fixed_source(self, it=0):
    assert self.A
    assert self.Q

    dolfin_solve(self.A, self.sln_vec, self.Q, "gmres", "petsc_amg")

  def assemble_algebraic_system(self):
    raise NotImplemented

  def solve(self, it=0):
    """
    Pick the appropriate solver for current problem (eigen/fixed-source) and solve the problem (i.e., update solution
    vector and possibly the eigenvalue
    ).
    """
    self.assemble_algebraic_system()
    self.save_algebraic_system(it)

    if self.eigenproblem:
      self.solve_keff(it)
    else:
      self.solve_fixed_source(it)

    self.up_to_date = {k : False for k in self.up_to_date.iterkeys()}

  def set_initial_approximation(self, x0):
    self.sln_vec[:] = x0

  def calculate_cell_powers(self):
    """
    Calculates cell-integrated powers (sets :attr:`E`). Also performs normalization to the specified core(-fraction)
    power (:attr:`core.power`).

    Note that the array is ordered by the associated DG(0) dof, not by the cell index in the mesh.
    """
    q_calc_timer = Timer("FM: Calculation of cell powers")
    ass_timer = Timer("FM: Assemble cell powers")

    if self.verb > 2:
      print0(self.print_prefix + "  calculating cell-wise powers.")

    self.E.fill(0)

    if not self.up_to_date["flux"]:
      self.update_phi()

    for g in xrange(self.DD.G):
      self.PD.get_xs('eSf', self.R, g)

      ass_timer.start()
      self.phig.assign(self.phi_mg[g])
      assemble(self.cell_RR_form, tensor=self._cell_RRg_vector)
      ass_timer.stop()

      self.E += self._cell_RRg_vector.get_local()

    self.up_to_date["cell_powers"] = True
    self.normalize_cell_powers()

  def normalize_cell_powers(self):
    assert self.up_to_date["cell_powers"]

    P = MPI.sum(numpy.sum(self.E))

    if self.verb > 2:
      print0(self.print_prefix + "  desired power: " + str(self.PD.core.power))
      print0(self.print_prefix + "  actual power: " + str(P))
      print0("")

    assert (P > 0)

    self.E *= self.PD.core.power / P

  # noinspection PyAttributeOutsideInit
  def calculate_cell_reaction_rate(self, reaction_xs, rr_vect=None, return_xs_arrays=False):
    """
    Calculates cell-integrated reaction rate and optionally returns the xs's needed for the calculation.

    Note that the array is ordered by the associated DG(0) dof, not by the cell index in the mesh.

    :param str reaction_xs: Reaction cross-section id.
    :param ndarray rr_vect: (optional) Output vector. If not given, reaction rate vector of the FluxModule class
      (:attr:`cell_RR`) will be updated. Must be pre-allocated to store :attr:`Discretization.local_ndof0`
      elements, but doesn't need to be pre-set to any value.
    :param bool return_xs_arrays: (optional) If True, a list that contains for each group the cell (dof) values array
      of the DG(0) representation of the specified reaction xs will be created and returned.
    :return: List with xs value arrays for each group if `return_xs_arrays == True`, None otherwise (see above)
    :rtype: list[ndarray] | None
    """
    if reaction_xs not in self.PD.used_xs:
      warning("Attempted to calculate cell-wise reaction rate for reaction without loaded cross-section (skipping).")
      return

    if self.verb > 1: print0(self.print_prefix + "Calculating cell-wise '{}' reaction rate.".format(reaction_xs))

    if not rr_vect:
      try:
        self.cell_RR
      except AttributeError:
        self.cell_RR = numpy.zeros(self.DD.local_ndof0)
      finally:
        rr_vect = self.cell_RR
    else:
      assert rr_vect.size == self.DD.local_ndof0

    if return_xs_arrays:
      xs_arrays = [None] * self.DD.G
    else:
      xs_arrays = None

    rr_vect.fill(0)

    if not self.up_to_date["cell_powers"]:
      self.update_phi()

    calc_timer = Timer("FM: Calculation of '{}' reaction rate".format(reaction_xs))
    ass_timer = Timer("FM: Assemble '{}' reaction rate".format(reaction_xs))

    for g in xrange(self.DD.G):
      self.PD.get_xs(reaction_xs, self.R, g)
      if xs_arrays:
        xs_arrays[g] = self.R.vector().get_local().copy()

      ass_timer.start()
      self.phig.assign(self.phi_mg[g])
      assemble(self.cell_RR_form, tensor=self._cell_RRg_vector)
      ass_timer.stop()

      rr_vect += self._cell_RRg_vector.get_local()

    return xs_arrays