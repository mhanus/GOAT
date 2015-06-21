"""
Created on 11.5.2014

@author: mhanus
"""
# TODO: try-except for dolfin imports
import os

from dolfin.cpp.common import Timer, Parameters,warning
from dolfin import assemble, Function, FacetNormal, split, project,parameters,FunctionSpace
from dolfin import solve as dolfin_solve
from dolfin.cpp.function import FunctionAssigner,assign
from dolfin.cpp.io import File
import ufl
from ufl.integral import Measure
from ufl.objects import dx
import numpy

from common import delta, print0, coupled_solver_error
from discretization_modules.discretizations import SNDiscretization
import flux_module


# noinspection PyArgumentList
def get_parameters():
  params = flux_module.get_parameters()
  params.add(
              Parameters(
                "group_GS",
                max_niter = -1,
                tol = 1e-6
              )
            )
  params["visualization"].add("angular_flux", 1)
  params["visualization"].add("adjoint_angular_flux", 1)
  return params 
  

class EllipticSNFluxModule(flux_module.FluxModule):  
  """
  Flux solver based on elliptic discrete ordinates formulation
  """
  
  class AngularTensors:
    def __init__(self, angular_quad, L):
      from transport_data import angular_tensors_ext_module

      i,j,k1,k2,p,q = ufl.indices(6)

      tensors = angular_tensors_ext_module.AngularTensors(angular_quad, L)
      self.Y = ufl.as_tensor( numpy.reshape(tensors.Y(), tensors.shape_Y()) )
      self.Q = ufl.as_tensor( numpy.reshape(tensors.Q(), tensors.shape_Q()) )
      self.QT = ufl.transpose(self.Q)
      self.Qt = ufl.as_tensor( numpy.reshape(tensors.Qt(), tensors.shape_Qt()) )
      self.QtT = ufl.as_tensor( self.Qt[k1,p,i], (p,i,k1) )
      self.G = ufl.as_tensor( numpy.reshape(tensors.G(), tensors.shape_G()) )
      self.T = ufl.as_tensor( numpy.reshape(tensors.T(), tensors.shape_T()) )
      self.Wp = ufl.as_vector( angular_quad.get_pw() )
      self.W = ufl.diag(self.Wp)
      
  def __init__(self, PD, DD, verbosity):
    """
    Constructor
    :param ProblemData PD: Problem information and various mesh-region <-> xs-material mappings
    :param SNDiscretization DD: Discretization data
    :param int verbosity: Verbosity level.
    """

    self.max_group_GS_it = parameters["flux_module"]["group_GS"]["max_niter"]
    self.group_GS = self.max_group_GS_it > 0  

    try:
      PD.eigenproblem
    except AttributeError:
      PD.distribute_material_data(DD.cell_regions, DD.M)

    if PD.eigenproblem and self.group_GS:
      print "Group Gauss-Seidel for eigenproblem not yet supported - switching to all-group coupled solution method."
      self.group_GS = False

    if DD.G == 1:
      self.group_GS = True
      self.max_group_GS_it = 1

    DD.init_solution_spaces(self.group_GS)

    super(EllipticSNFluxModule, self).__init__(PD, DD, verbosity)

    if PD.fixed_source_problem:
      self.vals_Q = numpy.empty(self.DD.ndof,dtype='float64')

    if self.verb > 1: print0("Defining coefficient functions and tensors")

    if self.DD.V is self.DD.Vpsi1 or DD.G == 1:
      # shallow copies of trial/test functions - allows unified treatment of both mixed/single versions
      self.u = [self.u]*self.DD.G
      self.v = [self.v]*self.DD.G

      self.slns_mg = [self.sln]
      for g in range(1, self.DD.G):
        self.slns_mg.append(Function(self.DD.Vpsi1))

    else:
      self.u = split(self.u)
      self.v = split(self.v)

      # self.group_assigner = []
      # for g in range(self.DD.G):
      #   self.group_assigner.append(FunctionAssigner(self.DD.V.sub(g), self.DD.Vpsi1))

    # auxiliary single-group angular fluxes (used for monitoring convergence of the group GS iteration and computing
    # the true forward/adjoint angular fluxes)
    self.aux_slng = Function(self.DD.Vpsi1)

    # multigroup angular fluxes
    self.psi_mg = []
    for g in range(self.DD.G):
      self.psi_mg.append(Function(self.DD.Vpsi1))

    # multigroup adjoint angular fluxes
    self.adj_psi_mg = []
    for g in range(self.DD.G):
      self.adj_psi_mg.append(Function(self.DD.Vpsi1))

    self.D = Function(self.DD.V0)

    self.L = PD.scattering_order-1
    self.tensors = self.AngularTensors(self.DD.angular_quad, self.L)

    self.C = numpy.empty(self.L+1, Function)
    self.S = numpy.empty(self.L+1, Function)
    for l in range(self.L+1):
      self.C[l] = Function(self.DD.V0)
      self.S[l] = Function(self.DD.V0)

    for var in {"angular_flux", "adjoint_angular_flux"}:
      self.vis_files[var] = \
      [
        [
          File(os.path.join(self.vis_folder, "{}_g{}_m{}.pvd".format(var,g,m)), "compressed") for m in range(self.DD.M)
        ]
        for g in range(self.DD.G)
      ]

    self.__define_boundary_terms()

  # TODO: Reflective boundary conditions  
  # noinspection PyAttributeOutsideInit,PyUnboundLocalVariable,PyTypeChecker
  def __define_boundary_terms(self):
    if self.verb > 2: print0("Defining boundary terms")

    self.bnd_vector_form = numpy.empty(self.DD.G, dtype=object)
    self.bnd_matrix_form = numpy.empty(self.DD.G, dtype=object)

    n = FacetNormal(self.DD.mesh)
    i,p,q = ufl.indices(3)
    
    natural_boundaries = self.BC.vacuum_boundaries.union(self.BC.incoming_fluxes.keys())

    nonzero = lambda x: numpy.all(x > 0)
    nonzero_inc_flux = any(map(nonzero, self.BC.incoming_fluxes.values()))

    if nonzero_inc_flux and not self.fixed_source_problem:
      coupled_solver_error(__file__,
                   "define boundary terms",
                   "Incoming flux specified for an eigenvalue problem"+\
                   "(Q must be given whenever phi_inc is; it may possibly be zero everywhere)")

    for g in range(self.DD.G):
      self.bnd_vector_form[g] = ufl.zero()
      self.bnd_matrix_form[g] = ufl.zero()

    if natural_boundaries:
      try:
        ds = Measure("ds")[self.DD.boundaries]
      except TypeError:
        coupled_solver_error(__file__,
                             "define boundary terms",
                             "File assigning boundary indices to facets required if vacuum or incoming flux boundaries "
                             "are specified.")
          
      for bnd_idx in natural_boundaries:
        # NOTE: The following doesn't work because ufl.abs requires a function
        #
        #for g in range(self.DD.G):
        #  self.bnd_matrix_form[g] += \
        #    abs(self.tensors.G[p,q,i]*n[i])*self.u[g][q]*self.v[g][p]*ds(bnd_idx)
        #
        # NOTE: Instead, the following explicit loop has to be used; this makes tensors.G unneccessary
        #
        for pp in range(self.DD.M):
          omega_p_dot_n = self.DD.ordinates_matrix[i,pp]*n[i]

          for g in range(self.DD.G):
            self.bnd_matrix_form[g] += \
              abs(omega_p_dot_n)*self.tensors.Wp[pp]*self.u[g][pp]*self.v[g][pp]*ds(bnd_idx)
    
      if nonzero_inc_flux:
        for pp in range(self.DD.M):
          omega_p_dot_n = self.DD.ordinates_matrix[i,pp]*n[i]
          
          for bnd_idx, psi_inc in self.BC.incoming_fluxes.iteritems():
              
            if psi_inc.shape != (self.DD.M, self.DD.G):
              coupled_solver_error(__file__,
                           "define boundary terms",
                           "Incoming flux with incorrect number of groups and directions specified: "+
                           "{}, expected ({}, {})".format(psi_inc.shape, self.DD.M, self.DD.G))
            
            
            for g in range(self.DD.G):
              self.bnd_vector_form[g] += \
                ufl.conditional(omega_p_dot_n < 0,
                                omega_p_dot_n*self.tensors.Wp[pp]*psi_inc[pp,g]*self.v[g][pp]*ds(bnd_idx),
                                ufl.zero())  # FIXME: This assumes zero adjoint outgoing flux
    else: # Apply vacuum b.c. everywhere
      ds = Measure("ds")

      for pp in range(self.DD.M):
        omega_p_dot_n = self.DD.ordinates_matrix[i,pp]*n[i]

        for g in range(self.DD.G):
          self.bnd_matrix_form[g] += abs(omega_p_dot_n)*self.tensors.Wp[pp]*self.u[g][pp]*self.v[g][pp]*ds()

  def solve_group_GS(self, it=0, init_slns_ary=None):
    if self.verb > 1: print0(self.print_prefix + "Solving..." )
  
    if self.eigenproblem:  
      coupled_solver_error(__file__,
                           "solve using group GS",
                           "Group Gauss-Seidel for eigenproblem not yet supported")
    
    sol_timer = Timer("-- Complete solution")
    mat_timer = Timer("---- MTX: Complete construction")
    ass_timer = Timer("---- MTX: Assembling")
    sln_timer = Timer("---- SOL: Solving")

    # To simplify the weak forms
    u = self.u[0]
    v = self.v[0]

    if init_slns_ary is None:
      init_slns_ary = numpy.zeros((self.DD.G, self.local_sln_size))

    for g in range(self.DD.G):
      self.slns_mg[g].vector()[:] = init_slns_ary[g]

    err = 0.

    for gsi in range(self.max_group_GS_it):
    #==========================================  GAUSS-SEIDEL LOOP  ============================================
    
      if self.verb > 2: print self.print_prefix + 'Gauss-Seidel iteration {}'.format(gsi)

      for gto in range(self.DD.G):
      #=========================================  LOOP OVER GROUPS  ===============================================

        self.sln_vec = self.slns_mg[gto].vector()

        prev_slng_vec = self.aux_slng.vector()
        prev_slng_vec.zero()
        prev_slng_vec.axpy(1.0, self.sln_vec)
        prev_slng_vec.apply("insert")

        spc = self.print_prefix + "  "
        
        if self.verb > 3 and self.DD.G > 1: print spc + 'GROUP [', gto, ',', gto, '] :'
        
        #====================================  ASSEMBLE WITHIN-GROUP PROBLEM  ========================================
        
        mat_timer.start()

        pres_fiss = self.PD.get_xs('chi', self.chi, gto)
    
        self.PD.get_xs('D', self.D, gto)
        self.PD.get_xs('St', self.R, gto)
        
        i,j,p,q,k1,k2 = ufl.indices(6)
        
        form = ( self.D*self.tensors.T[p,q,i,j]*u[q].dx(j)*v[p].dx(i) +
                 self.R*self.tensors.W[p,q]*u[q]*v[p] ) * dx + self.bnd_matrix_form[gto]
        
        ass_timer.start()
    
        add_values_A = False
        add_values_Q = False

        assemble(form, tensor=self.A, finalize_tensor=False, add_values=add_values_A)
        add_values_A = True
          
        ass_timer.stop()

        if self.fixed_source_problem:
          if self.PD.isotropic_source_everywhere:
            self.PD.get_Q(self.fixed_source, 0, gto)
            # FIXME: This assumes that adjoint source == forward source
            form = self.fixed_source*self.tensors.Wp[p]*v[p]*dx + self.bnd_vector_form[gto]
          else:
            form = ufl.zero()
            for n in range(self.DD.M):
              self.PD.get_Q(self.fixed_source, n, gto)
              # FIXME: This assumes that adjoint source == forward source
              form += self.fixed_source[n,gto] * self.tensors.Wp[n] * v[n] * dx + self.bnd_vector_form[gto]

          ass_timer.start()

          assemble(form, tensor=self.Q, finalize_tensor=False, add_values=add_values_Q)
          add_values_Q = True
          
          ass_timer.stop()
                      
        for gfrom in range(self.DD.G):
          
          if self.verb > 3 and self.DD.G > 1:
            print spc + 'GROUP [', gto, ',', gfrom, '] :'
          
          pres_Ss = False

          # TODO: Enlarge self.S and self.C to (L+1)^2 (or 1./2.*(L+1)*(L+2) in 2D) to accomodate for anisotropic
          # scattering (lines below using CC, SS are correct only for L = 0, when the following inner loop runs only
          # once.
          for l in range(self.L+1):
            for m in range(-l, l+1):
              if self.DD.angular_quad.get_D() == 2 and divmod(l+m,2)[1] == 0:
                continue

              pres_Ss |= self.PD.get_xs('Ss', self.S[l], gto, gfrom, l)
              self.PD.get_xs('C', self.C[l], gto, gfrom, l)
              
          if pres_Ss:
            Sd = ufl.diag(self.S)
            SS = self.tensors.QT[p,k1]*Sd[k1,k2]*self.tensors.Q[k2,q]
            Cd = ufl.diag(self.C)
            CC = self.tensors.QtT[p,i,k1]*Cd[k1,k2]*self.tensors.Qt[k2,q,j]
            
            ass_timer.start()
            
            if gfrom != gto:
              form = ( SS[p,q]*self.slns_mg[gfrom][q]*v[p] - CC[p,i,q,j]*self.slns_mg[gfrom][q].dx(j)*v[p].dx(i) ) * dx
              assemble(form, tensor=self.Q, finalize_tensor=False, add_values=add_values_Q)
            else:
              form = ( CC[p,i,q,j]*u[q].dx(j)*v[p].dx(i) - SS[q,p]*u[q]*v[p] ) * dx
              assemble(form, tensor=self.A, finalize_tensor=False, add_values=add_values_A)

            ass_timer.stop()
              
          if pres_fiss:
            pres_nSf = self.PD.get_xs('nSf', self.R, gfrom)
             
            if pres_nSf:
              ass_timer.start()
              
              if gfrom != gto:
                form = self.chi*self.R/(4*numpy.pi)*\
                       self.tensors.QT[p,0]*self.tensors.Q[0,q]*self.slns_mg[gfrom][q]*v[p]*dx
                assemble(form, tensor=self.Q, finalize_tensor=False, add_values=add_values_Q)
              else:
                # NOTE: Fixed-source case (eigenproblems can currently be solved only by the coupled-group scheme)
                if self.fixed_source_problem:
                  form = -self.chi*self.R/(4*numpy.pi)*self.tensors.QT[p,0]*self.tensors.Q[0,q]*u[q]*v[p]*dx
                  assemble(form, tensor=self.A, finalize_tensor=False, add_values=add_values_A)

              ass_timer.stop()

        #================================== END ASSEMBLE WITHIN-GROUP PROBLEM =======================================
        
        self.A.apply("add")
        self.Q.apply("add")
        
        mat_timer.stop()

        self.save_algebraic_system({'A':'A_{}'.format(gto), 'Q':'Q_{}'.format(gto)}, it)
        
        #====================================  SOLVE WITHIN-GROUP PROBLEM  ==========================================

        sln_timer.start()
        dolfin_solve(self.A, self.sln_vec, self.Q, "cg", "petsc_amg")
        sln_timer.stop()

        self.up_to_date["flux"] = False

        err = max(err, delta(self.sln_vec.array(), prev_slng_vec.array()))
          
      #==================================== END LOOP OVER GROUPS ==================================== 

      if err < self.parameters["group_GS"]["tol"]:
        break


  def assemble_algebraic_system(self):
    
    if self.verb > 1: print0(self.print_prefix + "Assembling algebraic system.")

    mat_timer = Timer("---- MTX: Complete construction")
    ass_timer = Timer("---- MTX: Assembling")
    
    add_values_A = False
    add_values_B = False
    add_values_Q = False
     
    for gto in range(self.DD.G):
    #===============================  LOOP OVER GROUPS AND ASSEMBLE  ================================
    
      spc = self.print_prefix + "  "
      
      if self.verb > 3 and self.DD.G > 1:
        print spc + 'GROUP [', gto, ',', gto, '] :'
      
      pres_fiss = self.PD.get_xs('chi', self.chi, gto)
  
      self.PD.get_xs('D', self.D, gto)
      self.PD.get_xs('St', self.R, gto)
      
      i,j,p,q,k1,k2 = ufl.indices(6)
      
      form = ( self.D*self.tensors.T[p,q,i,j]*self.u[gto][q].dx(j)*self.v[gto][p].dx(i) +
               self.R*self.tensors.W[p,q]*self.u[gto][q]*self.v[gto][p] ) * dx + self.bnd_matrix_form[gto]
      
      ass_timer.start()

      assemble(form, tensor=self.A, finalize_tensor=False, add_values=add_values_A)
      add_values_A = True
        
      ass_timer.stop()

      if self.fixed_source_problem:
        if self.PD.isotropic_source_everywhere:
          self.PD.get_Q(self.fixed_source, 0, gto)
          # FIXME: This assumes that adjoint source == forward source
          form = self.fixed_source * self.tensors.Wp[p] * self.v[gto][p] * dx + self.bnd_vector_form[gto]
        else:
          form = ufl.zero()
          for n in range(self.DD.M):
            self.PD.get_Q(self.fixed_source, n, gto)
            # FIXME: This assumes that adjoint source == forward source
            form += self.fixed_source[n,gto] * self.tensors.Wp[n] * self.v[gto][n] * dx + self.bnd_vector_form[gto]

        ass_timer.start()

        assemble(form, tensor=self.Q, finalize_tensor=False, add_values=add_values_Q)
        add_values_Q = True

        ass_timer.stop()
                    
      for gfrom in range(self.DD.G):
        
        if self.verb > 3 and self.DD.G > 1: print spc + 'GROUP [', gto, ',', gfrom, '] :'
        
        pres_Ss = False

        # TODO: Enlarge self.S and self.C to (L+1)^2 (or 1./2.*(L+1)*(L+2) in 2D) to accomodate for anisotropic
        # scattering (lines below using CC, SS are correct only for L = 0, when the following inner loop runs only
        # once.
        for l in range(self.L+1):
          for m in range(-l, l+1):
            if self.DD.angular_quad.get_D() == 2 and divmod(l+m,2)[1] == 0:
              continue

            pres_Ss |= self.PD.get_xs('Ss', self.S[l], gto, gfrom, l)
            self.PD.get_xs('C', self.C[l], gto, gfrom, l)
            
        if pres_Ss:
          Sd = ufl.diag(self.S)
          SS = self.tensors.QT[p,k1]*Sd[k1,k2]*self.tensors.Q[k2,q]
          Cd = ufl.diag(self.C)
          CC = self.tensors.QtT[p,i,k1]*Cd[k1,k2]*self.tensors.Qt[k2,q,j]
          
          ass_timer.start()
          
          form = ( CC[p,i,q,j]*self.u[gfrom][q].dx(j)*self.v[gto][p].dx(i) - 
                   SS[q,p]*self.u[gfrom][q]*self.v[gto][p] ) * dx
          assemble(form, tensor=self.A, finalize_tensor=False, add_values=add_values_A)

          ass_timer.stop()
            
        if pres_fiss:
          pres_nSf = self.PD.get_xs('nSf', self.R, gfrom)
           
          if pres_nSf:
            ass_timer.start()
            
            if self.fixed_source_problem:
              form = -self.chi*self.R/(4*numpy.pi)*\
                     self.tensors.QT[p,0]*self.tensors.Q[0,q]*self.u[gfrom][q]*self.v[gto][p]*dx
              assemble(form, tensor=self.A, finalize_tensor=False, add_values=add_values_A)
            else:
              form = self.chi*self.R/(4*numpy.pi)*\
                     self.tensors.QT[p,0]*self.tensors.Q[0,q]*self.u[gfrom][q]*self.v[gto][p]*dx
              assemble(form, tensor=self.B, finalize_tensor=False, add_values=add_values_B)
              add_values_B = True

            ass_timer.stop()
    
    #=============================  END LOOP OVER GROUPS AND ASSEMBLE  ===============================
                        
    self.A.apply("add")
    if self.fixed_source_problem:
      self.Q.apply("add")
    elif self.eigenproblem:
      self.B.apply("add")

                    
  def solve(self, it=0):
    if self.group_GS:
      self.solve_group_GS(it)
    else:
      super(EllipticSNFluxModule, self).solve(it)
      self.slns_mg = split(self.sln)

    i,p,q,k1,k2 = ufl.indices(5)

    sol_timer = Timer("-- Complete solution")
    aux_timer = Timer("---- SOL: Computing angular flux + adjoint")

    # TODO: Move to Discretization
    V11 = FunctionSpace(self.DD.mesh, "CG", self.DD.parameters["p"])

    for gto in range(self.DD.G):
      self.PD.get_xs('D', self.D, gto)

      form = self.D * ufl.diag_vector(ufl.as_matrix(self.DD.ordinates_matrix[i,p]*self.slns_mg[gto][q].dx(i), (p,q)))

      for gfrom in range(self.DD.G):
        pres_Ss = False

        # TODO: Enlarge self.S and self.C to (L+1)^2 (or 1./2.*(L+1)*(L+2) in 2D) to accomodate for anisotropic
        # scattering (lines below using CC, SS are correct only for L = 0, when the following inner loop runs only
        # once.
        for l in range(self.L+1):
          for m in range(-l, l+1):
            if self.DD.angular_quad.get_D() == 2 and divmod(l+m, 2)[1] == 0:
              continue

            pres_Ss |= self.PD.get_xs('Ss', self.S[l], gto, gfrom, l)
            self.PD.get_xs('C', self.C[l], gto, gfrom, l)

        if pres_Ss:
          Cd = ufl.diag(self.C)
          CC = self.tensors.Y[p,k1] * Cd[k1,k2] * self.tensors.Qt[k2,q,i]

          form += ufl.as_vector(CC[p,q,i] * self.slns_mg[gfrom][q].dx(i), p)

      # project(form, self.DD.Vpsi1, function=self.aux_slng, preconditioner_type="petsc_amg")
      # FASTER, but requires form compilation for each dir.:
      for pp in range(self.DD.M):
        assign(self.aux_slng.sub(pp), project(form[pp], V11, preconditioner_type="petsc_amg"))

      self.psi_mg[gto].assign(self.slns_mg[gto] + self.aux_slng)
      self.adj_psi_mg[gto].assign(self.slns_mg[gto] - self.aux_slng)

  def update_phi(self):
    timer = Timer("Scalar flux update")

    for g in range(self.DD.G):
      phig = self.phi_mg[g]
      phig_v = phig.vector()
      psig = self.psi_mg[g]

      for n in range(self.DD.M):
        # This doesn't work (Dolfin Issue #454)
        # assign(self.phi.sub(g), self.phi.sub(g) + self.tensors.Wp[n]*self.psi.sub(g).sub(n))
        psign = psig.sub(n, deepcopy=True)
        phig_v.axpy(float(self.tensors.Wp[n]), psign.vector())

    self.up_to_date["flux"] = True

  def eigenvalue_residual_norm(self,norm_type='l2'):
    if self.group_GS:
      warning("Residual norm can currently be computed only when the whole system is assembled.")
      return 0.
    else:
      super(EllipticSNFluxModule, self).eigenvalue_residual_norm(norm_type)

  def fixed_source_residual_norm(self,norm_type='l2'):
    if self.group_GS:
      warning("Residual norm can currently be computed only when the whole system is assembled.")
      return 0.
    else:
      super(EllipticSNFluxModule,self).fixed_source_residual_norm(norm_type)

  def visualize(self, it=0):
    super(EllipticSNFluxModule, self).visualize()

    labels = ["psi", "adj_psi"]
    functs = [self.psi_mg, self.adj_psi_mg]
    for var,lbl,fnc in zip(["angular_flux", "adjoint_angular_flux"], labels, functs):
      try:
        should_vis = divmod(it, self.parameters["visualization"][var])[1] == 0
      except ZeroDivisionError:
        should_vis = False

      if should_vis:
        for g in range(self.DD.G):
          for n in range(self.DD.M):
            fgn = fnc[g].sub(n)
            fgn.rename(lbl, "{}_g{}_{}".format(lbl, g, n))
            self.vis_files[var][g][n] << (fgn, float(it))
