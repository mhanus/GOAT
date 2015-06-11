"""
Created on 11.5.2014

@author: mhanus
"""
# TODO: try-except for dolfin imports
# TODO: currently isotropic source is expected; extend to arbitrary angle-dependent source
import os

from dolfin.cpp.common import Timer, Parameters
from dolfin.cpp.function import FunctionAssigner
from dolfin import assemble, Function, FacetNormal, split
from dolfin import solve as dolfin_solve
from dolfin.cpp.io import File
import ufl
from ufl.integral import Measure
from ufl.objects import dx
import numpy

from common import delta, print0, coupled_solver_error, dolfin_version_id
from discretization_modules.discretizations import SNDiscretization
import flux_module
from flux_modules import backend_ext_module


# noinspection PyArgumentList
def get_parameters():
  params = flux_module.get_parameters()
  params.add(backend_ext_module.GeneralizedEigenSolver.default_parameters())
  params.add(
              Parameters(
                "group_GS",
                max_niter = -1,
                tol = 1e-6
              )
            )
  params["visualization"].add("angular_flux", 0)
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
      self.Q = ufl.as_tensor( numpy.reshape(tensors.Q(), tensors.shape_Q()) )
      self.QT = ufl.transpose(self.Q)
      self.Qt = ufl.as_tensor( numpy.reshape(tensors.Qt(), tensors.shape_Qt()) )
      self.QtT = ufl.as_tensor( self.Qt[k2,q,j], (p,i,k1) )
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

    super(EllipticSNFluxModule, self).__init__(PD, DD, verbosity)

    self.max_group_GS_it = self.parameters["group_GS"]["max_niter"]
    self.group_GS = self.max_group_GS_it > 0  
    
    if self.eigenproblem and self.group_GS:
      print "Group Gauss-Seidel for eigenproblem not yet supported - switching to all-group coupled solution method."
      self.group_GS = False
      
    if self.group_GS:
      self.sln_fun_1g = Function(self.DD.Vpsi1)
    else:
      self.max_group_GS_it = 1
      
    if self.fixed_source_problem:
      self.vals_Q = numpy.empty(self.DD.ndof, dtype='float64')
    
    if self.verb > 1: print0("Defining coefficient functions and tensors")
    
    self.u = split(self.u)
    self.v = split(self.v)

    self.D = Function(self.DD.V0)

    self.L = PD.scattering_order-1
    self.tensors = self.AngularTensors(self.DD.angular_quad, self.L)

    self.C = numpy.empty(self.L+1, Function)
    self.S = numpy.empty(self.L+1, Function)
    for l in range(self.L+1):
      self.C[l] = Function(self.DD.V0)
      self.S[l] = Function(self.DD.V0)

    var = "angular_flux"
    self.vis_files[var] = \
    [
      [
        File(os.path.join(self.vis_folder, "{}_g{}_m{}.pvd".format(var, g, m)), "compressed") for g in range(self.DD.G)
      ]
      for m in range(self.DD.M)
    ]

  # TODO: Reflective boundary conditions  
  def define_boundary_terms(self, vacuum_boundaries=None, reflective_boundaries=None, incoming_flux=None):
    psi_inc_not_dict = False

    if not incoming_flux:
      incoming_flux = {}
      incoming_flux_boundaries = set()
    else:
      try:
        incoming_flux_boundaries = incoming_flux.keys()
      except AttributeError:
        incoming_flux_boundaries = set()
        psi_inc_not_dict = True

    if not vacuum_boundaries: vacuum_boundaries = set()
    if not reflective_boundaries: reflective_boundaries = set()

    if self.verb > 2: print0("Defining boundary forms")

    self.bnd_vector_form = numpy.empty(self.DD.G, dtype=object)
    self.bnd_matrix_form = numpy.empty(self.DD.G, dtype=object)

    n = FacetNormal(self.DD.mesh)
    i,p,q = ufl.indices(3)
    
    natural_boundaries = vacuum_boundaries.union(incoming_flux_boundaries)

    nonzero = lambda x: numpy.all(x > 0)
    if not psi_inc_not_dict:
      incoming_flux = {k:numpy.array(v) for k,v in incoming_flux.iteritems()}
      nonzero_inc_flux = incoming_flux and any(map(nonzero, incoming_flux.values()))
    else:
      nonzero_inc_flux = nonzero(incoming_flux)

    if nonzero_inc_flux and not self.fixed_source_problem:
        coupled_solver_error(__file__,
                     "define boundary forms", 
                     "Incoming flux specified for an eigenvalue problem"+\
                     "(Q must be given whenever phi_inc is; it may possibly be zero everywhere)")
    
    if natural_boundaries:    
      if not self.boundary_physical_name_idx_map:
        coupled_solver_error(__file__,
                     "define boundary forms", 
                     "File with boundary names required if vacuum_ or incoming_flux_boundaries are specified.")
              
      for g in range(self.DD.G):
        self.bnd_vector_form[g] = ufl.zero()
        self.bnd_matrix_form[g] = ufl.zero()
      
      ds = Measure("ds")[self.DD.boundaries]
          
      for bnd in natural_boundaries:
        for g in range(self.DD.G):
          self.bnd_matrix_form[g] += \
            abs(self.tensors.G[p,q,i]*n[i])*self.u[g][q]*self.v[g][p]*ds(self.boundary_physical_name_idx_map[bnd])
    
      if nonzero_inc_flux:
        for pp in range(self.DD.M):
          omega_p = ufl.as_vector( self.DD.angular_quad.get_ordinate(pp) )
          omega_p_dot_n = omega_p[i]*n[i]        
          
          for bnd in incoming_flux_boundaries:
            try:
              psi_inc_val = incoming_flux[bnd]
            except KeyError:
              coupled_solver_error(__file__,
                           "define boundary forms", 
                           "Incoming flux for boundary marker {} not specified".format(bnd))
              
            if psi_inc_val.shape != (self.DD.G, self.DD.M):
              coupled_solver_error(__file__,
                           "define boundary forms", 
                           "Incoming flux with incorrect number of groups and directions specified: "+
                           "{}, expected ({}, {})".format(incoming_flux[bnd].shape, self.DD.G, self.DD.M))
            
            
            for g in range(self.DD.G):
              self.bnd_vector_form[g] += \
                ufl.conditional(omega_p_dot_n < 0,
                                omega_p_dot_n*self.tensors.Wp[pp]*psi_inc_val[g,pp]*self.v[g][pp]*ds(self.boundary_physical_name_idx_map[bnd]),
                                ufl.zero())
        
    else: # Apply the boundary condition everywhere
      ds = Measure("ds")
      
      for g in range(self.DD.G):
        self.bnd_matrix_form[g] = abs(self.tensors.G[p,q,i]*n[i])*self.u[g][q]*self.v[g][p]*ds()
        self.bnd_vector_form[g] = ufl.zero()
      
      if nonzero_inc_flux:
        psi_inc_val = incoming_flux
      
        if psi_inc_val.shape != (self.DD.G, self.DD.M):
          coupled_solver_error(__file__,
                       "define boundary forms", 
                       "Incoming flux with incorrect number of groups and directions specified: "+
                       "{}, expected ({}, {})".format(psi_inc_val.shape, self.DD.G, self.DD.M))
                
        for pp in range(self.DD.M):
          omega_p = ufl.as_vector( self.DD.angular_quad.get_ordinate(pp) )
          omega_p_dot_n = omega_p[i]*n[i]
          for g in range(self.DD.G):
            self.bnd_vector_form[g] += \
              ufl.conditional(omega_p_dot_n < 0,
                              omega_p_dot_n*self.tensors.Wp[pp]*psi_inc_val[g,pp]*self.v[g][pp]*ds(),
                              ufl.zero())


  def solve_group_GS(self, it=0):
    if self.verb > 1: print0(self.print_prefix + "Solving..." )
  
    if self.eigenproblem:  
      coupled_solver_error(__file__,
                           "solve using group GS",
                           "Group Gauss-Seidel for eigenproblem not yet supported")
    
    sol_timer = Timer("-- Complete solution")
    mat_timer = Timer("---- Complete matrices construction")
    ass_timer = Timer("------ Matrix assembling")
    
        
    for gsi in range(self.max_group_GS_it):
    #==========================================  GAUSS-SEIDEL LOOP  ============================================
    
      if self.verb > 2: print self.print_prefix + 'Gauss-Seidel iteration {}'.format(gsi)
      
      self.prev_sln_vec.zero()
      self.prev_sln_vec.axpy(1.0, self.sln_vec)
      self.prev_sln_vec.apply("insert")
    
       
      for gto in range(self.DD.G):
      #=========================================  LOOP OVER GROUPS  ===============================================
        
        spc = self.print_prefix + "  "
        
        if self.verb > 3 and self.DD.G > 1: print spc + 'GROUP [', gto, ',', gto, '] :'
        
        #====================================  ASSEMBLE WITHIN-GROUP PROBLEM  ========================================
        
        mat_timer.start()

        pres_fiss = self.PD.get_xs('chi', self.chi, gto)
    
        self.PD.get_xs('D', self.D, gto)
        self.PD.get_xs('St', self.R, gto)
        
        i,j,p,q,k1,k2,p,q = ufl.indices(6)
        
        form = ( self.D*self.tensors.T[p,q,i,j]*self.u[gto][q].dx(j)*self.v[gto][p].dx(i) + 
                 self.R*self.tensors.W[p,q]*self.u[gto][q]*self.v[gto][p] ) * dx + self.bnd_matrix_form[gto]
        
        ass_timer.start()
    
        add_values_A = False
        add_values_Q = False

        assemble(form, tensor=self.A, finalize_tensor=False, add_values=add_values_A)
        add_values_A = True
          
        ass_timer.stop()

        if self.fixed_source_problem:
          self.PD.get_xs('Q', self.fixed_source, gto)
          form = self.fixed_source*self.tensors.Wp[p]*self.v[gto][p]*dx + self.bnd_vector_form[gto]
          
          ass_timer.start()

          assemble(form, tensor=self.Q, finalize_tensor=False, add_values=add_values_Q)
          add_values_Q = True
          
          ass_timer.stop()
                      
        for gfrom in range(self.DD.G):
          
          if self.verb > 3 and self.DD.G > 1:
            print spc + 'GROUP [', gto, ',', gfrom, '] :'
          
          pres_Ss = False
          
          # TODO: Currently only L==0 is supported, thus m is not used
          for l in range(self.L+1):
            for m in range(-l, l+1):
              pres_Ss |= self.PD.get_xs('Ss', self.S[l], gto, gfrom, l)
              self.PD.get_xs('C', self.C[l], gto, gfrom, l)
              
          if pres_Ss:
            Sd = ufl.diag(self.S)
            SS = self.tensors.QT[p,k1]*Sd[k1,k2]*self.tensors.Q[k2,q]
            Cd = ufl.diag(self.C)
            CC = self.tensors.QtT[p,i,k1]*Cd[k1,k2]*self.tensors.Qt[k2,q,j]
            
            ass_timer.start()
            
            if gfrom != gto:
              if gfrom < gto:
                form = ( SS[p,q]*self.sln_fun[gfrom][q]*self.v[gto][p] - 
                         CC[p,i,q,j]*self.sln_fun[gfrom][q].dx(j)*self.v[gto][p].dx(i) ) * dx
              else:
                form = ( SS[p,q]*self.prev_sln_fun[gfrom][q]*self.v[gto][p] - 
                         CC[p,i,q,j]*self.prev_sln_fun[gfrom][q].dx(j)*self.v[gto][p].dx(i) ) * dx
              
              assemble(form, tensor=self.Q, finalize_tensor=False, add_values=add_values_Q)
            else:
              form = ( CC[p,i,q,j]*self.u[gfrom][q].dx(j)*self.v[gto][p].dx(i) - 
                       SS[q,p]*self.u[gfrom][q]*self.v[gto][p] ) * dx

              assemble(form, tensor=self.A, finalize_tensor=False, add_values=add_values_A)

            ass_timer.stop()
              
          if pres_fiss:
            pres_nSf = self.PD.get_xs('nSf', self.R, gfrom)
             
            if pres_nSf:
              ass_timer.start()
              
              if gfrom != gto:
                if gfrom < gto:
                  form = self.chi*self.R/(4*numpy.pi)*self.tensors.QT[p,0]*self.tensors.Q[0,q]*self.sln_fun[gfrom][q]*self.v[gto][p]*dx
                else:
                  form = self.chi*self.R/(4*numpy.pi)*self.tensors.QT[p,0]*self.tensors.Q[0,q]*self.prev_sln_fun[gfrom][q]*self.v[gto][p]*dx

                assemble(form, tensor=self.Q, finalize_tensor=False, add_values=add_values_Q)
              else:
                # NOTE: Fixed-source case (eigenproblems can currently be solved only by the coupled-group scheme)
                if self.fixed_source_problem:
                  form = -self.chi*self.R/(4*numpy.pi)*self.tensors.QT[p,0]*self.tensors.Q[0,q]*self.u[gfrom][q]*self.v[gto][p]*dx            
                  assemble(form, tensor=self.A, finalize_tensor=False, add_values=add_values_A)

              ass_timer.stop()
        
      
        #================================== END ASSEMBLE WITHIN-GROUP PROBLEM =======================================
        
        self.A.apply("add")
        self.Q.apply("add")
        
        mat_timer.stop()

        self.save_algebraic_system({'A':'A_{}'.format(gto), 'Q':'Q_{}'.format(gto)}, it)
        
        #====================================  SOLVE WITHIN-GROUP PROBLEM  ==========================================
        
        dolfin_solve(self.A, self.sln_fun_1g, self.Q, "cg", "petsc_amg")
        assigner = FunctionAssigner(self.DD.Vpsi.sub(gto), self.DD.Vpsi1)
        assigner.assign(self.sln_fun.sub(gto), self.sln_fun_1g)
        self.up_to_date["flux"] = False
          
      #==================================== END LOOP OVER GROUPS ==================================== 
    
      err = delta(self.sln_vec.array(), self.prev_sln_vec.array())
      if err < self.parameters["group_GS"]["tol"]:
        break
              

  def assemble_algebraic_system(self):
    
    if self.verb > 1: print0(self.print_prefix + "Assembling algebraic system.")

    mat_timer = Timer("---- Complete matrices construction")
    ass_timer = Timer("------ Matrix assembling")
    
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
      
      i,j,p,q,k1,k2,p,q = ufl.indices(8)
      
      form = ( self.D*self.tensors.T[p,q,i,j]*self.u[gto][q].dx(j)*self.v[gto][p].dx(i) + 
               self.R*self.tensors.W[p,q]*self.u[gto][q]*self.v[gto][p] ) * dx + self.bnd_matrix_form[gto]
      
      ass_timer.start()

      assemble(form, tensor=self.A, finalize_tensor=False, add_values=add_values_A)
      add_values_A = True
        
      ass_timer.stop()
      
      if self.fixed_source_problem:
        self.PD.get_xs('Q', self.fixed_source, gto)
        form = self.fixed_source*self.tensors.Wp[p]*self.v[gto][p]*dx + self.bnd_vector_form[gto]
        
        ass_timer.start()

        assemble(form, tensor=self.Q, finalize_tensor=False, add_values=add_values_Q)
        add_values_Q = True
        
        ass_timer.stop()
                    
      for gfrom in range(self.DD.G):
        
        if self.verb > 3 and self.DD.G > 1: print spc + 'GROUP [', gto, ',', gfrom, '] :'
        
        pres_Ss = False
        
        # TODO: Currently only L==0 is supported, thus m is not used
        for l in range(self.L+1):
          for m in range(-l, l+1):
            pres_Ss |= self.PD.get_group_mat_leg_moment_xs('Ss', self.S[l], gto, gfrom, l)
            self.PD.get_group_mat_leg_moment_xs('C', self.C[l], gto, gfrom, l)
            
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
              form = -self.chi*self.R/(4*numpy.pi)*self.tensors.QT[p,0]*self.tensors.Q[0,q]*self.u[gfrom][q]*self.v[gto][p]*dx
              assemble(form, tensor=self.A, finalize_tensor=False, add_values=add_values_A)
            else:
              form = self.chi*self.R/(4*numpy.pi)*self.tensors.QT[p,0]*self.tensors.Q[0,q]*self.u[gfrom][q]*self.v[gto][p]*dx
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
      # Store the scalar flux for convergence monitoring
      self.prev_phi.vector().zero()
      self.prev_phi.vector().axpy(1.0, self.phi.vector())

      self.solve_group_GS(it)
    else:
      super(EllipticSNFluxModule, self).solve(it)
  
  def update_phi(self):
    for g in range(self.DD.G):
      for n in range(self.DD.M):
        self.phi[g] = self.phi[g] + self.tensors.Wp[n]*self.sln_fun[g][n]

    self.up_to_date["flux"] = True

  def visualize(self, it=0):
    super(EllipticSNFluxModule, self).visualize()

    var = "angular_power"
    try:
      should_vis = divmod(it, self.parameters["visualization"][var])[1] == 0
    except ZeroDivisionError:
      should_vis = False

    if should_vis:
      for g in range(self.DD.G):
        for n in range(self.DD.M):
          fn = self.sln_fun.sub(g).sub(n)
          fn.rename("psi", "psi_g{}_{}".format(g, n))
          self.vis_files["angular_flux"][g,n] << (fn, it)
