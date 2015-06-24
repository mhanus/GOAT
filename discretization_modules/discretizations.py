from dolfin import FunctionSpace, MixedFunctionSpace, VectorFunctionSpace
from dolfin.cpp.common import Timer
from common import print0
from ufl import as_matrix
from discretization_modules.generic_discretization import Discretization

__author__ = 'mhanus'

# noinspection PyAttributeOutsideInit
class SNDiscretization(Discretization):
  def __init__(self, problem, SN_order, verbosity=0):
    super(SNDiscretization, self).__init__(problem, verbosity)

    if self.verb > 1: print0("Obtaining angular discretization data")
    t_ordinates = Timer("Angular discretization data")

    from transport_data import ordinates_ext_module, quadrature_file
    from common import check_sn_order

    self.angular_quad = ordinates_ext_module.OrdinatesData(SN_order, self.mesh.topology().dim(), quadrature_file)
    self.M = self.angular_quad.get_M()
    self.N = check_sn_order(SN_order)

    ord_mat = [self.angular_quad.get_xi().tolist(), self.angular_quad.get_eta().tolist()]
    if self.angular_quad.get_D() == 3: ord_mat.append(self.angular_quad.get_mu().tolist())
    self.ordinates_matrix = as_matrix(ord_mat)

    if self.verb > 2:
      self.angular_quad.print_info()

  def init_solution_spaces(self, group_GS=False):
    super(SNDiscretization, self).init_solution_spaces()

    self.t_spaces.start()

    self.Vpsi1 = VectorFunctionSpace(self.mesh, "CG", self.parameters["p"], self.M)

    if not group_GS:
      self.V = MixedFunctionSpace([self.Vpsi1]*self.G)
    else:
      # Alias the solution space as Vpsi1
      self.V = self.Vpsi1

    self.Vphi1 = FunctionSpace(self.mesh, "CG", self.parameters["p"])

    self.ndof1 = self.Vpsi1.dim()
    self.ndof = self.V.dim()

    if self.verb > 1:
      print0("  {0} direction(s), {1} group(s), {2}".format(self.M, self.G, "group GS" if group_GS else "full system"))
      print0("    NDOF (solution): {0}".format(self.ndof) )
      print0("    NDOF (1gr): {0}".format(self.ndof1) )
      print0("    NDOF (1dir, 1gr): {0}".format(self.Vpsi1.sub(0).dim()) )
      print0("    NDOF (XS): {0}".format(self.ndof0) )