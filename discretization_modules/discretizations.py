from dolfin import FunctionSpace, MixedFunctionSpace, VectorFunctionSpace
from dolfin.cpp.common import Timer
from common import print0
from discretization_modules.generic_discretization import Discretization

__author__ = 'mhanus'


class DiffusionDiscretization(Discretization):
  def __init__(self, problem, verbosity=0):
    super(DiffusionDiscretization, self).__init__(problem, verbosity)

    self.t_spaces.start()
    self.Vphi1 = FunctionSpace(self.mesh, "CG", self.parameters["p"])
    self.Vphi = MixedFunctionSpace([self.Vphi1]*self.G)
    self.t_spaces.stop()

    self.ndof1 = self.Vphi1.dim()
    self.ndof = self.Vphi.dim()

    # Alias the solution space as Vphi
    self.V = self.Vphi

    if self.verb > 1:
      print0("  NDOF ({0}G): {1}".format(self.G, self.ndof) )
      print0("  NDOF (1G): {0}".format(self.ndof1) )
      print0("  NDOF (XS): {0}".format(self.ndof0) )

class SNDiscretization(Discretization):
  def __init__(self, problem, SN_order, verbosity=0):
    from dolfin import VectorFunctionSpace

    super(SNDiscretization, self).__init__(problem, verbosity)

    if self.verb > 1: print0("Obtaining angular discretization data")
    t_ordinates = Timer("! Angular discretization data")

    from transport_data import ordinates_ext_module, quadrature_file
    from common import check_sn_order

    self.angular_quad = ordinates_ext_module.OrdinatesData(SN_order, self.mesh.topology().dim(), quadrature_file)
    self.M = self.angular_quad.get_M()
    self.N = check_sn_order(SN_order)

    if self.verb > 2:
      self.angular_quad.print_info()

    if self.verb > 1: print0("Defining function spaces" )

    self.t_spaces.start()

    self.Vpsi1 = VectorFunctionSpace(self.mesh, "CG", self.parameters["p"], self.M)
    self.Vpsi = MixedFunctionSpace([self.Vpsi1]*self.G)
    self.Vphi = MixedFunctionSpace([FunctionSpace(self.mesh, "CG", self.parameters["p"])]*self.G)
    self.Vphi1 = FunctionSpace(self.mesh, "CG", self.parameters["p"])

    self.ndof1 = self.Vpsi1.dim()
    self.ndof = self.Vpsi.dim()
    self.ndof_phi = self.Vphi.dim()

    # Alias the solution space as Vpsi
    self.V = self.Vpsi

    if self.verb > 1:
      print0("  NDOF ({0} directions, {1} groups): {2}".format(self.M, self.G, self.ndof) )
      print0("  NDOF (1gr): {0}".format(self.ndof1) )
      print0("  NDOF (1dir, 1gr): {0}".format(self.Vpsi1.sub(0).dim()) )
      print0("  NDOF (scalar flux): {0}".format(self.ndof_phi) )
      print0("  NDOF (XS): {0}".format(self.ndof0) )

    self.t_spaces.stop()