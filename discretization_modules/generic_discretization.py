import os, numpy
from dolfin import FunctionSpace, parameters, compile_extension_module
from dolfin.cpp.io import File
from dolfin.cpp.common import Timer, Parameters, IntArray, DoubleArray, MPI
from dolfin.cpp.mesh import Mesh, CellFunction, MeshFunction, entities

from common import pid, comm, print0

# TODO: Split discretization and channel data

# noinspection PyArgumentList
def get_parameters():
  params = Parameters(
    "discretization", 
    p = 1,
  )
  return params

class Discretization(object):
  def __init__(self, problem, verbosity=0):
    """

    :param ProblemData problem:
    :param int verbosity:
    :return:
    """
    self.parameters = parameters["discretization"]

    self.verb = verbosity
    self.vis_folder = os.path.join(problem.out_folder, "MESH")
    self.core = problem.core
    self.G = problem.G

    if self.verb > 1: print pid+"Loading mesh"
        
    t_load = Timer("DD: Data loading")

    try:
      self.mesh = problem.mesh_module.mesh
      self.boundaries = problem.mesh_module.boundaries
      self.cell_regions = problem.mesh_module.regions

    except AttributeError:
      if self.verb > 1: print pid + "  mesh data"
      self.mesh = Mesh(problem.mesh_files.mesh)

      if self.verb > 1: print pid + "  physical data"
      self.cell_regions = MeshFunction("size_t", self.mesh, problem.mesh_files.physical_regions)

      if self.verb > 1: print pid + "  boundary data"
      self.boundaries = MeshFunction("size_t", self.mesh, problem.mesh_files.facet_regions)

    assert self.mesh
    assert self.boundaries.array().size > 0

    if self.verb > 2:
      print pid+"  mesh info: " + str(self.mesh)

    if self.verb > 1: print0("Defining function spaces" )

    self.t_spaces = Timer("DD: Function spaces construction")

    # Spaces that must be specified by the respective subclasses
    self.V = None     # solution space
    self.Vphi = None  # scalar flux space
    self.Vphi1 = None # 1-g scalar flux space
    
    # XS / TH space
    self.V0 = FunctionSpace(self.mesh, "DG", 0)
    self.ndof0 = self.V0.dim()

    dofmap = self.V0.dofmap()
    self.local_ndof0 = dofmap.local_dimension("owned")

    assert self.cell_regions.size == self.local_ndof0

  def __create_cell_dof_mapping(self, dofmap):
    """
    Generate cell -> dof mapping for all cells of current partition.
    Note: in DG(0) space, there is one dof per element and no ghost cells.

    :param GenericDofMap dofmap: DG(0) dofmap
    """

    if self.verb > 2: print0("Constructing cell -> dof mapping")
    timer = Timer("DD: Cell->dof construction")

    code = \
    '''
      #include <dolfin/mesh/Cell.h>

      namespace dolfin
      {
        void fill_in(Array<int>& local_cell_dof_map, const Mesh& mesh, const GenericDofMap& dofmap)
        {
          std::size_t local_dof_range_start = dofmap.ownership_range().first;
          int* cell_dof_data = local_cell_dof_map.data();

          for (CellIterator c(mesh); !c.end(); ++c)
            *cell_dof_data++ = dofmap.cell_dofs(c->index())[0] - local_dof_range_start;
        }
      }
    '''

    cell_mapping_module = compile_extension_module(code)
    cell_dof_array = IntArray(self.local_ndof0)
    cell_mapping_module.fill_in(cell_dof_array, self.mesh, dofmap)
    self._local_cell_dof_map = cell_dof_array.array()

    timer.stop()

  def __create_cell_layers_mapping(self):
    """
    Generate a cell -> axial layer mapping for all cells of current partition. Note that keys are ordered by the
    associated DG(0) dof, not by the cell index in the mesh.
    """

    if self.verb > 2: print0("Constructing cell -> layer mapping")
    timer = Timer("DD: Cell->layer construction")

    code = \
    '''
      #include <dolfin/mesh/Cell.h>

      namespace dolfin
      {
        void fill_in(Array<int>& local_cell_layers,
                     const Mesh& mesh, const Array<int>& cell_dofs, const Array<double>& layer_boundaries)
        {
          std::size_t num_layers = layer_boundaries.size() - 1;
          unsigned int layer;

          for (CellIterator c(mesh); !c.end(); ++c)
          {
            double midz = c->midpoint().z();
            for (layer = 0; layer < num_layers; layer++)
              if (layer_boundaries[layer] <= midz && midz <= layer_boundaries[layer+1])
                break;

            int dof = cell_dofs[c->index()];
            local_cell_layers[dof] = layer;
          }
        }
      }
    '''

    cell_mapping_module = compile_extension_module(code)

    cell_layers_array =  IntArray(self.local_ndof0)
    cell_mapping_module.fill_in(cell_layers_array, self.mesh, self.local_cell_dof_map, self.core.layer_boundaries)
    self._local_cell_layers = cell_layers_array.array()

    timer.stop()

  def __create_cell_vol_mapping(self):
    """
    Generate cell -> volume mapping for all cells of current partition. Note that keys are ordered by the
    associated DG(0) dof, not by the cell index in the mesh.

    This map is required for calculating various densities from total region integrals (like cell power densities from
    cell-integrated powers).
    """

    if self.verb > 2: print0("Constructing cell -> volume mapping")
    timer = Timer("DD: Cell->vol construction")

    code = \
    '''
      #include <dolfin/mesh/Cell.h>

      namespace dolfin
      {
        void fill_in(Array<double>& cell_vols, const Mesh& mesh, const Array<int>& cell_dofs)
        {
          for (CellIterator c(mesh); !c.end(); ++c)
            cell_vols[cell_dofs[c->index()]] = c->volume();
        }
      }
    '''

    cell_mapping_module = compile_extension_module(code)
    cell_vol_array = DoubleArray(self.local_ndof0)
    cell_mapping_module.fill_in(cell_vol_array, self.mesh, self.local_cell_dof_map)
    self._local_cell_volumes = cell_vol_array.array()

    timer.stop()

  @property
  def local_cell_dof_map(self):
    try:
      self._local_cell_dof_map
    except AttributeError:
      self.__create_cell_dof_mapping(self.V0.dofmap())
    return self._local_cell_dof_map

  @property
  def local_cell_volumes(self):
    try:
      self._local_cell_volumes
    except AttributeError:
      self.__create_cell_vol_mapping()
    return self._local_cell_volumes

  @property
  def local_cell_layers(self):
    try:
      self._local_cell_layers
    except AttributeError:
      self.__create_cell_layers_mapping()
    return self._local_cell_layers

  def visualize_mesh_data(self):
    timer = Timer("DD: Mesh data visualization")
    if self.verb > 2: print0("Visualizing mesh data")

    File(os.path.join(self.vis_folder, "mesh.pvd"), "compressed") << self.mesh
    File(os.path.join(self.vis_folder, "boundaries.pvd"), "compressed") << self.boundaries
    File(os.path.join(self.vis_folder, "mesh_regions.pvd"), "compressed") << self.cell_regions

    # Create MeshFunction to hold cell process rank
    processes = CellFunction('size_t', self.mesh, MPI.rank(comm))
    File(os.path.join(self.vis_folder, "mesh_partitioning.pvd"), "compressed") << processes

  def print_diagnostics(self):
    print MPI.rank(comm), self.mesh.num_entities(self.mesh.topology().dim())

    dofmap = self.V0.dofmap()

    print MPI.rank(comm), dofmap.ownership_range()
    print MPI.rank(comm), numpy.min(dofmap.collapse(self.mesh)[1].values()), \
          numpy.max(dofmap.collapse(self.mesh)[1].values())

    print "#Owned by {}: {}".format(MPI.rank(comm), dofmap.local_dimension("owned"))
    print "#Unowned by {}: {}".format(MPI.rank(comm), dofmap.local_dimension("unowned"))

    for cell in entities(self.mesh, self.mesh.topology().dim()):
      print MPI.rank(comm), cell.index(), dofmap.tabulate_entity_dofs(self.mesh.topology().dim(), cell.index())
