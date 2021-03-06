from collections import defaultdict
import os, imp, sys
from dolfin.cpp.common import warning, Timer, MPI
from dolfin.cpp.io import File
import numpy
from common import pid, coupled_solver_error, mkdir_p, comm
from material_data_parser import parse_material

__author__ = 'Milan'

class MeshFiles:
  def __init__(self, folder, mesh_base_name):
    self.mesh = os.path.join(folder, mesh_base_name + ".xml")
    self.physical_regions = os.path.join(folder, mesh_base_name + "_physical_region.xml")
    self.facet_regions = os.path.join(folder, mesh_base_name + "_facet_region.xml")

    assert os.path.isfile(self.mesh) and \
           os.path.isfile(self.physical_regions) and \
           os.path.isfile(self.facet_regions)

    self.reg_names = os.path.join(folder, mesh_base_name+"_reg_names.txt")
    self.mat_names = os.path.join(folder, mesh_base_name+"_mat_names.txt")
    self.bnd_names = os.path.join(folder, mesh_base_name+"_bnd_names.txt")

# noinspection PyAttributeOutsideInit,PyUnboundLocalVariable
class ProblemData(object):
  def __init__(self, problem_name, core_spec_file="", mesh_base_name="", verbosity=0):
    super(ProblemData, self).__init__()

    self.folder = os.path.abspath(os.path.join('PROBLEMS', problem_name))

    # FIXME:
    #if MPI.rank(comm) == 0:
    self.parse_material_data()

    MPI.barrier(comm)

    self.verb = verbosity
    self.name = problem_name
    self.core_spec_file = core_spec_file

    self.xsd = dict()

    self.mesh_base_name = mesh_base_name
    if self.mesh_base_name == "":
      self.mesh_base_name = self.name

    self.out_folder = os.path.join(self.folder, "OUT", self.mesh_base_name)
    self.xs_vis_folder = os.path.join(self.out_folder, "XS")

    mkdir_p(self.xs_vis_folder)

    self.used_xs = set()
    self.isotropic_source_everywhere = True

    # Two alternative ways of specifying a mesh:
    #   1:  Python module
    #   2:  set of Dolfin mesh files + optional helper files defining region/material/boundary maps
    #
    self.mesh_module = None
    self.mesh_files = None

    self.region_physical_name_map = dict()
    self.reg_name_mat_name_map = dict()
    boundary_physical_name_map = dict()
    
    try:
      # Try to import a mesh module
      self.mesh_module = self.mesh_data_from_module(self.mesh_base_name)
    
    except (ImportError, IOError):
      # If failed, Dolfin mesh file structure is expected
      
      self.mesh_files = MeshFiles(self.folder, self.mesh_base_name)
      
      self.region_physical_names_from_file(self.mesh_files.reg_names)
      self.reg_names_to_material_names_from_file(self.mesh_files.mat_names)
      self.bc = BoundaryData.from_file(self.mesh_files.bnd_names)
    
    else:
      # Import region/material/boundary maps from the mesh module 
      
      try:
        self.region_physical_name_map = self.mesh_module.region_map
      except AttributeError:
        pass
  
      # try to get bc data from boundary_id-to-boundary_name map and a file with boundary_name-to-bc correspondences
      try:
        boundary_physical_name_map = self.mesh_module.boundary_map
      except AttributeError:
        pass
      finally:
        # use either the boundary_physical_name_map, or - if not set - assume all-vacuum bc
        self.bc = BoundaryData.from_boundary_names_map(boundary_physical_name_map)
  
      try:
        self.reg_name_mat_name_map = self.mesh_module.material_map
      except AttributeError:
        pass

    self.core = CoreData()
    self.load_core_and_bc_data()

    # Check if bcs have been loaded from core_spec_file; if not, try loading them directly from the mesh module
    if self.mesh_module is not None:
      if len(self.bc.vacuum_boundaries) == 0:
        try:
          self.bc.vacuum_boundaries = set(self.mesh_module.vacuum_boundaries)
        except AttributeError:
          pass

      if len(self.bc.reflective_boundaries) == 0:
        try:
          self.bc.reflective_boundaries = set(self.mesh_module.reflective_boundaries)
        except AttributeError:
          pass

      if len(self.bc.incoming_fluxes) == 0:
        try:
          self.bc.incoming_fluxes = self.mesh_module.incoming_fluxes
        except AttributeError:
          pass

  def mesh_data_from_module(self, name):
    try:
      return sys.modules[name]
    except KeyError:
      pass

    return imp.load_source(name, os.path.join(self.folder, name + ".py"))


  def region_physical_names_from_file(self, filename):
    try:
      with open(filename) as f:
        for line in f:
          data = line.strip().split()
          assert (len(data) >= 2)

          reg = int(data[0])
          self.region_physical_name_map[reg] = data[1]
    except IOError:
      warning("File with region names not found - default names corresponding to subdomain indices will be used.")
    except AssertionError:
      warning("File with region names has incorrect format - default names corresponding to subdomain indices " +
              "will be used.")

  def reg_names_to_material_names_from_file(self, filename):
    try:
      with open(filename) as f:
        for line in f:
          data = line.strip().split()
          assert (len(data) == 2)
          self.reg_name_mat_name_map[data[0]] = data[1]
    except IOError:
      pass
    except AssertionError:
      warning("File with material names has incorrect format - default names equal to region names will be used.")

  def load_core_and_bc_data(self):
    data_file_name = os.path.join(self.folder, self.core_spec_file)

    try:
      with open(data_file_name) as f:
        lines = f.readlines()
    except IOError:
      if self.mesh_module is not None:
        warning("No file with core data and b.c. specification has been found - b.c. must be specified in the mesh "
                "module file or otherwise all-vacuum boundaries will be assumed.")
      else:
        warning("No file with core data and b.c. specification has been found - default settings will be used.")
      return

    l = -1
    while l < len(lines)-1:
      l += 1

      if "CORE POWER DATA" in lines[l].upper():
        l += 1
        l += self.core.parse_power_data(lines[l:])
        continue

      if "AXIAL DIMENSIONS" in lines[l].upper():
        l += 1
        l += self.core.parse_axial_data(lines[l:])
        continue

      if "BOUNDARY CONDITIONS" in lines[l].upper():
        l += 1
        l += self.bc.parse_boundaries(lines[l:])
        continue

  def data_file_name(self, mat_name):
    return os.path.join(self.folder, "MATERIALS", mat_name + ".npz")

  def parse_material_data(self):
    mat_dir = os.path.join(self.folder, "MATERIALS")
    mat_dir_unprocessed = [os.path.join(mat_dir, f) for f in os.listdir(mat_dir) if not f.endswith('.npz')]

    get_name = lambda path: os.path.splitext(os.path.basename(path))[0]
    mat_names = set(map(get_name, mat_dir_unprocessed))

    for mat_name, source_data_file in zip(mat_names, mat_dir_unprocessed):
      data_file_name = self.data_file_name(mat_name)

      try:
        xs_data = numpy.load(data_file_name)
        if os.path.getmtime(data_file_name) >= os.path.getmtime(source_data_file):
          G, K = xs_data["dimensions"]
        else:
          xs_data.close()
          raise ProblemData.OldData()
      except (IOError, OSError, KeyError, ProblemData.OldData):
        G, K = parse_material(source_data_file)
      else:
        xs_data.close()

      try:
        # Set maximal scattering order
        self.scattering_order = max(self.scattering_order, K)
        # Assert consistency of xs data among different materials
        assert G == self.G
      except AttributeError:
        self.G = G
        self.scattering_order = K

  def distribute_material_data(self, regions, M):
    """
    :param ndarray regions: Array that stores physical region number (physical group in GMSH mesh) for each cell of
                            current rank's mesh partition.
    :param int M: Number of discrete directions (to be matched by the source term).
    """

    self.M = M

    self.set_region_data(regions)

    self.regions_materials = self.reg_mat_map[regions]

    for mat, mat_name in enumerate(self.material_names):
      try:
        xs_data = numpy.load(self.data_file_name(mat_name))
      except (OSError, IOError) as e:
        coupled_solver_error(__file__,
                             "initialize XS data for material {}".format(mat_name),
                             "Data file " + self.data_file_name(mat_name) + " could not be loaded.\n" +
                             "DETAILS:  " + str(e))

      for xs, xsd in xs_data.iteritems():
        if xs == 'Q':
          if xsd.shape[0] == 1:
            # expand isotropic source to all directions
            xsd = numpy.repeat(xsd,M,0)
            xsd *= 1./(4*numpy.pi)
          elif xsd.shape[0] != M:
            coupled_solver_error(__file__,
                                 "initialize XS data for material {}".format(mat_name),
                                 "Invalid number of source directions ({}, expected {})".format(xsd.shape[0], M))
          else:
            self.isotropic_source_everywhere = False

        try:
          self.xsd[xs]
        except KeyError:
          shape = (self.num_mat,) + xsd.shape
          self.xsd[xs] = numpy.zeros(shape)

        self.xsd[xs][mat] = xsd
        self.used_xs.add(xs)

      xs_data.close()

    self.fixed_source_problem = 'Q' in self.used_xs
    self.eigenproblem = len({'nSf', 'chi'}.intersection(self.used_xs)) == 2 and not self.fixed_source_problem
    assert self.fixed_source_problem or self.eigenproblem

  # noinspection PyTypeChecker
  def get_xs(self, xs, xs_fun, gto=0, gfrom=0, k=0, vis=False):
    assert (0 <= gto < self.G)
    assert (0 <= gfrom < self.G)
    assert (0 <= k < self.scattering_order)

    timer = Timer("Fetching XS")

    try:
      xsd = self.xsd[xs]
    except KeyError:
      return False

    if xsd.ndim == 2:
      xs_values = xsd[:,gto]
    else:
      xs_values = xsd[:,k,gto,gfrom]

    # Assign xs data from materials to physical regions; skip if xs values for all materials are manually set to 0
    if numpy.any(xs_values > 0):
      xs_fun.vector()[:] = numpy.choose(self.regions_materials, xs_values)
    else:
      return False

    if vis:
      xs_label = xs

      if self.G > 0:
        xs_label += "_{}".format(gto)

        if xsd.ndim == 4:
          xs_label += "_{}".format(gfrom)

      if self.scattering_order > 1 and xsd.ndim == 4:
        xs_label += "_{}".format(k)

      xs_fun.rename(xs_label, xs_label)
      File(os.path.join(self.xs_vis_folder, xs_label + ".pvd"), "compressed") << xs_fun

    return True


  def get_Q(self, Q_fun, m=0, gto=0, vis=False):
    assert (0 <= gto < self.G)
    assert (0 <= m < self.M)

    timer = Timer("Fetching XS")

    try:
      xsd = self.xsd["Q"]
    except KeyError:
      return False

    Q_values = xsd[:, m, gto]

    # Assign xs data from materials to physical regions; skip if xs values for all materials are manually set to 0
    if numpy.any(Q_values > 0):
      Q_fun.vector()[:] = numpy.choose(self.regions_materials, Q_values)
    else:
      return False

    if vis:
      label = "Q"

      if self.G > 0:
        label += "_{}".format(gto)

      if self.M > 1 and xsd.ndim == 3:
        label += "_{}".format(m)

      Q_fun.rename(label, label)
      File(os.path.join(self.xs_vis_folder, label + ".pvd"), "compressed") << Q_fun

    return True

  def set_region_data(self, regions):
    if self.verb > 1: print pid + "Processing mesh regions."
    timer = Timer("Mesh regions processing")

    reg_set = set(numpy.unique(regions))  # faster than just set(regions);
                                          # will be amortized by repeated attempts to find an element within
    assert len(reg_set) > 0

    if not self.region_physical_name_map:
      for reg in reg_set:
        self.region_physical_name_map[reg] = str(reg)

    self.used_regions = numpy.empty(len(reg_set), dtype=numpy.int)

    self.matname_reg_map = defaultdict(list)
    num_reg_dofs = numpy.bincount(numpy.asarray(regions, dtype=numpy.int))

    i = 0
    for reg, phys_name in self.region_physical_name_map.iteritems():
      if num_reg_dofs[reg] > 0:
        self.used_regions[i] = reg
        i += 1

        try:
          mat_name = self.reg_name_mat_name_map[phys_name]
        except KeyError:
          mat_name = phys_name

        # Generate mapping from material name to corresponding region numbers (each material corresponds to one or more
        # regions with one set of cross-sections). In case of mat_name == phys_name,
        # len(self.matname_reg_map[mat_name] == 1.
        self.matname_reg_map[mat_name].append(reg)

    assert i == len(reg_set)

    # List of material names (str)
    self.material_names = self.matname_reg_map.keys()
    self.num_mat = len(self.material_names)

    # Mapping from regions to material numbers (positions in the `materials` list).
    # TODO: use only local range of region numbers
    self.reg_mat_map = numpy.empty(max(reg_set) + 1, dtype=numpy.int)
    self.reg_mat_map.fill(-1) # unset elements would correspond to region numbers that are not used in actual mesh
                              # partition

    for mat, (mat_name, regs) in enumerate(self.matname_reg_map.iteritems()):
      # convert each list in matname_reg_map to numpy.array so that it can be used for efficient indexing
      self.matname_reg_map[mat_name] = numpy.fromiter(regs, dtype=numpy.int)

      regs_array = self.matname_reg_map[mat_name]
      self.reg_mat_map[regs_array] = mat

  class OldData(Exception):
    pass

class CoreData(object):

  def __init__(self):
    super(CoreData, self).__init__()

    self.core_fraction = 1
    self.power = 1
    self.layered = False

  def __set_axial_layers_data(self, axial_layers_data):
    self.fuel_axial_range = axial_layers_data[0:2]
    self.num_fuel_layers = axial_layers_data[2]
    self.layer_boundaries = numpy.linspace(self.fuel_axial_range[0], self.fuel_axial_range[1], self.num_fuel_layers + 1)
    self.dz = numpy.diff(self.layer_boundaries)
    self.fuel_layer_midz = self.layer_boundaries[1:] - self.dz * 0.5 - self.fuel_axial_range[0]

  def parse_power_data(self, lines):
    """
    Parse core power data.

    :param list lines: list of lines to be parsed
    :return: index of the last processed line
    """
    for li, line in enumerate(lines):
      if line.startswith('*'):
        continue

      data = line.replace(',', ' ').replace(';', ' ').split()

      if len(data) != 2:
        continue

      try:
        data = map(float, data)
      except ValueError:
        warning("Invalid core power data - power normalization will not be available.")
      else:
        self.core_fraction = data[1]
        self.power = self.core_fraction * data[0]

      return li

    return -1

  def parse_axial_data(self, lines):
    """
    Parse data defining axial layers.

    :param list lines: list of lines to be parsed
    :return: index of the last processed line
    """
    for li, line in enumerate(lines):
      if line.startswith('*'):
        continue

      data = line.replace(',', ' ').replace(';', ' ').split()

      if len(data) != 3:
        continue

      try:
        data[0:2] = map(float, data[0:2])
        data[2] = int(data[2])
      except ValueError:
        warning("Invalid axial layer data - assuming 1 axial layer spanning the whole core height.")
      else:
        self.__set_axial_layers_data(data)
        self.layered = True

      return li

    return -1

  def info(self):
    if MPI.rank(comm) != 0:
      return

    print "Core power info:"
    print "  P_eff = {}".format(self.power)
    print "  P_tot = {}".format(self.power/self.core_fraction)

    if self.layered:
      print "Core layers info:"
      print "  dz = {}".format(self.dz)
      print "  layer boundaries: {}".format(self.layer_boundaries)
      print "  layer midpoints:  {}".format(self.fuel_layer_midz)

class BoundaryData(object):

  @classmethod
  def from_boundary_names_map(cls, bnd_map):
    bnd_idx_map = dict()

    for k, v in bnd_map.iteritems():
      bnd_idx_map[v] = k

    return cls(bnd_idx_map)

  @classmethod
  def from_file(cls, filename):
    bnd_idx_map = dict()

    try:
      with open(filename) as f:
        for line in f:
          data = line.split()
          assert (len(data) == 2)
          bnd_idx_map[data[1]] = int(data[0])
    except IOError:
      warning("File with boundary names not found.")
    except AssertionError:
      warning("File with boundary names has incorrect format.")

    return cls(bnd_idx_map)

  def __init__(self, bnd_idx_map):
    super(BoundaryData, self).__init__()

    self.boundary_names_idx_map = bnd_idx_map
    self.vacuum_boundaries = set()
    self.reflective_boundaries = set()
    self.incoming_fluxes = defaultdict(list)

  def info(self):
    if MPI.rank(comm) != 0:
      return

    print "\nBoundary names -> idx:"
    print " ", self.boundary_names_idx_map
    print "\nVacuum boundaries:", self.vacuum_boundaries
    print "\nReflective boundaries:", self.reflective_boundaries
    print "\nIncoming fluxes:"
    print self.incoming_fluxes

  def all_vacuum(self):
    return len(self.reflective_boundaries) == 0 and len(self.incoming_fluxes) == 0

  def parse_boundaries(self, lines):
    """
    Parse boundary conditions.

    :param list lines: list of lines to be parsed
    :return: index of the last processed line
    """
    l = -1
    while l < len(lines)-1:
      l += 1

      if lines[l].startswith('*'):
        continue

      data = lines[l].strip()

      if len(data) == 0:
        continue

      l += 1
      l += self.__parse_bc(data, lines[l:])

      if lines[l].lower() == "end":
        break

    return l

  def __parse_bc(self, boundary_name, lines):
    """
    Parse single boundary condition.

    :param str boundary_name: boundary label
    :param list lines: list of lines to be parsed
    :return: index of the last processed line
    """
    try:
      idx = self.boundary_names_idx_map[boundary_name]
    except KeyError:
      try:
        idx = int(boundary_name)
      except ValueError:
        # TODO: Invalid boundary error
        raise

    for li, line in enumerate(lines):
      if line.startswith("*"):
        continue

      data = line.strip().lower()

      if len(data) == 0:
        continue

      if data == "v" or data == "vacuum":
        self.vacuum_boundaries.add(idx)
        return li
      elif data == "r" or data == "reflective":
        self.reflective_boundaries.add(idx)
        return li

      data = data.replace(',', ' ').replace(';', ' ').split()
      set().update()
      try:
        self.incoming_fluxes[idx].append(map(float, data))
      except ValueError:
        # Encountered not-a-list-of-numbers => possibly a new boundary - finalize the inc. fluxes array for current
        # boundary and return
        self.incoming_fluxes[idx] = numpy.array(self.incoming_fluxes[idx])

        # Check consistency of incoming fluxes; intended use of attributes M, G (number of discrete
        # directions/groups) is to ensure consistency between specified boundary data and the Discretization object (
        # possibly repeating single group-flux vector loaded here M-times and dividing by 4pi (isotropic incoming flux).
        try:
          assert self.incoming_fluxes[idx].shape == (self.M, self.G)
        except AttributeError:
          try:
            self.M, self.G = self.incoming_fluxes[idx].shape
          except ValueError:
            #TODO: Invalid boundary data error
            raise

        return li

    return -1