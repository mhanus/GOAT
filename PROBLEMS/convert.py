try:
  from dolfin.cpp.mesh import Mesh, MeshFunction
  from dolfin.cpp.io import File
except ImportError:
  raise ImportError('Problem with Dolfin Installation')

import sys, os

if len(sys.argv) == 0 or len(sys.argv) > 3:
  print "Correct usage: python convert.py mesh_base_filename [verbosity_level]"
  sys.exit(-1)

try:
  mesh_base_filename = os.path.abspath(sys.argv[1])
except IndexError:
  print "You need to provide mesh base filename."
  sys.exit(-1)

mesh_file = mesh_base_filename + ".xml"
physical_regions_file = mesh_base_filename + "_physical_region.xml"
facet_regions_file = mesh_base_filename + "_facet_region.xml"

try:
  verb = int(sys.argv[2])
except ValueError:
  print "Incorrect verbosity level -- setting to 0"
  verb = 0
except IndexError:
  verb = 0

def save_new_meshfunction(_orig_file):
  _orig_folder, _orig_fname = os.path.split(_orig_file)
  _new_fname = _orig_fname+"_new.xml"
  _new_file = os.path.join(orig_folder, _new_fname)
  
  File(_new_file) << MeshFunction("size_t", mesh, _orig_file)
  
  os.remove(_orig_file)
  os.rename(_new_file, _orig_file)
  
class InvalidFormat(Exception):
  pass

try:
  
  with open(mesh_file) as f:
    if 'mesh' not in f.read(1024):
      raise InvalidFormat
  with open(physical_regions_file) as f:
    if 'mesh_value_collection' not in f.read(1024):
      raise InvalidFormat
  with open(facet_regions_file) as f:
    if 'mesh_value_collection' not in f.read(1024):
      raise InvalidFormat   
  
  # If we got to this point, all three required files describing the mesh were found and
  # have the correct format for parallel execution -- nothing more needs to be done here. 

except IOError as e:
  print "One or more of the required mesh-description files could not be loaded."
  print "Details: ERR#{0}: {1}".format(e.errno, e.strerror)
  sys.exit(-1)
   
except InvalidFormat:
  import os
        
  if verb > 0: print "Converting old-style DOLFIN mesh-specification files to new ones, suitable for parallel computation:"
  
  if verb > 1: print "   mesh data..."

  orig_file = mesh_file
  orig_folder, orig_fname = os.path.split(orig_file)
  new_fname = orig_fname+"_new.xml"
  new_file = os.path.join(orig_folder, new_fname)
  
  mesh = Mesh(orig_file)
  File(new_file) << mesh
  
  os.remove(orig_file)
  os.rename(new_file, orig_file) 
  
  if verb > 1: print "   physical data..."
  
  save_new_meshfunction(physical_regions_file)
    
  if verb > 1: print "   boundary data..."
  
  save_new_meshfunction(facet_regions_file)

  if verb > 1: print "Done."