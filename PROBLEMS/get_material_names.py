# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:56:24 2013

@author: mhanus
"""

import sys, os.path

from argparse import ArgumentParser

parser = ArgumentParser(description='Description.')
parser.add_argument('problem_name',
                    help="Problem name. Defines the subfolder in the 'problems' folder serving as the parent folder "
                         "for all problem data and results.")
parser.add_argument('-m', '--mesh', type=str, default="",
                    help='Mesh base filename. If not specified, mesh_base_filename = problem_name')
parser.add_argument('-v', '--verbosity', type=int, choices=range(6), default=0,
                    help='Output verbosity.')

args = parser.parse_args()

problem_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.problem_name)

mesh_base_filename = args.mesh if args.mesh else args.problem_name
mesh_filename = os.path.join(problem_folder, mesh_base_filename + ".msh")
mesh_file_last_mod = os.path.getmtime(mesh_filename)

if args.verbosity > 0: print "Extracting material and boundary names into separate files."

state = 0
num = 0
vol_markers = []
srf_markers = []
lin_markers = []
srf_ids = []
vol_ids = []
lin_ids = []

try:
  with open(os.path.join(problem_folder, mesh_filename), "rt") as f:
    reg_names_file = os.path.join(problem_folder, 'reg_names.txt')
    bnd_names_file = os.path.join(problem_folder, 'bnd_names.txt')

    if os.path.isfile(reg_names_file) and os.path.getmtime(reg_names_file) >= mesh_file_last_mod and \
       os.path.isfile(bnd_names_file) and os.path.getmtime(bnd_names_file) >= mesh_file_last_mod :
      sys.exit(0)

    for line in f:
      if "$PhysicalNames" in line:
        state = 1
        continue

      if "$EndPhysicalNames" in line:
        if num != 0:
          print "Warning: wrong number of material names found in the mesh file"
        break

      if state == 1:
        num = int(line)
        state = 2
        continue

      if state == 2:
        data = line.split()
        name = data[2].strip('"')
        if int(data[0]) == 3:
          vol_markers.append(name)
          vol_ids.append(int(data[1]))
        elif int(data[0]) == 2:
          srf_markers.append(name)
          srf_ids.append(int(data[1]))
        elif int(data[0]) == 1:
          lin_markers.append(name)
          lin_ids.append(int(data[1]))

        num -= 1

except IOError as e:
  print "Mesh file " + mesh_filename + " could not be loaded."
  print "Details: ERR#{0}: {1}".format(e.errno, e.strerror)
  sys.exit(-1)

if len(vol_markers) > 0:
  # 3D mesh

  assert len(srf_markers) > 0
  assert len(lin_markers) == 0

  with open(reg_names_file, "wt") as f:
    for idx, name in zip(vol_ids, vol_markers):
      print>> f, idx, name

  with open(bnd_names_file, "wt") as f:
    for idx, name in zip(srf_ids, srf_markers):
      print>> f, idx, name

else:
  # 2D mesh

  assert len(srf_markers) > 0
  assert len(lin_markers) > 0

  with open(reg_names_file, "wt") as f:
    for idx, name in zip(srf_ids, srf_markers):
      print>> f, idx, name

  with open(bnd_names_file, "wt") as f:
    for idx, name in zip(lin_ids, lin_markers):
      print>> f, idx, name