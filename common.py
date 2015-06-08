import errno
import os.path
import sys
import numpy

try:
  import dolfin
  import dolfin.cpp.common as dolfin_common
  from dolfin.cpp.common import warning
  from dolfin.cpp.la import has_linear_algebra_backend
except ImportError:
  raise ImportError('Problem with Dolfin Installation')

backend = dolfin.parameters.linear_algebra_backend = "PETSc"

dolfin_version_id = "".join(dolfin_common.dolfin_version().split('.')[0:2])

if int(dolfin_version_id) < 12:
  print "At least DOLFIN 1.2.0 is required."
  sys.exit(-1)
if not has_linear_algebra_backend(backend):
  print "DOLFIN has not been configured with", backend, "support."
  sys.exit(-1)


#------------------------------------------------------------------------------#
#                             ARGUMENT CHECKING                                #
#------------------------------------------------------------------------------#

from argparse import ArgumentTypeError

def check_file_name(name):
  if name != "" and not os.path.isfile(name):
    raise ArgumentTypeError("%r is not a valid file name" % name)
  return name

def check_folder_name(name):
  if name != "" and not os.path.isdir(name):
    raise ArgumentTypeError("%r is not a valid folder name" % name)
  return name

def check_sn_order(value):
  ivalue = int(value)
  if ivalue < 0 or ivalue > 32 or (ivalue % 2) != 0:
    raise ArgumentTypeError("%r is not a valid SN order (only even positive orders up to 32 are allowed)" % value)
  return ivalue

def check_scattering_order(value, SN_order=32):
  ivalue = int(value)
  if ivalue < 0 or ivalue > SN_order:
    raise ArgumentTypeError("%r is not a valid scattering order (only positive orders up to the SN order are allowed)"
                            % value)
  return ivalue


#------------------------------------------------------------------------------#
#                                 MPI STUFF                                    #
#------------------------------------------------------------------------------#


from mpi4py.MPI import __TypeDict__, COMM_WORLD, SUM, MIN, MAX

MPI_type = lambda array: __TypeDict__[array.dtype.char]
comm = COMM_WORLD
pid = "Process " + str(comm.rank) + ": " if comm.size > 1 else ""


def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc: # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else: 
      raise

def print0(x, end="\n", _comm=COMM_WORLD):
  if _comm.rank == 0:
    print str(x)+end,

def MPI_sum(arg,ax=None):
  if ax is not None:
    r = numpy.atleast_1d(numpy.sum(arg,ax))
  else:
    r = numpy.atleast_1d(numpy.sum(arg))
  rout = numpy.zeros_like(r)

  comm.Allreduce([r, MPI_type(r)], [rout, MPI_type(rout)], op=SUM)

  if rout.size == 1:
    return rout[0]
  else:
    return rout

def MPI_sum0(arg,ax=None):
  if ax is not None:
    r = numpy.atleast_1d(numpy.sum(arg,ax))
  else:
    r = numpy.atleast_1d(numpy.sum(arg))

  if comm.rank == 0:
    rout = numpy.zeros_like(r)
  else:
    rout = None

  comm.Reduce([r, MPI_type(r)], [rout, MPI_type(rout)], op=SUM, root=0)

  if rout.size == 1:
    return rout[0]
  else:
    return rout

def MPI_max(arg,ax=None):
  if ax is not None:
    r = numpy.atleast_1d(numpy.max(arg,ax))
  else:
    r = numpy.atleast_1d(numpy.max(arg))
  rout = numpy.zeros_like(r)

  comm.Allreduce([r, MPI_type(r)], [rout, MPI_type(rout)], op=MAX)

  if rout.size == 1:
    return rout[0]
  else:
    return rout

def MPI_min(arg,ax=None):
  if ax is not None:
    r = numpy.atleast_1d(numpy.min(arg,ax))
  else:
    r = numpy.atleast_1d(numpy.min(arg))
  rout = numpy.zeros_like(r)

  comm.Allreduce([r, MPI_type(r)], [rout, MPI_type(rout)], op=MIN)

  if rout.size == 1:
    return rout[0]
  else:
    return rout


def MPI_local_range(N):
  if int(dolfin_version_id) < 14:
    return dolfin_common.MPI.local_range(N)
  else:
    return dolfin_common.MPI.compute_local_range(comm.rank, N, comm.size)

def MPI_index(index, N):
  """
  Return which process owns index (inverse of local_range) and local position of the index at that process.

  :param int index: global index
  :param int N: length of the array split between processes
  :return: (owner, local_index) tuple
  :rtype: int, int
  """
  assert index < N

  # compute number of items per process and remainder
  n,r = divmod(N, comm.size)

  # first r processes own n+1 indices
  if index < r*(n+1):
    return divmod(index, n+1)

  # remaining processes own n indices
  rem_owner, loc_idx = divmod(index - r*(n+1), n)
  return r + rem_owner, loc_idx

#------------------------------------------------------------------------------#
#                            CONVERGENCE CHECKING                              #
#------------------------------------------------------------------------------#  
  
delta = lambda x,y: MPI_max(numpy.linalg.norm( x - y, ord=numpy.Inf ))\
                   / MPI_max(numpy.linalg.norm( x, ord=numpy.Inf ))

delta0 = lambda x,y:  numpy.linalg.norm( x - y, ord=numpy.Inf ) / numpy.linalg.norm( x, ord=numpy.Inf )

delta0_dict = lambda x,y: max(map(delta0, x.itervalues(), y.itervalues()))
delta0_list_dict = lambda x,y: max(map(lambda a,b: abs((a-b)/b), x.itervalues(), y.itervalues()))  


#------------------------------------------------------------------------------#
#                                OTHER UTILITIES                               #
#------------------------------------------------------------------------------#

# Pretty printing numpy arrays:
#   with printoptions(precision=3, suppress=True, strip_zeros=False):
#     print(x)
#    
import numpy.core.arrayprint as arrayprint
import contextlib

@contextlib.contextmanager
def printoptions(strip_zeros=True, **kwargs):
  origcall = arrayprint.FloatFormat.__call__
  def __call__(self, x, _strip_zeros=strip_zeros):
      return origcall.__call__(self, x, _strip_zeros)
  arrayprint.FloatFormat.__call__ = __call__
  original = numpy.get_printoptions()
  numpy.set_printoptions(**kwargs)
  yield 
  numpy.set_printoptions(**original)
  arrayprint.FloatFormat.__call__ = origcall


#------------------------------------------------------------------------------#
#                                ERROR PRINTOUT                                #
#------------------------------------------------------------------------------#

# based on dolfin/log/Logger::dolfin_error

def coupled_solver_error(location, task, reason):
  import traceback

  s = ""
  s += "\n\n"
  s += "*** -------------------------------------------------------------------------"
  s += "\n"
  s += "*** Coupled solver encountered an error. If you are not able to resolve this "
  s += "\n"
  s += "*** issue using the information listed below, you can ask for help at"
  s += "\n"
  s += "***"
  s += "\n"
  s += "***     mhanus@kma.zcu.cz"
  s += "\n"
  s += "***"
  s += "\n"
  s += "*** Remember to include the error message listed below and, if possible,"
  s += "\n"
  s += "*** include input data needed to reproduce the error."
  s += "\n"
  s += "***"
  s += "\n"
  s += "*** -------------------------------------------------------------------------"
  s += "\n"
  s += "*** "
  s += "Error:   Unable to "
  s += task + "." + "\n"
  if reason:
    s += "*** " + "Reason:  "
    rl = reason.splitlines(False)
    s += rl[0]
    for l in rl[1:]:
      s += "\n***          " + l
  s += "\n*** "
  s += "Where:   This error was encountered inside "
  s += location.replace(".pyc", ".py").replace(".pyo", ".py") + "."
  s += "\n"
  s += "*** "
  s += "Process: "
  s += str(comm.rank) + "\n"
  s += "*** "
  s += "\n"
  s += "*** "
  s += "DOLFIN version: "
  s += dolfin_common.dolfin_version()
  s += "\n"
  s += "*** "
  s += "Git changeset:  "
  s += dolfin_common.git_commit_hash()
  s += "\n"
  s += "*** "
  s += "-------------------------------------------------------------------------"
  s += "\n\n"
  s += traceback.format_exc()
  s += "\n"

  print s
  sys.exit(-1)
