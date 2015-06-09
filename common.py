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

if int(dolfin_version_id) < 15:
  print "At least DOLFIN 1.5.0 is required."
  sys.exit(-1)
if not has_linear_algebra_backend(backend):
  print "DOLFIN has not been configured with", backend, "support."
  sys.exit(-1)

#------------------------------------------------------------------------------#
#                             ARGUMENT CHECKING                                #
#------------------------------------------------------------------------------#

from argparse import ArgumentTypeError

def check_sn_order(value):
  ivalue = int(value)
  if ivalue < 0 or ivalue > 32 or (ivalue % 2) != 0:
    raise ArgumentTypeError("%r is not a valid SN order (only even positive orders up to 32 are allowed)" % value)
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

#------------------------------------------------------------------------------#
#                            CONVERGENCE CHECKING                              #
#------------------------------------------------------------------------------#  
  
delta = lambda x,y: MPI_max(numpy.linalg.norm( x - y, ord=numpy.Inf ))\
                   / MPI_max(numpy.linalg.norm( x, ord=numpy.Inf ))

delta0 = lambda x,y:  numpy.linalg.norm( x - y, ord=numpy.Inf ) / numpy.linalg.norm( x, ord=numpy.Inf )

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

def print_timings(timings_table, screen=True, txt_file="", tex_file=""):
  if screen:
    print pid + "\n\n" + timings_table.str(True) + "\n"

  if txt_file:
    timings_table_str = pid + "\n\n" + timings_table.str(True)
    timings_table_str = comm.gather(timings_table_str, root=0)

    if comm.rank == 0:
      timings_table_str = "\n___________________________________________________________\n".join(timings_table_str)

      try:
        with open(txt_file, "wt") as f:
          print>> f, timings_table_str
      except Exception as e:
        print "Writing timing results to {} failed: {}".format(txt_file, e)

  if tex_file:
    timings_table_tex = pid + "\n\n" + timings_table.str_latex()
    timings_table_tex = comm.gather(timings_table_tex, root=0)

    if comm.rank == 0:
      timings_table_tex = "\n___________________________________________________________\n".join(timings_table_tex)

      try:
        with open(tex_file, "wt") as f:
          print>> f, timings_table_tex
      except Exception as e:
        print "Writing timing results to {} failed: {}".format(tex_file, e)

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
  s += "*** GOAT solver encountered an error. If you are not able to resolve this "
  s += "\n"
  s += "*** issue using the information listed below, you can ask for help at"
  s += "\n"
  s += "***"
  s += "\n"
  s += "***     mhanus@tamu.edu"
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

