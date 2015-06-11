// TODO: Make COO's take into account the direction dimension

#include "PETSc_utils.h"
#include <slepcversion.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/parameter/GlobalParameters.h>
using namespace dolfin;

void PETScMatrixExt::diag(std::shared_ptr<dolfin::PETScVector> vec) const
{
	Vec d = vec->vec();
	assert(d);
	MatGetDiagonal(_matA, d);
}

std::size_t PETScMatrixExt::local_nnz() const
{
	MatInfo info;
	MatGetInfo(_matA, MAT_LOCAL, &info);
	return std::size_t(info.nz_used);
}

std::size_t PETScMatrixExt::global_nnz() const
{
	MatInfo info;
	MatGetInfo(_matA, MAT_GLOBAL_SUM, &info);
	return std::size_t(info.nz_used);
}

/*
void PETScMatrixExt::mult(const PETScVector& xx, PETScVector& yy) const
{
  dolfin_assert(_matA);

  if (PETScBaseMatrix::size(1) != xx.size())
  {
    dolfin_error("PETSc_utils.cpp",
                 "compute matrix-vector product with PETSc matrix",
                 "Non-matching dimensions for matrix-vector product");
  }

  // Resize RHS if empty
  if (yy.size() == 0)
  	init_vector(yy, 0);

  if (size(0) != yy.size())
  {
    dolfin_error("PETSc_utils.cpp",
                 "compute matrix-vector product with PETSc matrix",
                 "Vector for matrix-vector result has wrong size");
  }

  PetscErrorCode ierr = MatMult(_matA, xx.vec(), yy.vec());
  if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "MatMult");
}
*/
















void COO::init(const PETScMatrix& mat)
{
	_matA = mat.mat();

	dolfin_assert(_matA);

	row_range = mat.local_range(0);
	local_size = row_range.second - row_range.first;

	MatInfo info;
	MatGetInfo(_matA, MAT_LOCAL, &info);
	local_nnz = std::size_t(info.nz_used);

	rows.reserve(local_nnz);
	columns.reserve(local_nnz);
	values.reserve(local_nnz);
}

COO::COO(const PETScMatrix& mat)
{
	init(mat);

	for (std::size_t r = 0; r < local_size; r++)
		process_row(row_range.first+r);
}

COO::COO(const PETScMatrix& mat, int N, int gto, int gfrom, bool negate)
{
	init(mat);

	for (std::size_t r = 0; r < local_size; r++)
		process_row(row_range.first+r);

	std::transform(rows.begin(), rows.end(), rows.begin(), bind2nd(std::plus<int>(), N*gto));
	std::transform(columns.begin(), columns.end(), columns.begin(), bind2nd(std::plus<int>(), N*gfrom));
	if (negate)
		std::transform(values.begin(), values.end(), values.begin(), std::negate<double>());
}

void COO::process_row(std::size_t row)
{
	PetscErrorCode ierr;
	const PetscInt *cols = 0;
	const double *vals = 0;
	PetscInt ncols = 0;
	ierr = MatGetRow(_matA, row, &ncols, &cols, &vals);
	if (ierr != 0) PETScObject::petsc_error(ierr, "PETSc_utils.cpp", "MatGetRow");

	// Insert values to std::vectors
	rows.insert(rows.end(), ncols, row);
	columns.insert(columns.end(), cols, cols + ncols);
	values.insert(values.end(), vals, vals + ncols);

	ierr = MatRestoreRow(_matA, row, &ncols, &cols, &vals);
	if (ierr != 0) PETScObject::petsc_error(ierr, "PETSc_utils.cpp", "MatRestoreRow");
}














COO2::COO2(const PETScMatrix& mat, int G, int gto, int gfrom, bool negate)
{
	init(mat);

	for (std::size_t r = 0; r < local_size; r++)
		process_row(row_range.first+r, G, gto, gfrom, negate);
}

void COO2::process_row(std::size_t row, int G, int gto, int gfrom, bool negate)
{
	PetscErrorCode ierr;
	const PetscInt *cols = 0;
	const double *vals = 0;
	PetscInt ncols = 0;
	ierr = MatGetRow(_matA, row, &ncols, &cols, &vals);
	if (ierr != 0) PETScObject::petsc_error(ierr, "PETSc_utils.cpp", "MatGetRow");

	// Insert values to std::vectors
	for (int i = 0; i < ncols; i++)
	{
		rows.push_back(row*G + gto);
		columns.push_back(cols[i]*G + gfrom);
		values.push_back(negate ? -vals[i] : vals[i]);
	}

	ierr = MatRestoreRow(_matA, row, &ncols, &cols, &vals);
	if (ierr != 0) PETScObject::petsc_error(ierr, "PETSc_utils.cpp", "MatRestoreRow");
}













std::size_t GeneralizedEigenSolver::get_number_converged() const
{
  dolfin::la_index num_conv;
  EPSGetConverged(eps, &num_conv);
  return num_conv;
}
std::size_t GeneralizedEigenSolver::get_iteration_number() const
{
  dolfin::la_index num_iter;
  EPSGetIterationNumber(eps, &num_iter);
  return num_iter;
}
//-----------------------------------------------------------------------------
GeneralizedEigenSolver::GeneralizedEigenSolver(std::shared_ptr<const PETScMatrix> A,
                                               std::shared_ptr<const PETScMatrix> B)
{
  dolfin_assert(A->size(0) == A->size(1));
  dolfin_assert(B->size(0) == A->size(0));
  dolfin_assert(B->size(1) == A->size(1));

  // Set default parameter values
  try
  {
  	this->parameters = dolfin::parameters("flux_module")("eigensolver");
  }
  catch(...)
  {
  	info("Parameters set 'flux_module.eigensolver' has not been added to the global set 'dolfin'; "
  	     "you will not be able to set the parameters from command command line via --eigensolver.param param_val.");
  	this->parameters = default_parameters();
  }

	// Set up solver environment
  EPSCreate(PETSC_COMM_WORLD, &eps);

  if (this->parameters["switch_mat"])
  {
  	_matA = B;
  	_matB = A;
  	EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE);
  }
  else
  {
  	_matA = A;
  	_matB = B;
  	EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE);
  }
}
//-----------------------------------------------------------------------------
GeneralizedEigenSolver::~GeneralizedEigenSolver()
{
  // Destroy solver environment
  if (eps)
    EPSDestroy(&eps);
}
//-----------------------------------------------------------------------------
void GeneralizedEigenSolver::solve(std::size_t n)
{
  dolfin_assert(_matA);
  dolfin_assert(_matB);

  // Associate matrix (matrices) with eigenvalue solver
  dolfin_assert(_matA->size(0) == _matA->size(1));
  dolfin_assert(_matB->size(0) == _matB->size(1) && _matB->size(0) == _matA->size(0));

  EPSSetOperators(eps, _matA->mat(), _matB->mat());

  // Set number of eigenpairs to compute
  dolfin_assert(n <= _matA->size(0));
  EPSSetDimensions(eps, n, PETSC_DECIDE, PETSC_DECIDE);

  // Set parameters from local parameters
  read_parameters();

  // Set parameters from PETSc parameter database
  EPSSetFromOptions(eps);

  // Solve
  EPSSolve(eps);

  // Check for convergence
  EPSConvergedReason reason;
  EPSGetConvergedReason(eps, &reason);
  if (reason < 0)
    warning("Eigenvalue solver did not converge");

  // Report solver status

  #if SLEPC_VERSION_MAJOR == 3 && SLEPC_VERSION_MINOR < 4
  const EPSType eps_type = NULL;
  #else
  EPSType eps_type = NULL;
  #endif
  EPSGetType(eps, &eps_type);

  int verb = this->parameters["verbosity"];
  if (verb > 2)
  {
  	dolfin::la_index num_iterations = 0;
    EPSGetIterationNumber(eps, &num_iterations);
  	info("Eigenvalue solver (%s) converged in %d iterations.", eps_type, num_iterations);
  }
  if (verb > 2)
  {
  	dolfin::la_index ops, dots, lits;
  	EPSGetOperationCounters(eps, &ops, &dots, &lits);
		info(" - number of operator applications / dot products / inner iterations: %d / %d / %d",
				 ops, dots, lits);
  }

}
//-----------------------------------------------------------------------------
double GeneralizedEigenSolver::get_eigenpair(PETScVector& r, std::size_t i) const
{
  double lr;
  const dolfin::la_index ii = static_cast<dolfin::la_index>(i);

  // Get number of computed eigenvectors/values
  dolfin::la_index num_computed_eigenvalues;
  EPSGetConverged(eps, &num_computed_eigenvalues);

  if (ii < num_computed_eigenvalues)
  {
    dolfin_assert(_matA);
    _matA->init_vector(r, 0);
    dolfin_assert(r.vec());
    EPSGetEigenpair(eps, ii, &lr, NULL, r.vec(), NULL);
  }
  else
  {
    dolfin_error("PETSc_utils.cpp",
                 "extract eigenpair from the generalized eigenvalue solver",
                 "Requested eigenpair (%d) has not been computed", i);
  }

  // Switch sign if negative.
  double sum;
  VecSum(r.vec(), &sum);
  if (sum < 0)
  	VecScale(r.vec(), -1.0);

  return lr;
}

//-----------------------------------------------------------------------------
void GeneralizedEigenSolver::read_parameters()
{
	set_problem_type(this->parameters["problem_type"]);
  set_tolerance(this->parameters["tol"], this->parameters["max_niter"]);
}
//-----------------------------------------------------------------------------
void GeneralizedEigenSolver::set_problem_type(std::string type)
{
  if (type == "gen_hermitian")
    EPSSetProblemType(eps, EPS_GHEP);
  else if (type == "gen_non_hermitian")
    EPSSetProblemType(eps, EPS_GNHEP);
  else if (type == "pos_gen_non_hermitian")
    EPSSetProblemType(eps, EPS_PGNHEP);
  else
  {
    dolfin_error("PETSc_utils.cpp",
                 "set problem type for generalized eigensolver",
                 "Unknown problem type (\"%s\")", type.c_str());
  }
}
//-----------------------------------------------------------------------------
void GeneralizedEigenSolver::set_tolerance(double tolerance, std::size_t maxiter)
{
  dolfin_assert(tolerance > 0.0);
  EPSSetTolerances(eps, tolerance, maxiter);
}
