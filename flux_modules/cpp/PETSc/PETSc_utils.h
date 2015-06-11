#ifndef _PETSC_UTILS_H	// #pragma once doesn't work here (this whole file will be inserted into .cpp file)
#define _PETSC_UTILS_H

#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/SLEPcEigenSolver.h>
#include <dolfin/la/PETScOptions.h>

namespace dolfin {

	class PETScMatrixExt : public PETScMatrix
	{
		public:
			PETScMatrixExt(std::shared_ptr<dolfin::PETScMatrix> mat) : PETScMatrix(mat->mat()) {}

			void diag(std::shared_ptr<dolfin::PETScVector> vec) const;
			std::size_t local_nnz() const;
			std::size_t global_nnz() const;
			//void mult(const PETScVector& xx, PETScVector& yy) const;
	};

  class COO
  {
    public:
	  COO() {};
      COO(const PETScMatrix& mat);
      COO(const PETScMatrix& mat, int N, int gto, int gfrom, bool negate);

	    std::vector<double> get_vals() const { return values; }
      std::vector<int> get_cols() const { return columns; }
      std::vector<int> get_rows() const { return rows; }
      std::size_t get_local_nnz() const { return local_nnz; }

    private:
      void process_row(std::size_t row);

    protected:
      Mat _matA;
      std::pair<std::size_t,std::size_t> row_range;
      std::size_t local_size;
      std::size_t local_nnz;

      std::vector<double> values;
      std::vector<int> columns;
      std::vector<int> rows;

      void init(const PETScMatrix& mat);
  };

  class COO2 : public COO
	{
		public:
			COO2(const PETScMatrix& mat, int G, int gto, int gfrom, bool negate);

		private:
			void process_row(std::size_t row, int G, int gto, int gfrom, bool negate);
	};

  class GeneralizedEigenSolver : public Variable, public PETScObject
  {
  public:

  	/// Constructor
  	GeneralizedEigenSolver(std::shared_ptr<const PETScMatrix> A, std::shared_ptr<const PETScMatrix> B);

    /// Destructor
		~GeneralizedEigenSolver();

    void set_initial_space(const std::shared_ptr<const PETScVector> v) {
        Vec v0 = v->vec(); EPSSetInitialSpace(eps, 1, &v0);
    }

	/// Compute the n first generalized eigenpairs of the matrices A, B (solve Ax = \lambda B x)
	void solve(std::size_t n = 1);

	/// Get the first eigenpair of the actual eigenproblem:
	/// (x_1,\lambda_1) of Ax = \lambda B x if parameters["switch_mat"] == false, Bx = \lambda A x otherwise)
	double get_first_eigenpair(PETScVector& r) const { return get_eigenpair(r, 0); }

	/// Get the first eigenpair of the problem Ax = \lambda B x
	double get_first_eigenpair_AB(PETScVector& r) const { return get_eigenpair_AB(r, 0); }

	/// Get eigenpair i of the actual eigenproblem
	double get_eigenpair(PETScVector& r, std::size_t i) const;

	/// Get eigenpair i of the problem Ax = \lambda B x
	double get_eigenpair_AB(PETScVector& r, std::size_t i) const {
		return this->parameters["switch_mat"] ? 1./(get_eigenpair(r, i)) : get_eigenpair(r, i);
	}

	/// Set eigenvalue shift for the problem Ax = \lambda B x
	void set_shift_AB(double shift) const {
		PETScOptions::set("st_shift", parameters["switch_mat"] ? 1./shift : shift);
	}

	/// Get the number of iterations used by the solver
	std::size_t get_iteration_number() const;

	/// Get the number of converged eigenvalues
	std::size_t get_number_converged() const;

	/// Get the string describing the actual eigenproblem
	const char* get_actual_problem_description() const {
		return parameters["switch_mat"] ? "B.x = lam.A.x" : "A.x = lam.B.x";
	}

	/// Default parameter values
	static Parameters default_parameters()
	{
		Parameters p("eigensolver");

		p.add("problem_type",       		 "gen_non_hermitian");
		p.add("tol",			          		 1e-12);
		p.add("max_niter", 							 50);
		p.add("verbosity",          		 0);
		p.add("adaptive_shifting",  		 false);
		p.add("inner_solver_adaptive_tol_multiplier", 0);
		p.add("switch_mat",         		 false);

		return p;
	}

  protected:

		/// Callback for changes in parameter values
		void read_parameters();

		/// Set problem type (used for SLEPc internals)
		void set_problem_type(std::string type);

		/// Set tolerance
		void set_tolerance(double tolerance, std::size_t maxiter);

		// Operators (Ax = \lambda B x if parameters["switch_mat"] == false, Bx = \lambda A x otherwise)
		std::shared_ptr<const PETScMatrix> _matA;
		std::shared_ptr<const PETScMatrix> _matB;

		// SLEPc solver pointer
		EPS eps;

	};
}

#endif
