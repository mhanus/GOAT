#include "OrdinatesData.h"
#include <boost/range/irange.hpp>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace dolfin;

inline double sqr(double x) { return x*x; }

/* Ordinates in the four octants of the upper hemisphere are ordered as
 *
 *      13         12
 *     17  5     4  16
 *    21  9 1   0 8  20
 *            o
 *    22 10 2   3 11 23
 *     18  6     7  19
 *       14        15
**/
OrdinatesData::OrdinatesData(unsigned int N, unsigned int D, const string& filename) : N(N), M(N*(N+2)), D(D)
{
	if (D < 2 || D > 3)
		dolfin_error("OrdinatesData.cpp",
		             "construct the OrdinatesData object",
		             "Only 2D or 3D ordinates currently supported");

	if (D == 2)
		M /= 2;

	ifstream ifs(filename.c_str());

	if (ifs.fail())
		dolfin_error("OrdinatesData.cpp",
		             "construct the OrdinatesData object",
		             "Discrete ordinates could not be loaded from %s", filename.c_str());

	string tmp;
	getline(ifs, tmp);
	getline(ifs, tmp);

	int read_n;
	do
	{
		ifs >> read_n;
		if (read_n == N)
			break;

		getline(ifs, tmp);
		for (int n = 0; n < read_n; n++)
			getline(ifs, tmp);
		getline(ifs, tmp);
	}
	while (!ifs.eof());

	if (ifs.eof())
		dolfin_error("OrdinatesData.cpp",
		             "construct the OrdinatesData object",
		             "Required set of discrete ordinates could not be found in %s", filename.c_str());

	double *mu_base = new double [N/2];

	getline(ifs, tmp);
	for (int n = 0; n < N/2; n++)
		ifs >> mu_base[n];

	getline(ifs, tmp, '"');
	getline(ifs, tmp);
	getline(ifs, tmp);

	do
	{
		ifs >> read_n;
		if (read_n == N)
			break;

		getline(ifs, tmp);
		for (int n = 0; n < read_n; n++)
			getline(ifs, tmp);
		getline(ifs, tmp);
	}
	while (!ifs.eof());

	if (ifs.eof())
		dolfin_error("OrdinatesData.cpp",
		             "construct the OrdinatesData object",
		             "Required set of weights could not be found in %s", filename.c_str());

	double *wt = new double [N/2];

	getline(ifs, tmp);
	for (int n = 0; n < N/2; n++)
		ifs >> wt[n];

	ifs.close();

	xi.reserve(M);
	eta.reserve(M);
	mu.reserve(M);
	pw.reserve(M);
	reflections_about_x.reserve(M);
	reflections_about_y.reserve(M);
	if (D == 3) reflections_about_z.reserve(M);

	int dir = 0;

	// upper hemisphere

	for (int n = 1; n <= N/2; n++) // for each polar level
	{
		for (int i = 1; i <= n; i++) // for each azimuthal point in the first octant
		{
			double omega = (2*n - 2*i + 1)/(2.*n) * M_PI/2.;
			double xi1q = sqrt(1-sqr(mu_base[n-1])) * cos(omega);
			double eta1q = sqrt(1-sqr(mu_base[n-1])) * sin(omega);

			// octant 1, ordinate index dir
			xi.push_back( xi1q );
			eta.push_back( eta1q );
			reflections_about_x.push_back( dir + 3 );
			reflections_about_y.push_back( dir + 1 );

			// octant 2, ordinate index dir+1
			xi.push_back(-xi1q );
			eta.push_back( eta1q );
			reflections_about_x.push_back( dir + 2 );
			reflections_about_y.push_back( dir );

			// octant 3, ordinate index dir+2
			xi.push_back(-xi1q );
			eta.push_back(-eta1q );
			reflections_about_x.push_back( dir + 1 );
			reflections_about_y.push_back( dir + 3 );

			// octant 4, ordinate index dir+3
			xi.push_back( xi1q );
			eta.push_back(-eta1q );
			reflections_about_x.push_back( dir );
			reflections_about_y.push_back( dir + 2 );

			for (int j = 0; j < 4; j++)
			{
				mu.push_back(mu_base[n-1]);
				pw.push_back( wt[n-1] / n); // equal weights in each polar level, summing up to 4pi over the whole sphere.
			}

			dir+=4;
		}
	}

	if (D == 3)
	{
		// lower hemisphere

		xi.insert(xi.end(), xi.begin(), xi.end());
		eta.insert(eta.end(), eta.begin(), eta.end());
		pw.insert(pw.end(), pw.begin(), pw.end());

		std::vector<double> base_aux(M/2);// base_aux.reserve(M/2);
		transform(mu.begin(), mu.end(), base_aux.begin(), negate<double>());
		mu.insert(mu.end(), base_aux.begin(), base_aux.end());

		//TODO: use zip_iterators or something similar to do the following steps at once

		transform(reflections_about_x.begin(), reflections_about_x.end(), base_aux.begin(), bind2nd(plus<int>(), M/2));
		reflections_about_x.insert(reflections_about_x.end(), base_aux.begin(), base_aux.end());
		transform(reflections_about_y.begin(), reflections_about_y.end(), base_aux.begin(), bind2nd(plus<int>(), M/2));
		reflections_about_y.insert(reflections_about_y.end(), base_aux.begin(), base_aux.end());

		boost::integer_range<int> aux = boost::irange<int>(M/2,M);
		reflections_about_z.assign(aux.begin(), aux.end());
		aux = boost::irange<int>(0,M/2);
		reflections_about_z.insert(reflections_about_z.end(), aux.begin(), aux.end());
	}

	delete [] mu_base;
	delete [] wt;
}

namespace dolfin {
	ostream& operator<<(ostream& os, const OrdinatesData& odata)
	{
		os << "_______________________________________________________" << endl;
		os << "                Discrete ordinates (" << odata.D << "D)" << endl;
		os << "                        N = " << odata.N << endl;
		os << "-------------------------------------------------------" << endl;


		int m = 0;
		vector<double>::const_iterator xi = odata.xi.begin();
		vector<double>::const_iterator eta = odata.eta.begin();
		vector<double>::const_iterator mu = odata.mu.begin();
		vector<double>::const_iterator pw = odata.pw.begin();
		for ( ; xi != odata.xi.end(); ++xi, ++eta, ++mu, ++pw)
		{
			os << endl << *xi << ", " << *eta << ", " << *mu << ", " << *pw << endl;
			os << " --- " << odata.xi[odata.reflections_about_x[m]] << ", "
                    << odata.eta[odata.reflections_about_x[m]] << ", "
                    << odata.mu[odata.reflections_about_x[m]] << endl;
			os << "  |  " << odata.xi[odata.reflections_about_y[m]] << ", "
			              << odata.eta[odata.reflections_about_y[m]] << ", "
			              << odata.mu[odata.reflections_about_y[m]] << endl;

			if (odata.D == 3)
				os << "  /  " << odata.xi[odata.reflections_about_z[m]] << ", "
				              << odata.eta[odata.reflections_about_z[m]] << ", "
				              << odata.mu[odata.reflections_about_z[m]] << endl;

			m++;
		}

		os << endl << m << " ordinates loaded (M = " << odata.M << ")." << endl;

		double sum = 0.0;
		for (int n = 0; n < odata.M; n++)
			sum += odata.pw[n];

		os << "sum of weights over the whole sphere: " << (odata.D == 3 ? sum : 2*sum) << endl;

		return os;
	}
}

void OrdinatesData::print_info() const
{
	stringstream ss;
	ss << *this;
	dolfin::info(ss.str());
}

void OrdinatesData::write_pw(const string& filename)
{
	FILE* fp;
	fp = fopen(filename.c_str(), "wt");
	fprintf(fp, "pw = [ \n");
	for (int n = 0; n < M; n++)
		fprintf(fp, "\t%1.15f\n", pw[n]);
	fprintf(fp, "];");
	fclose(fp);

	cout << "weights written to: " << filename << endl << endl;
}

std::vector<double> OrdinatesData::get_ordinate(int n) const
{
	std::vector<double> ret;
	ret.reserve(D);
	ret.push_back(xi[n]);
	ret.push_back(eta[n]);
	if (D == 3)
	  ret.push_back(mu[n]);

	return ret;
}
