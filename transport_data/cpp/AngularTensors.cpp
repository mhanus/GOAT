#include "AngularTensors.h"
#include <dolfin/common/constants.h>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace dolfin;

inline double sqr(double x) { return x*x; }


AngularTensors::AngularTensors(const OrdinatesData& odata, int L)
{
	int K;
	int D = odata.D;
	int M = odata.M;

	if (D == 3)
		K = sqr(L+1);
	else if (D == 2)
		K = 1./2.*(L+1)*(L+2);

	_Y.reserve(K*M);
	_Q.reserve(K*M);
	_Qt.reserve(K*M*D);
	_G.resize(M*M*D, 0.0);
	_T.resize(M*M*D*D, 0.0);

	_shape_Y.push_back(K);
	_shape_Y.push_back(M);

	_shape_Q.push_back(K);
	_shape_Q.push_back(M);

	_shape_Qt.push_back(K);
	_shape_Qt.push_back(M);
	_shape_Qt.push_back(D);

	_shape_G.insert(_shape_G.end(),2,M);
	_shape_G.push_back(D);

	_shape_T.insert(_shape_T.end(),2,M);
	_shape_T.insert(_shape_T.end(),2,D);

	std::vector<double>::iterator Git = _G.begin();
	std::vector<double>::iterator Tit = _T.begin();

	for (int l = 0; l <= L; l++)
	{
		for (int m = -l; m <= l; m++)
		{
			if (D == 2 && (l+m)%2 != 0)
				continue;

			SphericalHarmonic Ylm(l, m);
			for (int p = 0; p < M; p++)
			{
        double Ylmp = Ylm(odata.xi[p], odata.eta[p], odata.mu[p]);
				_Y.push_back(Ylmp);
				_Q.push_back(Ylmp * odata.pw[p]);

				double omega_p[3] = { odata.xi[p], odata.eta[p], odata.mu[p] };

				for (int i = 0; i < D; i++)
					_Qt.push_back(Ylmp * odata.pw[p] * omega_p[i] );
			}
		}
  }

	for (int p = 0; p < M; p++)
	{
		double omega_p[3] = { odata.xi[p], odata.eta[p], odata.mu[p] };

		for (int q = 0; q < M; q++)
		{
			for (int i = 0; i < D; i++)
			{
				if (p == q)
					*Git = omega_p[i]*odata.pw[p];
				Git++;

				for (int j = 0; j < D; j++)
				{
					if (p == q)
						*Tit = omega_p[i]*omega_p[j]*odata.pw[p];
					Tit++;
				}
			}
		}
	}
}



double SphericalHarmonic::plgndr(double x) const
{
	if (abs(x) > 1)
		dolfin_error("OrdinatesData.cpp",
								"evaluate a Legendre polynomial",
								"Invalid argument: x = %g (req. |x| < 1)", x);

	double pmm = 1;

	if (m > 0)
	{
		double prod = 1;
		for (int i = 1; i <= 2*m; i+=2)
			prod *= i;
		pmm = prod * pow( sqrt((1-x)*(1+x)), m );
	}

	if (l == m)
		return pmm;
	else
	{
		double pmmp1 = (2*m+1)*x*pmm;

		if (l == m+1)
			return pmmp1;
		else
		{
			double pll;
			for (int ll = m+2; ll <= l; ll++)
			{
				pll = ( (2*ll-1)*x*pmmp1-(ll+m-1)*pmm ) / ( ll-m );
				pmm = pmmp1;
				pmmp1 = pll;
			}
			return pll;
		}
	}
}

double SphericalHarmonic::operator()(double xi, double eta, double mu) const
{
	if(abs(mu*mu+eta*eta+xi*xi-1.0) > DOLFIN_EPS)
		dolfin_error("OrdinatesData.cpp",
								"evaluate a spherical harmonic function",
								"Invalid direction cosines: xi=%g, eta=%g, mu=%g (req. xi^2 + eta^2 + mu^2 == 1)", xi, eta, mu);

	if((l == 0) && (m == 0))
		return 1.0;
	else if((l == 1) && (m == -1))
		return eta;
	else if((l == 1) && (m == 0))
		return mu;
	else if((l == 1) && (m == 1))
		return xi;
	else if((l == 2) && (m == -2))
		return sqrt(3.0)*xi*eta;
	else if((l == 2) && (m == -1))
		return sqrt(3.0)*mu*eta;
	else if((l == 2) && (m == 0))
		return 0.5*(3.0*mu*mu-1.0);
	else if((l == 2) && (m == 1))
		return sqrt(3.0)*mu*xi;
	else if((l == 2) && (m == 2))
		return 0.5*sqrt(3.0)*(xi*xi-eta*eta);
	else if((l == 3) && (m == -3))
		return sqrt(5./8.)*eta*(3.0*xi*xi-eta*eta);
	else if((l == 3) && (m == -2))
		return sqrt(15.0)*xi*eta*mu;
	else if((l == 3) && (m == -1))
		return sqrt(3./8.)*eta*(5.0*mu*mu-1.0);
	else if((l == 3) && (m == 0))
		return 0.5*mu*(5.0*mu*mu-3.0);
	else if((l == 3) && (m == 1))
		return sqrt(3./8.)*xi*(5.0*mu*mu-1.0);
	else if((l == 3) && (m == 2))
		return sqrt(15.0/4.0)*mu*(xi*xi-eta*eta);
	else if((l == 3) && (m == 3))
		return sqrt(5./8.)*xi*(xi*xi-3.0*eta*eta);
	else if((l == 4) && (m == -4))
		return 0.5*sqrt(35.)*xi*eta*(xi*xi-eta*eta);
	else if((l == 4) && (m == -3))
		return 0.5*sqrt(0.5*35.)*mu*eta*(3.*xi*xi-eta*eta);
	else if((l == 4) && (m == -2))
		return sqrt(5.)*(21.*mu*mu-3.)*xi*eta/6.;
	else if((l == 4) && (m == -1))
		return 0.5*sqrt(2.5)*mu*eta*(7.*mu*mu-3.);
	else if((l == 4) && (m == 0))
		return (35.*sqr(mu)*sqr(mu)-30.*mu*mu+3.)/8.;
	else if((l == 4) && (m == 1))
		return 0.5*sqrt(2.5)*mu*xi*(7.*mu*mu-3.);
	else if((l == 4) && (m == 2))
		return sqrt(5.)*(21.*mu*mu-3.)*(xi*xi-eta*eta)/12.;
	else if((l == 4) && (m == 3))
		return 0.5*sqrt(0.5*35.)*mu*xi*(xi*xi-3.*eta*eta);
	else if((l == 4) && (m == 4))
		return sqrt(35.)*(sqr(xi)*sqr(xi)-6.*sqr(xi*eta)+sqr(eta)*sqr(eta))/8.;
	else if((l == 5) && (m == -5))
		return 21.*eta*(5.*sqr(xi)*sqr(xi)-10.*sqr(xi*eta)+sqr(eta)*sqr(eta))/(8.*sqrt(14.));
	else if((l == 5) && (m == -4))
		return 0.5*105.*mu*xi*eta*(xi*xi-eta*eta)/sqrt(35.);
	else if((l == 5) && (m == -3))
		return 35.*(9*mu*mu-1.)*eta*(3.*xi*xi-eta*eta)/(8.*sqrt(70.));
	else if((l == 5) && (m == -2))
		return 0.5*sqrt(105.)*mu*(3.*mu*mu-1.)*xi*eta;
	else if((l == 5) && (m == -1))
		return sqrt(15.)*eta*(21.*sqr(mu)*sqr(mu)-14.*mu*mu+1.)/8.;
	else if((l == 5) && (m == 0))
		return mu*(63.*sqr(mu)*sqr(mu)-70.*mu*mu+15.)/8.;
	else if((l == 5) && (m == 1))
		return sqrt(15.)*xi*(21.*sqr(mu)*sqr(mu)-14.*mu*mu+1.)/8.;
	else if((l == 5) && (m == 2))
		return 0.25*sqrt(105.)*mu*(3.*mu*mu-1.)*(xi*xi-eta*eta);
	else if((l == 5) && (m == 3))
		return 35.*(9*mu*mu-1.)*xi*(xi*xi-3.*eta*eta)/(8.*sqrt(70.));
	else if((l == 5) && (m == 4))
		return 105.*mu*(sqr(xi)*sqr(xi)-6.*sqr(xi*eta)+sqr(eta)*sqr(eta))/(8.*sqrt(35.));
	else if((l == 5) && (m == 5))
		return 21.*xi*(sqr(xi)*sqr(xi)-10.*sqr(xi*eta)+5.*sqr(eta)*sqr(eta))/(8.*sqrt(14.));
	else
	{
		double phi = 1./sqrt(1 - sqr(mu)) * acos(xi);
		if (eta < 0)
			phi = 2*M_PI - phi;

		if (m > 0)
			return sqrt(2*factorial(l - m) / factorial(l + m) ) * plgndr(mu) * cos(m * phi);
		else if (m == 0)
			return plgndr(mu);
		else
			return sqrt(2*factorial(l + m) / factorial(l - m) ) * plgndr(mu) * sin(-m * phi);
	}
}
