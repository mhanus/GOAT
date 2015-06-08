#ifndef __ANGULAR_TENSORS_H
#define __ANGULAR_TENSORS_H

#include <vector>
#include <string>
#include <iostream>
#include <dolfin/log/log.h>
#include "OrdinatesData.h"

namespace dolfin
{
  class AngularTensors
  {
  public:
    AngularTensors(const dolfin::OrdinatesData& odata, int L);

    const std::vector<double>& Q() const { return _Q; }
    const std::vector<double>& Qt() const { return _Qt; }
    const std::vector<double>& G() const { return _G; }
    const std::vector<double>& T() const { return _T; }

    const std::vector<int>& shape_Q() const { return _shape_Q; }
    const std::vector<int>& shape_Qt() const { return _shape_Qt; }
    const std::vector<int>& shape_G() const { return _shape_G; }
    const std::vector<int>& shape_T() const { return _shape_T; }

  private:
    std::vector<double> _Q, _Qt, _G, _T;
    std::vector<int> _shape_Q, _shape_Qt, _shape_G, _shape_T;
  };

  class SphericalHarmonic
  {
    unsigned int l;
    int m;

    double plgndr(double x) const;

    int factorial(unsigned int n) const
    {
      int fact=1;
      for (int i=1; i<=n; i++)
        fact*=i;
      return fact;
    }

    public:
      SphericalHarmonic(unsigned int l, int m) : l(l), m(m)
      {
        if (std::abs(m) > l)
          dolfin_error("OrdinatesData.h",
                       "create an object for representation of spherical harmonics",
                       "Incompatible spherical harmonic degree (l = %g) and order (m = %g): |m| < l required");
      }

      double operator() (double xi, double eta, double mu) const;
  };
}
#endif
