#ifndef __ORDINATES_DATA_H
#define __ORDINATES_DATA_H

#include <vector>
#include <string>
#include <iostream>
#include <dolfin/log/log.h>

namespace dolfin 
{
  class OrdinatesData
  {
    friend class AngularTensors;

  public:
    OrdinatesData(unsigned int N, unsigned int D, const std::string& filename);

    void write_pw(const std::string& filename);
    friend std::ostream & operator<< (std::ostream& os, const OrdinatesData& odata);

    const std::vector<double>& get_xi() const { return xi; }
    const std::vector<double>& get_eta() const { return eta; }
    const std::vector<double>& get_mu() const { return mu; }
    const std::vector<double>& get_pw() const { return pw; }
    
    const std::vector<int>& get_reflections_about_x() const { return reflections_about_x; }
    const std::vector<int>& get_reflections_about_y() const { return reflections_about_y; }
    const std::vector<int>& get_reflections_about_z() const { return reflections_about_z; }
    
    int get_N() const { return N; }
    int get_M() const { return M; }
    int get_D() const { return D; }
    
    std::vector<double> get_ordinate(int n) const;

    void print_info() const;

  private:
    unsigned int N, M, D;
    std::vector<double> xi, eta, mu, pw;
    
    std::vector<int> reflections_about_x;
    std::vector<int> reflections_about_y;
    std::vector<int> reflections_about_z;

  };
}

#endif
