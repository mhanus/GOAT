#include <iostream>
#include <algorithm>
#include <iterator>

#include "OrdinatesData.h"
#include "AngularTensors.h"
using namespace dolfin;

int main(int argc, char **argv) 
{
  OrdinatesData odata(2, 3, "../lgvalues.txt");
  odata.write_pw("../pweights.txt");
  std::cout << odata << std::endl;

  AngularTensors atens(odata, 0);

  std::vector<int> v;
  std::ostream_iterator<int> out(std::cout, " ");

  v = atens.shape_Q();
  std::copy(v.begin(), v.end(), out);
  std::cout << std::endl;

  v = atens.shape_Qt();
  std::copy(v.begin(), v.end(), out);
  std::cout << std::endl;

  v = atens.shape_G();
  std::copy(v.begin(), v.end(), out);
  std::cout << std::endl;

  v = atens.shape_T();
  std::copy(v.begin(), v.end(), out);
  std::cout << std::endl;

  return 0;
}
