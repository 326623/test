
// #include <opencv2/core.hpp>
// #include <iostream>
// #include <string>
// #include <vector>
// using std::vector;
// using std::string;
// using std::cout;
// using std::endl;
// using std::cerr;
// using std::ostream;
// using namespace cv;



// // struct data {
// //   float a,b,c;
// // };

// int main(int argc, char** argv)
// {


//   try {
// //    FileStorage fs("svm_1_split_0.json", 0); // use postfix to know it's json, 0 for read
//     FileStorage fs("svm_1.json", 0);
//     FileNode root = fs[(string)"svm" + (string)"_1"];
//     cout << root.size() << std::endl;
//     cout << fs.getFormat() << std::endl;
//   }
//   catch(const Exception& e) {
//     cout << "error: " << e.what() << std::endl;
//   }
// }

// #include <iostream>

// using std::cout;

// int main(int argc, char** argv) {
//   auto glambda = [](int a, double&& b) { return a < b; }; // type unknow
//   cout << glambda(3, 3.14);

//   return 0;
// }
// #include <boost/numeric/ublas/matrix.hpp>
// #include <boost/numeric/ublas/io.hpp>

// int main () {
//   using namespace boost::numeric::ublas;
//   matrix<double> m (3, 3);
//   for (unsigned i = 0; i < m.size1 (); ++ i)
//     for (unsigned j = 0; j < m.size2 (); ++ j)
//       m (i, j) = 3 * i + j;
//   std::cout << m << std::endl;
// }
// #include <boost/numeric/ublas/vector.hpp>
// #include <boost/numeric/ublas/io.hpp>

// int main () {
//   using namespace boost::numeric::ublas;
//   vector<double> v (3);
//   for (unsigned i = 0; i < v.size (); ++ i)
//     v (i) = i;
//   std::cout << v << std::endl;
// }

#include <gsl/gsl_integration.h>
#include <iostream>
#include <cmath>

double f(double x, void * params) {
  return std::sin(x);
}

int main() {
  gsl_integration_glfixed_table * integral_table =
    gsl_integration_glfixed_table_alloc(10);

  gsl_function F;
  F.function = f;

  std::cout
    << gsl_integration_glfixed(&F, 0, 1, integral_table) << std::endl;
}
