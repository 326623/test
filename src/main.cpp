// #include <boost/numeric/ublas/vector.hpp>
// #include <boost/numeric/ublas/matrix.hpp>
// #include <boost/numeric/ublas/io.hpp>

// int main() {
//   // It seems that namespace are in seperate scope, which makes sense
//   {
//     using namespace std;
//     vector<double> v(3,1);
//     for (int i = 0; i < v.size(); ++ i) {
//       cout << v[i] << endl;
//     }
//   }

//   {
//     using namespace boost::numeric::ublas;
//     //vector<double> v (3);
//     for (unsigned i = 0; i < 3; ++ i) {
//       unit_vector<double> v (3, i);
//       std::cout << v << std::endl;
//       //v (i) = i;
//     }
//   }

//   {
//     using namespace boost::numeric::ublas;
//     zero_vector<double> v (3);
//     std::cout << v << std::endl;
//   }

//   {
//     using namespace boost::numeric::ublas;
//     scalar_vector<double> v (3);
//     std::cout << v << std::endl;
//   }

//   {
//     using namespace boost::numeric::ublas;
//     matrix<double> m (3, 3);
//     //matrix<double> m (3, 3);
//     for (unsigned i = 0; i < m.size1 (); ++ i)
//       for (unsigned j = 0; j < m.size2 (); ++ j)
//         m (i, j) = 3 * i + j;
//     std::cout << m << std::endl;
//   }

// }

#include "operator_overloading.hpp"

int main() {
  using namespace op_overload;
  //std::cout << "hello there" << std::endl;
  hello_you a(1);
  //std::cout << std::is_base_of<hello_you, hello_you>::value << std::endl;

  //std::cout << a << std::endl;

  operator<< (std::cout, a);// << std::endl;

  std::cout << a << std::endl;
}
