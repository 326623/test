
// //#include "operator_overloading.hpp"
// #include <stdio.h>
// #include <math.h>
// #include <gsl/gsl_integration.h>

// double f (double x, void * params) {
//   double alpha = *(double *) params;
//   double f = log(alpha*x) / sqrt(x);
//   return f;
// }

// int main(void) {
//   gsl_integration_workspace * w
//     = gsl_integration_workspace_alloc (1000);

//   double result, error;
//   double expected = -4.0;
//   double alpha = 1.0;

//   gsl_function F;
//   F.function = &f;
//   F.params = &alpha;

//   gsl_integration_qags (&F, 0, 1, 0, 1e-7, 1000,
//                         w, &result, &error);

//   printf ("result          = % .18f\n", result);
//   printf ("exact result    = % .18f\n", expected);
//   printf ("estimated error = % .18f\n", error);
//   printf ("actual error    = % .18f\n", result - expected);
//   printf ("intervals       = %zu\n", w->size);

//   gsl_integration_workspace_free (w);

//   int n = 10000;
//   gsl_integration_glfixed_table * w1 =
//     gsl_integration_glfixed_table_alloc(2*n-1);

//   double xi, wi;
//   for (int i = 1; i <= n; ++ i) {
//     gsl_integration_glfixed_point(-1, 1, i, &xi, &wi, w1);
//     printf ("% .10f % .10f\n", xi, wi);
//   }

//   printf( "% .18f\n", gsl_integration_glfixed(&F, 0, 1, w1));

//   gsl_integration_glfixed_table_free (w1);
//   return 0;
// }

#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/range/irange.hpp>
#include <chrono>
#include <vector>

int main() {
  using namespace boost::numeric::ublas;
  // would use less time if we know the size of non zero element before hand
  mapped_matrix<double> m (3, 3);
  std::cout << m << std::endl;
  auto pre = std::chrono::high_resolution_clock::now();

  for ( unsigned int i = 0; i < m.size1(); ++ i ) {
    for ( unsigned int j = 0; j < m.size2(); ++ j ) {
      m (i, j) += i + j;
    }
  }

  std::cout << std::chrono::duration_cast<std::chrono::duration<double>>
    (std::chrono::high_resolution_clock::now() - pre).count() << std::endl;


  std::cout << m << std::endl;
}

template<typename Matrix, typename Vector, typename number>
bool Gauss_Seidel_Method(const Matrix & A,
                         // An initial x should be provided,
                         // no default value
                         // please don't pass in the b==x, it is not allowed!!!
                         const Vector & b, Vector & x,
                         number precision, size_t MAX_ITER = 1e2) {
  // This is an iterative process

  // This variable is used to determine if the process has converged
  double norm = precision + 1;
  size_t count = 0;
  while ( norm > precision && count++ < MAX_ITER) {
    norm = 0;
    for ( size_t i = 0; i < A.size1(); ++ i ) {

      double next_x_i = b(i);

      for ( size_t j = 0; j < i; ++ j )
        next_x_i -= A(i, j) * x(j);

      for ( size_t j = i+1; j < A.size2(); ++ j )
        next_x_i -= A(i, j) * x(j);

      next_x_i /= A(i, i);

      norm += fabs(next_x_i - x(i));
      x(i) = next_x_i;
      std::cout << count << std::endl;
    }
  }

  return true;
}

// int main () {
//   using namespace boost::numeric::ublas;
//   constexpr double tmp[3][3] = { {2, -1, 0},
//                                  {-1, 2, -1},
//                                  {0, -1, 1} };
//   constexpr double x_tmp[3] = {1, 2, 3};
//   //  std::vector<std::vector<double>>

//   matrix<double> A(3, 3);

//   for ( auto i : boost::irange(0, 3) ) 
//     for ( auto j : boost::irange(0, 3) )
//       A (i, j) = tmp[i][j];

//   //std::cout << banded_adaptor<matrix<double>>(A, 0, 0) << std::endl;
//   //std::cout << triangular_adaptor<matrix<double>, lower>(A) << std::endl;

//   vector<double> b(3);
//   vector<double> x(3);

//   for ( auto i : boost::irange(0, 3) )
//     b (i) = x_tmp[i];

//   x = b;

//   Gauss_Seidel_Method(A, b, x, 1e-3);

//   std::cout << A << std::endl;
//   std::cout << b << std::endl;
//   std::cout << x << std::endl;

// }
