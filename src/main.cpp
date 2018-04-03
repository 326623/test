
//#include "operator_overloading.hpp"
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>

double f (double x, void * params) {
  double alpha = *(double *) params;
  double f = log(alpha*x) / sqrt(x);
  return f;
}

int main(void) {
  gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);

  double result, error;
  double expected = -4.0;
  double alpha = 1.0;

  gsl_function F;
  F.function = &f;
  F.params = &alpha;

  gsl_integration_qags (&F, 0, 1, 0, 1e-7, 1000,
                        w, &result, &error);

  printf ("result          = % .18f\n", result);
  printf ("exact result    = % .18f\n", expected);
  printf ("estimated error = % .18f\n", error);
  printf ("actual error    = % .18f\n", result - expected);
  printf ("intervals       = %zu\n", w->size);

  gsl_integration_workspace_free (w);

  int n = 10000;
  gsl_integration_glfixed_table * w1 =
    gsl_integration_glfixed_table_alloc(2*n-1);

  double xi, wi;
  for (int i = 1; i <= n; ++ i) {
    gsl_integration_glfixed_point(-1, 1, i, &xi, &wi, w1);
    printf ("% .10f % .10f\n", xi, wi);
  }

  printf( "% .18f\n", gsl_integration_glfixed(&F, 0, 1, w1));

  gsl_integration_glfixed_table_free (w1);
  return 0;
}
