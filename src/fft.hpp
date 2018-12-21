#ifndef _NEWJOY_FFT_HPP_
#define _NEWJOY_FFT_HPP_

#include <complex>
#include <cmath>
#include <vector>

void seperate(std::complex<double> *X, int N) {
  std::vector<std::complex<double>> b(N);
  for (int i = 0; i < N / 2; ++ i)
    b[i] = X[2 * i];
  for (int i = 0; i < N / 2; ++ i)
    b[i + N / 2] = X[2 * i + 1];
  for (int i = 0; i < N; ++ i)
    X[i] = b[i];
}

void ditfft2(std::complex<double> *X, int N) {
  if (N >= 2) {
    seperate(X, N);
    ditfft2(X, N/2); ditfft2(X+N/2, N/2);
    for (int k = 0; k < N/2; ++ k) {
      const auto even = X[k];
      const auto odd_twiddle = std::exp(std::complex<double>(0, 2 * M_PI * k / N)) * X[k+N/2];
      X[k] = even + odd_twiddle;
      X[k+N/2] = even - odd_twiddle;
    }
  }
}

#endif /* _NEWJOY_FFT_HPP_ */
