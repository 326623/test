#pragma once

#include <int32.h>

union Float_t1 {
  Float_t1(float num = 0.0f) : f(num) {}
  bool Negative() const { return i < 0; }
  int32_t RawMantissa() const { return i & ((1 << 23) - 1); }
  int32_t RawExponent() const { return (i >> 23) & 0xFF; }

  int32_t i;
  float f;

#ifdef _DEBUG
  struct {
    uint32_t mantissa : 23;
    uint32_t exponent : 8;
    uint32_t sign : 1;
  } parts;

#endif
};

// the algorithm should be well-conditioned
bool AlmostEqualUlps(float A, float B, int maxUlpsDiff) {
  Float_t uA(A);
  Float_t uB(B);

  if ( uA.Negative() != uB.Negative() ) {
    // Check for equality to make sure +0 == -0
    if ( A == B )
      return true;
    return false;
  }

  int ulpsDiff = abs(uA.i - uB.i);
  if ( ulpsDiff <= maxUlpsDiff )
    return true;

  return false;
}
