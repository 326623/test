// Pointer Prefix
// The pointer prefix for 256 memory operands is
// ymmword ptr
// vdivps ymm0, ymm1, ymmword ptr [rcx]
// 16 registers in AVX, named YMM0 to YMM15
// Each is 256 bits wide and each is aliased to the 16 SSE registers
// Different size elements
// The AVX instructions do not deal with Integers

// 8 singles, 4 doubles
// New broadcasting <= filled register with certain value

#include <iostream>
#include <immintrin.h>

std::ostream & operator<<(std::ostream &out, const __m128 &vector) {
  out << '['
      << vector[0] << ", " << vector[1] << ", "
      << vector[2] << ", " << vector[3] <<
    ']';
  return out;
}

int main() {
  std::ios_base::sync_with_stdio(false);

  __m128 vector = _mm_set_ps(1.0, 1.0, 1.0, 1.0);
  __m128 vector2 = _mm_set_ps1(1.0);

  vector = _mm_add_ps(vector, vector2);

  std::cout << vector << '\n'
            << vector2 << '\n';

  return 0;
}

// extern "C" bool GetAVXSupportFlag();

// int main() {
//   if (GetAVXSupportFlag())
//     std::cout << "You've AVX yay!!!" << std::endl;
//   else
//     std::cout << "You've no AVX buddy...:{" << std::endl;

//   return 0;
// }
