uint8_t _mm256_hmax_index(const __m256i v) {
  __m256i vmax = v;
  vmax = _mm256_max_epu32(vmax, _mm256_alignr_epi8(vmax, vmax, 4));
  vmax
}
