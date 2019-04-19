// the cpu variant
void inclusive_scan(float* output, float* input, int length) {
  if (length <= 0) return;
  output[0] = input[0];
  for (int i = 1; i < length; ++i) {
    output[i] = output[i - 1] + input[i];
  }
}

void exclusive_scan(float* output, float* input, int length) {
  if (length <= 0) return;
  output[0] = 0;
  for (int i = 1; i < length; ++i) {
    output[i] = output[i - 1] + input[i - 1];
  }
}