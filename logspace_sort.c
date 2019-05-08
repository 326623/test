
// This function takes in an array, whose elements are between [0, 2*n]
// it prints out the sorted array in stdout
void logspace_sort(const int* array, int n) {
  int smaller_so_far = -1, smallest = 2 * n;
  int count_same = 0;
  int i, j, k;

  // Each iteration we will store a threshold, smaller_so_far, so that smaller
  // value that is already written will not be considered again. We use
  // count_same to record the number of nonunique elements that are equal to smallest.
  for (i = 0; i < n; i = i + 1) {
    for (j = 0; j < n; ++j) {
      if (smaller_so_far < array[j] && array[j] <= smallest) {
        if (array[j] == smallest)
          ++count_same;
        else
          count_same = 0;
        smallest = array[j];
      }
    }
    for (k = 0; k <= count_same; ++k)
      if (i + k < n - 1)
        printf("%d ", smallest);
      else
        printf("%d", smallest);
    i += count_same;
    smaller_so_far = smallest;
    smallest = 2 * n;
  }
}

void main(int argc, char** argv) {
  int array[10] = {10, 9, 8, 6, 6, 5, 4, 3, 2, 1};
  logspace_sort(array, 10);
}
