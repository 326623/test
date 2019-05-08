#include <stdio.h>

// This function takes in an array, whose elements are between [0, 2*n]
// it prints out the sorted array in stdout
void logspace_sort(const int* array, int n) {
  int smaller_so_far = -1, smallest = 2 * n + 1;
  int count_same = 0;
  int i, j, k;

  // Each iteration we will store a threshold, smaller_so_far, so that smaller
  // value that is already written out will not be considered again. We use
  // count_same to record the number of nonunique elements that are equal to
  // smallest. This works because, suppose we find the first smallest(by index)
  // that is larger then the previous smallest(smaller_so_far) and if there it
  // is nonunique in the array, then no element will be less than smallest. Then
  // we can safely count the number of elements that are equal to smallest with
  // count_same, and write to output stream correspondingly
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

    // Output the same number of times the value occur in the array
    for (k = 0; k <= count_same; ++k)
      if (i + k < n - 1)
        printf("%d ", smallest);
      else
        printf("%d\n", smallest);

    i += count_same;
    smaller_so_far = smallest;
    smallest = 2 * n + 1;
  }
}

int main(int argc, char** argv) {
  int array0[10] = {10, 9, 8, 6, 6, 5, 4, 3, 2, 1};
  logspace_sort(array0, 10);
  int array1[10] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  logspace_sort(array1, 10);
  int array3[10] = {20, 18, 16, 14, 12, 10, 8, 6, 4, 2};
  logspace_sort(array3, 10);
  int array4[10] = {20, 18, 16, 12, 12, 10, 8, 6, 4, 2};
  logspace_sort(array4, 10);
  return 0;
}
