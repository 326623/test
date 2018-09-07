#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
//#include <cstdlib>
// insertion sort
template <typename Iterator>
void insertionSort(Iterator first, Iterator last) {

  for (Iterator ith = first; ith != last; ++ ith) {

    Iterator back = ith;
    // pick the ith element and insert back
    while (back != first && *(back) < *(back-1)) {
      std::swap(*(back), *(back-1));
      back--;
    }
  }
}

template <typename ForwardIterator>
void display(ForwardIterator first, ForwardIterator last) {
  for (; first != last; ++ first)
    std::cout << *first << ' ';
  std::cout << std::endl;
}

// check if is sorted
template <typename ForwardIterator>
bool check(ForwardIterator first, ForwardIterator last) {
  // somewhat less efficient here
  for (; first+1 != last; ++ first)
    if (*first > *(first+1)) return false;
  return true;
}

int main() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, 10000);
  std::vector<int> test_vec(1000);
  for (auto & item : test_vec)
    item = dis(gen);

  //  display(test_vec.begin(), test_vec.end());
  std::cout << check(test_vec.begin(), test_vec.end()) << std::endl;

  insertionSort(test_vec.begin(), test_vec.end());

  // display(test_vec.begin(), test_vec.end());
  std::cout << check(test_vec.begin(), test_vec.end()) << std::endl;
}
