/* Copyright (C) 2018 New Joy - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3
 *
 *
 * You should have received a copy of the GPLv3 license with
 * this file. If not, please visit https://www.gnu.org/licenses/gpl-3.0.en.html
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: yangqp5@outlook.com (New Joy)
 *
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <ios>

/*
 * Returns the ForwardIterator pointing to the element that is equal or
 * is the first element that is larger than pivot
 * (the array is unsorted, larger and the first to encounter)
 */
template <typename Type, typename ForwardIt>
ForwardIt partition(ForwardIt a, ForwardIt end,
                    const Type &pivot)
{
  ForwardIt space = a; // the seperation pointer, left all < pivot, right >=

  while (a != end) {
    if (*a < pivot) {
      std::iter_swap(a, space);
      ++space;
    }
    ++a;
  }
  return space;
}

// int partition(int *array, int n, int k) {
//   const int pivot = array[k];
//   std::iter_swap(array, array + k);
//   int space = 1;

//   for (int i = 1; i < n; ++ i) {
//     if (array[i] < pivot) {
//       std::swap(array[i], array[space]);
//       ++space;
//     }
//   }
//   std::iter_swap(array, array + space - 1);
//   return space;
// }

int partition(int *array, int n, int pivot) {
  int space = 0;
  for (int i = 0; i < n; ++ i) {
    if (array[i] < pivot) {
      std::swap(array[i], array[space]);
      ++space;
    }
  }
  std::iter_swap(array, array + space - 1);
  return space;
}


template <typename Type>
std::ostream & operator<<(std::ostream & out, std::vector<Type> vs) {
  for (const auto & v : vs)
    out << v << ' ';
  return out;
}

TEST(testPartition, testOnVector) {
  std::vector<int> testSubject = {4, 3, 5, 7, 1, 32, 29, 0, 3};
  const int pivot = 1;
  auto seperation = ::partition(testSubject.begin(),
                                testSubject.end(), pivot);
  std::cout << testSubject << '\n';
  auto iter = testSubject.begin();
  std::cout << *seperation << '\n';
  while (iter != seperation) {
    EXPECT_TRUE(*iter <= *seperation);
    ++iter;
  }

  // EXPECT_TRUE(pivot == *iter);
  // ++iter;

  while(iter != testSubject.end()) {
    EXPECT_TRUE(*iter >= pivot);
    ++iter;
  }
}

TEST(testPartition, testInPlace) {
  std::vector<int> testSubject = {4, 3, 5, 7, 1, 32, 29, 0, 3};
  const int pivotPos = 1;
  auto sep = ::partition(testSubject.data(),
                         testSubject.size(), pivotPos);
  int pivot = testSubject[sep-1];
  std::cout << testSubject << '\n';
  std::cout << testSubject[sep] << '\n';

  for (int i = 0; i < sep-1; ++ i) {
    EXPECT_TRUE(testSubject[i] < pivot);
  }

  for (int i = sep; i < static_cast<int>(testSubject.size()); ++ i) {
    EXPECT_TRUE(testSubject[i] >= pivot);
  }
}

// left closed, right open <==> [)
int select(int array[], int left, int right, int k, int groupSize) {
  if (right - left < 15) {
    std::sort(array + left, array + right);
    return array[k];
  }
  else {
    int i = 0;
    int *a = array + left;
    const int n = right - left;
    int numGroups = (n + groupSize - 1) / groupSize;
    const int middle = groupSize / 2;

    // all treat the last one differently
    for (i = 0; i + groupSize < n; i += groupSize)
      std::sort(a + i, a + i + groupSize);
    const int lastMiddle = (n - i) / 2;
    std::sort(a + i, a + n);

    for (i = 0; i < numGroups-1; ++ i) {
      std::swap(a[i], a[i * groupSize + middle]);
    }
    std::swap(a[i], a[i * groupSize + lastMiddle]);

    const int pivot = select(array, left, left + numGroups, numGroups/2, groupSize);
    // implement a partition such to return the position where the pivot is
    // and left side smaller, right side larger
    auto dist = ::partition(a, right-left, pivot) + left;

    if (k < dist)
      return select(array, left, dist, k, groupSize);
    else
      return select(array, dist, right, k, groupSize);
  }
}

TEST(SELECT, NTH) {
  for (int i = 0; i < 20; ++ i) {
    std::vector<int> testSubject1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::vector<int> testSubject2 = {21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};

    testSubject1.insert(testSubject1.end(), testSubject2.begin(), testSubject2.end());
    testSubject2 = testSubject1;
    const int nth = i;
    auto result = select(testSubject1.data(), 0, testSubject1.size(), nth, 5);

    std::sort(testSubject2.begin(), testSubject2.end());
    EXPECT_EQ(result, testSubject2[nth]);
    // std::cout << testSubject1 << '\n'
    //           << testSubject2 << '\n';
  }
}

TEST(SELECT, NTH_SAME_ELEMENT) {
  for (int i = 0; i < 10; ++ i) {
    std::vector<int> testSubject1 = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    std::vector<int> testSubject2 = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5};

    testSubject1.insert(testSubject1.end(), testSubject2.begin(), testSubject2.end());
    testSubject2 = testSubject1;
    const int nth = i;
    auto result = select(testSubject1.data(), 0, testSubject1.size(), nth, 5);

    std::sort(testSubject2.begin(), testSubject2.end());
    EXPECT_EQ(result, testSubject2[nth]);
    // std::cout << testSubject1 << '\n'
    //           << testSubject2 << '\n';
  }
}


TEST(SELECT, RANDOM_NTH) {
  std::vector<int> testSubject1;
  std::vector<int> testSubject2;
  const int sep = 500;
  std::random_device rd;

  for (int n = 0; n < sep; ++ n) {
    std::uniform_int_distribution<int> distrib1(sep * n, sep * 2 * n);
    std::uniform_int_distribution<int> distrib2(sep * n * 3, sep * n * 4);
    testSubject1.emplace_back(distrib1(rd));
    testSubject2.emplace_back(distrib2(rd));
  }
  testSubject1.insert(testSubject1.end(), testSubject2.begin(), testSubject2.end());
  testSubject2 = testSubject1;

  const int nth = 0;
  auto result = select(testSubject1.data(), 0, testSubject1.size(), nth, 5);

  std::sort(testSubject2.begin(), testSubject2.end());
  EXPECT_EQ(result, testSubject2[nth]);
}
