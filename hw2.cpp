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

// only accept length dividable by 5
template <typename Type>
Type select(Type a[], int size, int index) {
  if (size <= 50) {
    std::sort(a, a + size);
    return a[index];
  }
  Type *p = a;

  for (int i = 0; i < size/5; ++ i) {
    const int j = i * 5;
    std::sort(a + j, a + j + 5);
    std::swap(p[i], a[j + 2]);
  }

  int mid = select(p, size/5, size/10 + (size / 5) % 2);

  int i,j,k;
  i = j = k = 0;
  for (int iter = 0; iter < size; ++ iter) {
    if (a[iter] < mid)
      std::swap(p[i++], a[iter]);
    else if (a[iter] == mid)
      ++j;
    else
      ++k;
  }

  if (index < i)
    return select(p, i, index);
  else if (index < i + j)
    return mid;
  else
    return select(a + i + j, k, index - i - j);
}

TEST(SELECT, NTH) {
  for (int i = 0; i < 20; ++ i) {
    std::vector<int> testSubject1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::vector<int> testSubject2 = {21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};

    testSubject1.insert(testSubject1.end(), testSubject2.begin(), testSubject2.end());
    testSubject2 = testSubject1;
    const int nth = i;
    auto result = select(testSubject1.data(), testSubject1.size(), nth);

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
    auto result = select(testSubject1.data(), testSubject1.size(), nth);

    std::sort(testSubject2.begin(), testSubject2.end());
    EXPECT_EQ(result, testSubject2[nth]);
    // std::cout << testSubject1 << '\n'
    //           << testSubject2 << '\n';
  }
}


TEST(SELECT, RANDOM_NTH) {
  std::vector<int> testSubject1;
  std::vector<int> testSubject2;
  const int sep = 5000;
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
  auto result = select(testSubject1.data(), testSubject1.size(), nth);

  std::sort(testSubject2.begin(), testSubject2.end());
  EXPECT_EQ(result, testSubject2[nth]);
}

template <typename Type>
void mid(Type A[], int i, Type p[]) {
  int j = i * 5;
  std::sort(A + j, A + j + 5);
  p[i] = A[j+2];
}

// One major fallback of this code is its dynamical allocation, without delete
template <typename Type>
Type Select(Type A[], int n, int k) {
  int i, j, s, t;
  Type m, *p, *q, *r;
  if (n <= 38) {
    std::sort(A, A + n);
    return A[k];
  }

  p = new Type[3*n/4];
  q = new Type[3*n/4];
  r = new Type[3*n/4];
  for (i=0;i<n/5;++i) {
    mid(A, i, p);
  }
  // larger one
  m = Select(p, i, i/2+i%2);
  // partition
  i = j = s = 0;
  for (t=0;t<n;++t) {
    if (A[t] < m)
      p[i++] = A[t];
    else if (A[t]==m)
      q[j++] = A[t];
    else
      r[s++] = A[t];
  }
  if (i>k)
    return Select(p,i,k);
  else if (i+j>k)
    return m;
  else
    return Select(r,s,k-i-j);
}

TEST(NEW_SELECT, NTH) {
  for (int i = 0; i < 20; ++ i) {
    std::vector<int> testSubject1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::vector<int> testSubject2 = {21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};

    testSubject1.insert(testSubject1.end(), testSubject2.begin(), testSubject2.end());
    testSubject2 = testSubject1;
    const int nth = i;
    auto result = Select(testSubject1.data(), testSubject1.size(), nth);

    std::sort(testSubject2.begin(), testSubject2.end());
    EXPECT_EQ(result, testSubject2[nth]);
    // std::cout << testSubject1 << '\n'
    //           << testSubject2 << '\n';
  }
}

TEST(NEW_SELECT, RANDOM_NTH) {
  std::vector<int> testSubject1;
  std::vector<int> testSubject2;
  const int sep = 50000;
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
  auto result = Select(testSubject1.data(), testSubject1.size(), nth);

  std::sort(testSubject2.begin(), testSubject2.end());
  EXPECT_EQ(result, testSubject2[nth]);
}
