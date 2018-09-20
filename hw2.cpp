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
 * Returns the ForwardIterator pointing to the element that is equal or the
 * first element that is larger than pivot
 * (the array is unsorted, larger and the first to encounter)
 */
template <typename Type, typename ForwardIt>
ForwardIt partition(ForwardIt a, ForwardIt end,
                    const Type &pivot)
{
  ForwardIt space = a; // the seperation pointer, left all < pivot, right >=

  while (a != end) {
    if (*a < pivot) {
      std::swap(*a, *space);
      ++space;
    }
    ++a;
  }
  return space;
}


TEST(testPartition, testOnVector) {
  std::vector<int> testSubject = {4, 3, 5, 7, 1, 32, 29, 0, 3};
  auto seperation = ::partition(testSubject.begin(),
                              testSubject.end(), 32);
  auto iter = testSubject.begin();
  while (iter != seperation) {
    EXPECT_TRUE(*iter < *seperation);
    ++iter;
  }

  EXPECT_TRUE(32 == *seperation);

  while(iter != testSubject.end()) {
    EXPECT_TRUE(*iter > *seperation);
    ++iter;
  }
}


// int select(int a[], int n, int k, int groupSize) {
//   int i = 0;
//   int numGroups = (n + groupSize) / groupSize;
//   const int middle = groupSize / 2;

//   // all treat the last one differently
//   for (i = 0; i + groupSize < n; i += groupSize)
//     std::sort(a + i, a + i + groupSize);
//   const int lastMiddle = (n - i) / 2;
//   std::sort(a + i, a + n);

//   for (i = 0; i < numGroups-1; ++ i) {
//     std::swap(a[i], a[i * groupSize + middle]);
//   }
//   //const int middle = (i * groupSize + n) / 2;
//   std::swap(a[i], a[i * groupSize + lastMiddle]);

//   const int beforeMiddle = (numGroups-1) * middle + lastMiddle;

//   const int pivot = select(a, numGroups, numGroups/2, groupSize);
//   partition(a, pivot, )
//   // cast after
//   if (k < beforeMiddle) {

//   }
// }
