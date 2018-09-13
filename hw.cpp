
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

template <typename Integral>
Integral totalNumDecomp(Integral n, Integral m) {
  if (m == 1) return 1;
  if (m > n) return totalNumDecomp(n, n);
  if (m == n) return totalNumDecomp(n, n-1) + 1;
  return totalNumDecomp(n - m, m) + totalNumDecomp(n, m - 1);
}

template <typename Integral>
Integral totalNumDecomp2(Integral n, Integral m) {
  if (m == 1) return 1;
  if (m == n) return totalNumDecomp2(n, n-1) + 1;
  return totalNumDecomp2(n - m, std::min(n - m, m)) + totalNumDecomp2(n, m - 1);
}

// for std::vector<int> output
template <typename Integral>
std::ostream &operator<<(std::ostream &os,
                         const std::vector<Integral> &numbers)
{
  for (const auto & number : numbers)
    os << number << ' ';
  return os;
}

/*
 * sometime if Integral is unsigned, don't compare it with 0, please
 * the short and clear version, carrying the vector around
 */
template <typename Integral, typename Alloc,
          template <typename, typename> class Vector>
void printDecomp(Integral n, Integral m,
                 Vector<Integral, Alloc> decomNum)
{
  // end of recursion, output all
  if (m == 1) {
    std::cout << decomNum;
    for (Integral i = 0; i < n; ++ i)
      std::cout << 1 << ' ';
    std::cout << '\n';
  }
  else if (m == n) {
    // another end
    std::cout << decomNum
              << n << '\n';
    // call next case, last call, use move
    printDecomp(n, n-1, std::move(decomNum));
  }
  else {
    decomNum.emplace_back(m);
    printDecomp(n - m, std::min(n - m, m), decomNum);
    decomNum.pop_back();
    printDecomp(n, m - 1, std::move(decomNum));
  }
}

TEST(testTotalNumDecomp, testOutput) {
  EXPECT_EQ(totalNumDecomp(1, 1), 1);
  EXPECT_EQ(totalNumDecomp(2, 2), 2);
  EXPECT_EQ(totalNumDecomp(3, 3), 3);
  EXPECT_EQ(totalNumDecomp(4, 4), 5);
  EXPECT_EQ(totalNumDecomp(5, 5), 7);
  EXPECT_EQ(totalNumDecomp(6, 6), 11);
}

TEST(testTotalNumDecomp, testCompare) {
  for (int i = 1; i < 50; ++ i) {
    EXPECT_EQ(totalNumDecomp(i, i), totalNumDecomp2(i, i));
  }
}

TEST(testPrintDecomp, testingOutput) {
  //std::ios_base::sync_with_stdio(true);
  for (int i = 1; i < 50; ++ i) {
    std::cout << "iteration " << i << " start" << '\n';
    printDecomp(i, i, std::vector<int>());
    std::cout << "iteration " << i << " end" << '\n';
  }

}
