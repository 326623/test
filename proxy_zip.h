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
#ifndef _NEWJOY_PROXY_ZIP_H__
#define _NEWJOY_PROXY_ZIP_H_
// ProxyZip is a class that can wrap up two or more containers with
// the same size and same linear accessing pattern into a single container.
// It's useful when one wants to pass in a STL algorithm like std::sort
// and be able to sort the them as once without making them as a
// std::vector<std::tuple> or std::vector<std::pair>, which can hit performance
// or simply making it very hard when one is trying to access one of
// them independently like performing SIMD operations on one of them

// Its possible use case is as follows:
// std::vector a {1, 2, 3, ..., n};
// std::vector b {n, ..., 3, 2, 1};
// ProxyZip<std::vector, std::vector> zip(a, b);
// std::sort(zip.begin(), zip.end());
// std::vector c(a.size());
// for (int i = 0; i < a.size(); ++ i) {
//   c[i] = 2 * a[i];
// }

template <typename T, typename ... Ts>
class ProxyZip {
 private:

 public:
  // Do we need this
  ProxyZip() {}

  template<typename T, typename ... Ts>
   ProxyZip(T a)
};

#endif /* _NEWJOY_PROXY_ZIP_HPP_ */
