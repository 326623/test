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
#ifndef _TEST_CUDA_MACRO_H_
#define _TEST_CUDA_MACRO_H_
#include <glog/logging.h>

#define CUDA_CALL(x)                            \
  do {                                          \
    CHECK_EQ((x), cudaSuccess);                 \
  } while (0)

#define CURAND_CALL(x)                          \
  do {                                          \
    CHECK_EQ((x), CURAND_STATUS_SUCCESS);       \
  } while (0)

// #define CUDA_CALL(x)                                  \
//   do {                                                \
//     if ((x) != cudaSuccess) {                         \
//       printf("Error at %s:%d\n", __FILE__, __LINE__); \
//       return EXIT_FAILURE;                            \
//     }                                                 \
//   } while (0)

// #define CUDA_CALL(x)                                              \
//   do {                                                            \
//     CHECK_EQ((x), cudaSuccess)                                    \
//         << absl::StrFormat("Error at %s:%d", __FILE__, __LINE__); \
//   } while (0)

// #define CURAND_CALL(x)                                              \
//   do {                                                              \
//     CHECK_EQ((x), CURAND_STATUS_SUCCESS)                            \
//         << absl::StrFormat("Error at %s:%d\n", __FILE__, __LINE__); \
//   } while (0)

// #define CURAND_CALL(x)                                \
//   do {                                                \
//     if ((x) != CURAND_STATUS_SUCCESS) {               \
//       printf("Error at %s:%d\n", __FILE__, __LINE__); \
//       return EXIT_FAILURE;                            \
//     }                                                 \
//   } while (0)
#endif /* _TEST_CUDA_MACRO_H_ */
