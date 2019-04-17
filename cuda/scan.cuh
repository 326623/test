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

// template <typename Iterator>
// __host__ __device__
// void inclusive_scan(Iterator first, Iterator last, Iterator::value_type initial,
//                     BinaryOp op) {
  
// }

// __device__
__global__
void inclusive_prefix_sum(float* first, float* last) {
  int num_elements = last - first;
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  // may need to check overflow
  int maxJ = static_cast<int>(ceil(log(static_cast<float>(num_elements))));
  for (int j = 0; j < maxJ; ++ j) {
    for (int i = id; i < num_elements; i += blockDim.x * gridDim.x) {
      if (i >= (1 << (j-1))) {
        first[i] = first[i - (1 << (j-1))] + first[i];
      }
    }
  }
}
