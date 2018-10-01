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

#ifndef _NEWJOY_VITERBI_HPP_
#define _NEWJOY_VITERBI_HPP_
#include <vector>
#include <string>
#include <map>
#include <stack>

#define MAX_STRING_SIZE 400

// given a sentence, return the segmented words
template <typename Float, typename Integral = int>
std::vector<std::string>
viterbiSegment(const std::string &sentence,
                const std::map<std::string, Float> &probDict,
                Integral max_word_length = 20)
{
  std::vector<Float> prevBest;
  std::vector<Integral> lastSegment;
  prevBest.reserve(400);
  lastSegment.reserve(400);

  prevBest.emplace_back(1.0);
  lastSegment.emplace_back(0);
  Integral sentenceLen = static_cast<int>(sentence.size());

  for (Integral i = 1; i < sentenceLen + 1; ++ i) {

    Float maxP = 0;
    Integral argMax = -1;
    for (Integral j = std::max(0, i - max_word_length); j < i; ++ j) {
      Float wordP = 0;
      auto probIter = probDict.find(sentence.substr(j, i - j));

      if (probIter == probDict.end())
        continue;
      else
        wordP = probIter->second;

      Float curP = wordP * prevBest[j];
      if (maxP < curP) {
        maxP = curP;
        argMax = j;
      }
    }
    if (argMax == -1) {
      // TODO: possible disconnection chain
      // have to segment at previous character
      argMax = i-1;
    }

    prevBest.emplace_back(maxP);
    lastSegment.emplace_back(argMax);
  }

  // main a stack of segmented points
  std::stack<Integral> segmentPoint;
  segmentPoint.push(static_cast<Integral>(sentence.size()));

  for (Integral i = lastSegment.back(); i > 0; i = lastSegment[i])
    segmentPoint.push(lastSegment[i]);

  std::vector<std::string> result;
  if (segmentPoint.empty()) return result;

  Integral start = segmentPoint.top();
  segmentPoint.pop();
  while (!segmentPoint.empty()) {
    Integral end = segmentPoint.top();
    result.emplace_back(sentence.substr(start, end-start));
    start = end;
    segmentPoint.pop();
  }
  return result;
}
#endif /* _NEWJOY_VITERBI_HPP_ */
