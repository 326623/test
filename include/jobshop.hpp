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
#ifndef _NEWJOY_JOBSHOP_HPP_
#define _NEWJOY_JOBSHOP_HPP_

#include <cstdio>
#include <cstdlib>

#include "ortools/base/filelineiter.h"
#include "ortools/base/integral_types.h"
#include "ortools/base/logging.h"
#include "ortools/base/split.h"
#include "ortools/base/stringprintf.h"
#include "ortools/base/strtoint.h"

namespace operations_research {
  class JobShopData {
  public:
    struct Task {
      Task(int j, int m, int d) : job_id(j), machine_id(m), duration(d) {}
      int job_id;
      int machine_id;
      int duration;
    };

    enum ProblemType {UNDEFINED, JSSP, TAILLARD};

    enum TaillardState{
      START,
      JOBS_READ,
      MACHINES_READ,
      SEED_READ,
      JOB_ID_READ,
      JOB_LENGTH_READ,
      JOB_READ
    };

    JobShopData()
      : name_(""),
        machine_count_(0),
        job_count_(0),
        horizon_(0),
        current_job_index_(0),
        problem_type_(UNDEFINED),
        taillard_state_(START) {}

    void Load(const std::string &filename) {
      for (const std::string &line : FileLines(filename)) {
        if (line.empty()) {
          continue;
        }
        ProcessNewLine(line);
      }
    }

    int machine_count() const { return machine_count_; }

    int job_count() const { return job_count_; }

    const std::string &name() const { return name_; }

    int horizon() const { return horizon_; }

    const std::vector<Task> &TaskOfJob(int job_id) const {
      return all_tasks_[job_id];
    }

  private:
    void ProcessNewLine(const std::string &line) {
      const std::vector<std::string> words = absl::StrSplit(line, ' ', absl::SkipEmpty());
      switch (problem_type_) {
        case UNDEFINED: {
          if (words.size() == 2 && words[0] == "instance") {
            problem_type_ = JSSP;
            LOG(INFO) << "Reading jssp instance " << words[1];
            name_ = words[1];
          } else if (words.size() == 1 && atoi32(words[0]) > 0) {
            problem_type_ = TAILLARD;
            taillard_state_ = JOBS_READ;
            job_count_ = atoi32(words[0]);
            CHECK_GT(job_count_, 0);
            all_tasks_.resize(job_count_);
          }
          break;
        }
        case JSSP: {
          if (words.size() == 2) {
            job_count_ = atoi32(words[0]);
            machine_count_ = atoi32(words[1]);
            CHECK_GT(machine_count_, 0);
            CHECK_GT(job_count_, 0);
            LOG(INFO) << machine_count_ << " machines and " << job_count_
                      << " jobs";
            all_tasks_.resize(job_count_);
          }

          if (words.size() > 2 && machine_count_ != 0) {

          }
        }

      }
    }

    std::string name_;
    int machine_count_;
    int job_count_;
    int horizon_;
    std::vector<std::vector<Task>> all_tasks_;
    int current_job_index_;
    ProblemType problem_type_;
    TaillardState taillard_state_;
  };
}


}

#endif /* _NEWJOY_JOBSHOP_HPP_ */
