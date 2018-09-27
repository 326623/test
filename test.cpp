// tensorflow/cc/example/example.cc

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/core/framework/tensor.h"

int main() {
  using namespace tensorflow;
  Scope root = Scope::NewRootScope();
  auto c = ops::Const(root, { {1, 1} });
  auto m = ops::MatMul(root, c, { {42}, {1} });
  ClientSession session(root);
  std::vector<Tensor> outputs;
  Status run_status = session.Run({m}, &outputs);
  if (!run_status.ok()) {
    std::cout << run_status.ToString() << "\n";
    return 1;
  }
  auto output_c = outputs[0].scalar<int>();
  std::cout << outputs[0].DebugString() << "\n";
  std::cout << output_c() << "\n";
}
