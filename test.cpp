// #include <vector>
// #include <algorithm>
// #include <iostream>
// #include <random>
// //#include <cstdlib>
// // insertion sort
// template <typename Iterator>
// void insertionSort(Iterator first, Iterator last) {

//   for (Iterator ith = first; ith != last; ++ ith) {

//     Iterator back = ith;
//     // pick the ith element and insert back
//     while (back != first && *(back) < *(back-1)) {
//       std::swap(*(back), *(back-1));
//       back--;
//     }
//   }
// }

// template <typename ForwardIterator>
// void display(ForwardIterator first, ForwardIterator last) {
//   for (; first != last; ++ first)
//     std::cout << *first << ' ';
//   std::cout << std::endl;
// }

// // check if is sorted
// template <typename ForwardIterator>
// bool check(ForwardIterator first, ForwardIterator last) {
//   // somewhat less efficient here
//   for (; first+1 != last; ++ first)
//     if (*first > *(first+1)) return false;
//   return true;
// }

// int main() {
//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_int_distribution<> dis(1, 10000);
//   std::vector<int> test_vec(1000);
//   for (auto & item : test_vec)
//     item = dis(gen);

//   //  display(test_vec.begin(), test_vec.end());
//   std::cout << check(test_vec.begin(), test_vec.end()) << std::endl;

//   insertionSort(test_vec.begin(), test_vec.end());

//   // display(test_vec.begin(), test_vec.end());
//   std::cout << check(test_vec.begin(), test_vec.end()) << std::endl;
// }

// tensorflow/cc/example/example.cc

// #include "tensorflow/cc/client/client_session.h"
// #include "tensorflow/cc/ops/standard_ops.h"
// #include "tensorflow/core/framework/tensor.h"

// int main() {
//   using namespace tensorflow;
//   using namespace tensorflow::ops;
//   Scope root = Scope::NewRootScope();
//   // Matrix A = [3 2; -1 0]
//   auto A = Const(root, { {3.f, 2.f}, {-1.f, 0.f} });
//   // Vector b = [3 5]
//   auto b = Const(root, { {3.f, 5.f} });
//   // v = Ab^T
//   auto v = MatMul(root.WithOpName("v"), A, b, MatMul::TransposeB(true));
//   std::vector<Tensor> outputs;
//   ClientSession session(root);
//   // Run and fetch v
//   TF_CHECK_OK(session.Run({v}, &outputs));
//   // Expect outputs[0] == [19; -3]
//   LOG(INFO) << outputs[0].matrix<float>();
//   return 0;
// }
// #include <iostream>
// #include <algorithm>

// int n; // the function instantiated will have a pointer to it

// template <typename Integral>
// void perm(Integral a[], Integral k) {
//   if (k == 0) {
//     for(Integral i = 0; i < n; ++ i)
//       std::cout << a[i];
//     std::cout << '\n';
//   }
//   else {
//     const Integral start = n - k;
//     for (Integral i = start; i < n; ++ i) {
//       std::swap(a[i], a[start]);
//       perm(a, k-1);
//       std::swap(a[i], a[start]);
//     }
//   }
// }

// int main() {
//   std::cin >> n;
//   int *a = new int[n];

//   for (int i = 0; i < n; ++ i)
//     std::cin >> a[i];

//   perm(a, n);
// }

// class Solution {
// public:
//   string getPermutation(int n, int k) {
//     std::vector<int> factorial_num(n-1, 1);
//     std::string res;
//     for (int i = 1; i < factorial_num.size(); ++ i)
//       factorial_num[i] = (i + 1) * factorial_num[i-1];

//     int index = k-1;
//     for (int i = factorial_num.size()-1; i >= 0; -- i) {
//       int num = index / factorial_num[i];
//       res += static_cast<char>(num + '0');
//       index = index % factorial_num[i];
//     }
//     res += static_cast<char>(index);
//     return res;
//   }
// };

// #include <stdio.h>
// #include <time.h>
// #define SIZE 25600
// int a[SIZE], b[SIZE], c[SIZE];
// void foo () {
//   int i,j;
//   for (j=0; j<SIZE; ++j) {
//     for (i=0; i<SIZE; i++){
//       a[i] = b[i] + c[i];
//     }
//   }

// }

// int main() {
//   clock_t t;
//   t = clock();
//   foo();
//   t = clock() - t;
//   float usedTime = ((float)t)/CLOCKS_PER_SEC;
//   long long int numComp = SIZE * SIZE;
//   printf("%f FLOPS", numComp / usedTime / 1024 / 1024 / 1024);

//   return 0;
// }

// #include <gtest/gtest.h>
// #include "viterbi.hpp"

// TEST(VITERBIALGO, TEST1) {
//   // std::map<std::string, float> probDict = {std::make_pair("good", 0.2),
//   //                                          std::make_pair("morning", 0.2),
//   //                                          std::make_pair("sir", 0.2),
//   //                                          std::make_pair("could", 0.2),
//   //                                          std::make_pair("please", 0.2)};

//   std::map<std::string, float> probDict = {{"good", 0.2},
//                                            {"morning", 0.2},
//                                            {"sir", 0.2},
//                                            {"could", 0.2},
//                                            {"please", 0.2}};


//   std::string sentence("goodmorningMrBlue");

//   auto strVec = viterbiSegment(sentence, probDict);// << '\n';

//   for (const auto &str : strVec)
//     std::cout << str << ' ';
//   std::cout << '\n';
// }
#include <fstream>
#include <iostream>
#include <atomic>
#include <regex>
#include <unordered_map>
#include <benchmark/benchmark.h>
#include <Vc/Vc>

// Int main() {
//   // std::regex re("[a-zA-Z]+|[0-9]+");
//   // std::string s = "http://davidroyko.webs.com/hrreuniontrib.htm/%&$@";
//   // auto words_begin = std::sregex_iterator(s.begin(), s.end(), re);
//   // auto words_end = std::sregex_iterator();

//   // std::size_t last = 0;
//   // while (words_begin != words_end) {
//   //   std::cout << words_begin->format("$`") << ' ' << words_begin->format("$&") << ' ';
//   //   last = words_begin->position() + words_begin->length();
//   //   ++words_begin;
//   // }
//   // if (last != s.length()) {
//   //   std::cout << s.substr(last) << '\n';
//   // }

//   std::unordered_map<std::string, long> dictionary;
//   std::hash<std::string> hashFun = dictionary.hash_function();
//   std::cout << hashFun("sssssss") << '\n';
// }

// using Vc::float_v;

// static constexpr std::size_t N = 10240000, PrintStep = 1000000;

// static constexpr float epsilon = 1e-7f;
// static constexpr float lower = 0.f;
// static constexpr float upper = 40000.f;
// static constexpr float h = (upper - lower) / N;

// static inline float fu(float x) { return ( std::sin(x) ); }
// static inline float dfu(float x) { return ( std::cos(x) ); }

// static inline Vc::float_v fu(Vc::float_v::AsArg x) {
//   #ifdef USE_SCALAR_SINCOS
//   Vc::float_v r;
//   for (size_t i = 0; i < Vc:float_v::Size; ++ i) {
//     r[i] = std::sin(x[i]);
//   }
//   return r;
//   #else
//   return Vc::sin(x);
//   #endif
// }

// // Passing by value is good enough, since sane compiler
// // would just pass value to register, but using reference is overkill
// // since it would require stack operation
// static inline Vc::float_v dfu(Vc::float_v::AsArg x) {
//   #ifdef USE_SCALAR_SINCOS
//   Vc::float_v r;
//   for (size_t i = 0; i < Vc::float_v::Size; ++ i) {
//     r[i] = std::cos(x[i]);
//   }
//   return r;
//   #else
//   return Vc::cos(x);
//   #endif
// }

// static void BM_Simple(benchmark::State &state) {
//   //std::unordered_map<std::string, long> dictionary;
//   //auto hashFun =  dictionary.hash_function();
//   //std::ios::sync_with_stdio(false);
//   float_v f = 1;
//   for (auto _ : state) {
//     for (size_t i = 0; i < Vc::float_v::Size; ++ i) {
//       //benchmark::DoNotOptimize(std::sin(f[i]));
//       benchmark::DoNotOptimize(f[i] = f[i] + f[i]);
//     }
//     //hashFun("AnyStringAtAllPleaseSuggestSomethingLongerThanThatWouldYouMind");
//   }
// }

// static void BM_VC(benchmark::State &state) {
//   //std::unordered_map<std::string, long> dictionary;
//   //auto hashFun =  dictionary.hash_function();
//   //std::ios::sync_with_stdio(false);
//   float_v f = 1;
//   for (auto _ : state) {
//     //    std::cout << fu(f) << '\n';
//     benchmark::DoNotOptimize(f=f+f);
//     //hashFun("AnyStringAtAllPleaseSuggestSomethingLongerThanThatWouldYouMind");
//   }
// }

// BENCHMARK(BM_Simple);
// BENCHMARK(BM_VC);

// BENCHMARK_MAIN();

// int main() {
//   std::random_device rd;
//   std::mt19937 gen(rd());

//   std::poisson_distribution<> d(4);

//   std::map<int, int> hist;
//   for (int n = 0; n < 10000; ++ n) {
//     ++hist[d(gen)];
//   }
//   for (auto p : hist) {
//     std::cout << p.first
//               <<  ' ' << std::string(p.second/100, '*') << '\n';
//   }
// }

// #include <ortools/linear_solver/linear_solver.h>
// #include <ortools/linear_solver/linear_solver.pb.h>

// namespace operations_research {
//   void RunTest(
//     MPSolver::OptimizationProblemType optimization_problem_type) {
//     MPSolver solver("LinearExample", optimization_problem_type);
//     const double infinity = solver.infinity();
//     MPVariable* const x = solver.MakeNumVar(0.0, infinity, "x");
//     MPVariable* const y = solver.MakeNumVar(0.0, infinity, "y");
//     // Objective function: 3x + 4y.
//     MPObjective* const objective = solver.MutableObjective();
//     objective->SetCoefficient(x, 3);
//     objective->SetCoefficient(y, 4);
//     // x + 2y <= 14.
//     MPConstraint* const c0 = solver.MakeRowConstraint(-infinity, 14.0);
//     c0->SetCoefficient(x, 1);
//     c0->SetCoefficient(y, 2);

//     // 3x - y >= 0.
//     MPConstraint* const c1 = solver.MakeRowConstraint(0.0, infinity);
//     c1->SetCoefficient(x, 3);
//     c1->SetCoefficient(y, -1);

//     // x - y <= 2
//     MPConstraint* const c2 = solver.MakeRowConstraint(-infinity, 2.0);
//     c2->SetCoefficient(x, 1);
//     c2->SetCoefficient(y, -1);
//     printf("\nNumber of variables = %d", solver.NumVariables());
//     printf("\nNUMber of constraints = %d", solver.NumConstraints());
//     solver.Solve();
//     // the value of each variable in the solution.
//     printf("\nSolution:");
//     printf("\nx = %.1f", x->solution_value());
//     printf("\ny = %.1f", y->solution_value());

//     // The objective value of the solution.
//     printf("\nOptimal objective value = %.1f", objective->Value());
//     printf("\n");
//   }
//   void RunExample() {
//     RunExample(MPSolver::GLOP_LINEAR_PROGRAMMING);
//   }
// }

// int main(int , char** ) {
//   operations_research::RunExample();
//   return 0;
// }

// #include <ortools/base/commandlineflags.h>
// #include "ortools/linear_solver/linear_solver.h"
// #include "ortools/linear_solver/linear_solver.pb.h"

// namespace operations_research {
//   void RunTest(
//     MPSolver::OptimizationProblemType optimization_problem_type) {
//     MPSolver solver("Glop", optimization_problem_type);
//     const double infinity = solver.infinity();
//     // Create the variables x and y.
//     MPVariable* const x = solver.MakeNumVar(-infinity, infinity, "x");
//     MPVariable* const y = solver.MakeNumVar(-infinity, infinity, "y");

//     // x + 2y <= 14
//     //MPConstraint* const c0 = solver.MakeRowConstraint(-infinity, 14.0);
//     // const auto
//     auto* const c0 = solver.MakeRowConstraint(-infinity, 14.0);
//     c0->SetCoefficient(x, 1.0);
//     c0->SetCoefficient(y, 2.0);

//     // 3x - y >= 0
//     auto* const c1 = solver.MakeRowConstraint(0.0, infinity);
//     c1->SetCoefficient(x, 3.0);
//     c1->SetCoefficient(y, -1.0);

//     // x - y <= 2
//     auto* const c2 = solver.MakeRowConstraint(-infinity, 2.0);
//     c2->SetCoefficient(x, 1.0);
//     c2->SetCoefficient(y, -1.0);

//     // Create the objective function, x + y.
//     MPObjective* const objective = solver.MutableObjective();
//     objective->SetCoefficient(x, 3.0);
//     objective->SetCoefficient(y, 4.0);
//     objective->SetMaximization();
//     // Call the solver and display the results.
//     solver.Solve();
//     printf("\nSolution:");
//     printf("\nx = %.1f", x->solution_value());
//     printf("\ny = %.1f", y->solution_value());
//     printf("\nOptimal objective value = %.1f", objective->Value());
//     printf("\n");
//   }

//   void RunExample() {
//     RunTest(MPSolver::GLOP_LINEAR_PROGRAMMING);
//   }
// }

// #include <ortools/base/commandlineflags.h>
// #include <ortools/base/logging.h>
// #include "ortools/linear_solver/linear_solver.h"
// #include "ortools/linear_solver/linear_solver.pb.h"

// DEFINE_string(input, "", "Jobshop data file name.");

// int main(int argc, char** argv) {
//   gflags::ParseCommandLineFlags(&argc, &argv, true);
//   if (FLAGS_input.empty()) {
//     LOG(FATAL) << "Please supply a data file with --input=";
//   }
//   std::cout << FLAGS_input << '\n';
//   //operations_research::RunExample();
//   return 0;
// }

// #include <ortools/constraint_solver/constraint_solver.h>
// #include <ortools/base/logging.h>
// namespace operations_research {
//   void pheasant() {
//     Solver s("pheasant");
//     IntVar* const p = s.MakeIntVar(0, 20, "pheasant");
//     IntVar* const r = s.MakeIntVar(0, 20, "rabbit");
//     IntExpr* const legs = s.MakeSum(s.MakeProd(p, 2), s.MakeProd(r, 4));
//     IntExpr* const heads = s.MakeSum(p, r);
//     Constraint* const ct_legs = s.MakeEquality(legs, 56);
//     Constraint* const ct_heads = s.MakeEquality(heads, 20);
//     CHECK_EQ(0, 0);
//     s.AddConstraint(ct_legs);
//     s.AddConstraint(ct_heads);
//     DecisionBuilder* const db = s.MakePhase(p, r,
//                                             Solver::CHOOSE_FIRST_UNBOUND,
//                                             Solver::ASSIGN_MIN_VALUE);
//     s.NewSearch(db);
//     CHECK(s.NextSolution());
//     LOG(INFO) << "rabbits -> " << r->Value() << ", pheasants -> "
//               << p->Value();
//     LOG(INFO) << s.DebugString();
//     s.EndSearch();
//   }
// }

// int main(int , char** ) {
//   operations_research::pheasant();
//   return 0;
// }

#include <iostream>
#include <sstream>
#include <cassert>
#include <ortools/base/logging.h>
#include <Eigen/Dense>

//using namespace Eigen;
//using Eigen::MatrixXd;

// template <typename ... Args>
// std::string numPacking() {
//   return "";
// }

// template <typename IndexType, typename ... Args>
// std::string numPacking(IndexType head, Args ... tail) {
//   if (sizeof...(tail))
//     return std::to_string(head) + ',' + numPacking(tail...);
//   else
//     return std::to_string(head);
// }

// template <typename ... IndexTypes>
// std::string numToBracket(IndexTypes ... indices) {
//   return '[' + numPacking(indices...) + ']';
// }

// template <typename ... IndexTypes>
// std::string makeName(std::string &&prefix, IndexTypes ... nums) {
//   if (sizeof...(nums))
//     return prefix + '_' + numToBracket(nums...);
//   else
//     return prefix;
// }

// template <typename I, typename P>
// I stable_partition(I f, I l, P p) {
//   auto n = l - f;
//   // return the first element that returns false from p
//   if (n == 0) return f;
//   if (n == 1) return f + p(*f); // return true, skip to next

//   auto m = f + (n / 2);
//   return std::rotate(stable_partition(f, m), m,
//                      stable_partition(m+1, l));
// }

// template <typename Integral>
// void permutation(Integral k, std::vector<Integral> &xs) {

//   const auto n = static_cast<Integral>(xs.size());
//   if (k == n) {
//     for (const auto &x: xs)
//       std::cout << x << ' ';
//     std::cout << '\n';
//   }
//   else {
//     for (Integral i = k; i < n; ++ i) {
//       std::swap(xs[k], xs[i]);
//       permutation(k+1, xs);
//       std::swap(xs[k], xs[i]);
//     }
//   }
// }

// int main()
// {
//   std::vector<int> xs = {1, 2, 3, 4};
//   permutation(0, xs);
  //std::cout << makeName("machine", 1, 2) << '\n';
  // Matrix<double, Dynamic, Dynamic> m(4, 4);
  // m(0, 1) = true;
  // m(0, 2) = true;
  // m(2, 3) = true;
  // std::cout << (Matrix<double, Dynamic, Dynamic>::Identity(m.rows(), m.cols()) - m).inverse() << '\n';
  // DLOG(INFO) << "Found cookies";
  // MatrixXd m(2, 2);
  // m(0, 0) = 3;
  // m(1, 0) = 2.5;
  // m(0, 1) = -1;
  // m(1, 1) = m(1, 0) + m(0, 1);
  // std::cout << m << '\n';
  // auto i = 0ul;
  // std::stringstream ss("1 1");
  // ss >> i >> i >> i;
  // if (!ss) {
  //   std::cout << "bad number passed\n";
  // }
  // else {
  //   std::cout << "good number passed\n";
  // }
  // //CHECK_EQ(0, 1);
  // assert(0 == 1);
  // std::cout << i << '\n';
  // auto a = std::numeric_limits<double>::max();
  // std::cout << (a+2312312) << '\n';
// }
// #include <chrono>
// #include <thread>
// int main() {
//   std::atomic<int> x{0};
//   std::atomic<int> y{0};

//   std::thread t1([&]() {
//                    auto r1 = y.load(std::memory_order_relaxed);
//                    x.store(r1, std::memory_order_relaxed);
//                    std::cout << r1 << '\n';
//                  });
//   std::thread t2([&]() {
//                    auto r2 = x.load(std::memory_order_relaxed);
//                    y.store(42, std::memory_order_relaxed);
//                    std::cout << r2 << '\n';
//                  });
//   t1.join();
//   t2.join();
// }
// #include <vector>
// #include <iostream>
// #include <thread>
// #include <atomic>

// std::atomic<int> cnt = {0};

// void f() {
//   for (int n = 0; n < 100000; ++ n) {
//     cnt.fetch_add(1);
//   }
// }

// int main() {
//   std::vector<std::thread> v;
//   auto start = std::chrono::high_resolution_clock::now();
//   for (int n = 0; n < 1000; ++ n) {
//     v.emplace_back(f);
//   }
//   for (auto &t : v) {
//     t.join();
//   }
//   auto end = std::chrono::high_resolution_clock::now();
//   std::cout << std::chrono::duration<double>(end - start).count() << '\n';
//   std::cout << "Final counter value is " << cnt << '\n';
// }
// #include <thread>
// #include <vector>
// #include <iostream>
// #include <atomic>

// std::atomic_flag lock = ATOMIC_FLAG_INIT;

// void f(int n)
// {
//   for (int cnt = 0; cnt < 100; ++cnt) {
//     while (lock.test_and_set(std::memory_order_acquire))  // acquire lock
//       ; // spin
//     std::cout << "Output from thread " << n << '\n';
//     lock.clear(std::memory_order_release);               // release lock
//   }
// }

// int main()
// {
//   std::vector<std::thread> v;
//   for (int n = 0; n < 10; ++n) {
//     v.emplace_back(f, n);
//   }
//   for (auto& t : v) {
//     t.join();
//   }
// }
// #include <thread>
// #include <atomic>
// #include <cassert>
// #include <vector>

// std::vector<int> data;
// std::atomic<int> flag = {0};

// void thread_1() {
//   data.push_back(42);
//   flag.store(1, std::memory_order_relaxed);
// }

// void thread_2() {
//   int expected=1;
//   while (!flag.compare_exchange_strong(expected, 2, std::memory_order_relaxed)) {
//     expected = 1;
//   }
// }

// void thread_3() {
//   while (flag.load(std::memory_order_relaxed) < 2) ;
//   assert(data.at(0) = 42);
// }

// int main() {
//   std::thread c(thread_3);
//   std::thread b(thread_2);
//   //std::thread a(thread_1);
//   c.join();
//   b.join();
//   //a.join();
// }

template <typename Iterator>
Iterator find_max1(Iterator first, Iterator last) {
  if (first == last) return last;
  auto maxIter = first++;
  for (; first != last; ++first) {
    if (*maxIter < *first)
      maxIter = first;
  }
  return maxIter;
}

template <typename Iterator>
Iterator find_max2(Iterator first, Iterator last) {
  if (first == last) return last;

  auto max = *first;
  auto maxIter = first++;
  for (; first != last; ++first) {
    if (max < *first) {
      max = *first;
      maxIter = first;
    }
  }

  return maxIter;
}

// template <typename Iterator>
// Iterator find_max3(Iterator first, Iterator last) {

// }

// static inline Vc::float_v fu(Vc::float_v::AsArg x) {
//   #ifdef USE_SCALAR_SINCOS
//   Vc::float_v r;
//   for (size_t i = 0; i < Vc:float_v::Size; ++ i) {
//     r[i] = std::sin(x[i]);
//   }
//   return r;
//   #else
//   return Vc::sin(x);
//   #endif
// }
