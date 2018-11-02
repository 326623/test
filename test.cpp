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

int main() {
  std::random_device rd;
  std::mt19937 gen(rd());

  std::poisson_distribution<> d(4);

  std::map<int, int> hist;
  for (int n = 0; n < 10000; ++ n) {
    ++hist[d(gen)];
  }
  for (auto p : hist) {
    std::cout << p.first
              <<  ' ' << std::string(p.second/100, '*') << '\n';
  }
}
