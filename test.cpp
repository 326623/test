
// #include <opencv2/core.hpp>
// #include <iostream>
// #include <string>
// #include <vector>
// using std::vector;
// using std::string;
// using std::cout;
// using std::endl;
// using std::cerr;
// using std::ostream;
// using namespace cv;



// // struct data {
// //   float a,b,c;
// // };

// int main(int argc, char** argv)
// {


//   try {
// //    FileStorage fs("svm_1_split_0.json", 0); // use postfix to know it's json, 0 for read
//     FileStorage fs("svm_1.json", 0);
//     FileNode root = fs[(string)"svm" + (string)"_1"];
//     cout << root.size() << std::endl;
//     cout << fs.getFormat() << std::endl;
//   }
//   catch(const Exception& e) {
//     cout << "error: " << e.what() << std::endl;
//   }
// }

#include <iostream>

using std::cout;

int main(int argc, char** argv) {
  auto glambda = [](int a, double&& b) { return a < b; }; // type unknow
  cout << glambda(3, 3.14);

  return 0;
}
