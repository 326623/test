#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/spirit/include/qi.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

using std::vector;
using std::pair;
using std::string;
using std::cout;
using std::endl;
using std::cerr;
using std::ostream;
using namespace cv;


#define SPLIT_NUM 15 // 4
#define X_OFFSET 0
#define Y_OFFSET 1
#define C_OFFSET 2
#define ALL_OFFSET 3
#define LABEL_COLUMN 1
//const double DATA_SET_WEIGHT[] = {311, 333, 404, 158, 594, 227, 272, 199, 193, 835};
const int NUMBER_CLASSES = 10;
const string PATH[2] = {"_split_", ".json"};
typedef pair<int, int> row_col;

template <typename Iterator>
int load_data(Iterator first, Iterator last) {

  using namespace boost::spirit;
  using qi::double_;
  using qi::phrase_parse;
  using ascii::space;

  phrase_parse(first, last);

  return 0;
}

int training_svm_and_save() {

}

int main() {
  const auto timerBegin = std::chrono::high_resolution_clock::now();

  std::ifstream inputstream("feature_exclude.json");
  //std::string s;
  //inputstream >> s;
  //cout << s << endl;
  std::string str((std::istreambuf_iterator<char>(inputstream)),
                  std::istreambuf_iterator<char>());
  //cout << str << endl;

  const auto now = std::chrono::high_resolution_clock::now();
  const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now-timerBegin).count()
    * 1e-9;
  cout << totalTimeSec << endl;
}
