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

template <typename Iterator, typename Contain>
int load_data(Iterator first, Iterator last,
              std::vector<Contain> &matrix,
              bool feature = true) {

  using namespace boost::spirit;
  using qi::double_;
  using qi::int_;
  using qi::phrase_parse;
  using ascii::space;

  if ( feature )
    phrase_parse(first, last,
                 ('[' >>
                  ('[' >> double_ % ',' >> ']') %','
                  >> ']'),
                 space,
                 matrix);
  else {
    phrase_parse(first, last,
                 ('['
                  >>
                  int_ % ','
                  >>
                  ']'),
                 space,
                 matrix);
  }


  if ( first != last ) return -1;

  return 0;
}

std::string load_all_from_file(const char * filename) {
  std::ifstream in(filename);
  if ( in ) {
    std::string contents;
    in.seekg(0, std::ios::end);
    contents.resize(in.tellg());
    in.seekg(0, std::ios::beg);
    in.read(&contents[0], contents.size());
    in.close();

    return contents;
  }
  throw "no good reading";
}

int training_svm_and_save(const string &saved_file_prefix,
                          Ptr<ml::TrainData> dataset) {

  using namespace cv::ml;
  Ptr<SVM> svm = SVM::create();
  svm->setType(SVM::C_SVC);
  svm->setKernel(SVM::RBF);
  svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
  //int sum = std::accumulate(weight.begin(), weight.end(), 0);
  //Mat weightMat(weight);
  //svm->setClassWeights(weightMat);
  // svm->setC(10);
  // svm->setGamma(35);
  // svm->train(dataset);

  // kFold = 10

  svm->trainAuto(dataset, 10,
                 // default C grid
                 ml::SVM::getDefaultGrid(SVM::C),
                 // overriding default grid of Gamma to 1 ~ 100
                 ml::ParamGrid(1, 50, 3));

  // auto defaultgrid = ml::SVM::getDefaultGrid(SVM::GAMMA);
  // cout << defaultgrid.minVal << ' ' << defaultgrid.maxVal
  //      << ' ' << defaultgrid.logStep << '\n';
  Mat resp;
  std::cout << 100 - svm->calcError(dataset, true, resp) << endl;

  std::cout << 100 - svm->calcError(dataset, false, resp) << endl;

  cout << "the best C is: " << svm->getC() << "  "
       << "the best gamma is: " << svm->getGamma() << '\n';

  svm->save("SVM_FIGHTS.yml");
}

int main() {
  auto timerBegin = std::chrono::high_resolution_clock::now();

  std::string feature_str(load_all_from_file("feature_exclude.json")),
    label_str(load_all_from_file("label_exclude.json"));

  std::vector<std::vector<float>> feature;
  std::vector<int> label;

  load_data(feature_str.begin(), feature_str.end(), feature);
  load_data(label_str.begin(), label_str.end(), label, false);

  Mat featureMat(feature.size(), 30, CV_32F);

  // have to do such conversion, which means OpenCV doesn't play well with stl
  for ( auto i = 0u; i < featureMat.rows; ++ i ) {
    for ( auto j = 0u; j < featureMat.cols; ++ j ) {
      featureMat.at<float>(i, j) = feature[i][j];
    }
  }

  Mat labelMat(label);

  Ptr<ml::TrainData> dataset = ml::TrainData::create(featureMat,
                                                     ml::ROW_SAMPLE,
                                                     labelMat);

  dataset->setTrainTestSplitRatio(0.5);
  training_svm_and_save("poop",
                        dataset);

  // cout << feature.size() << endl;
  // for ( auto i = 0u; i < feature.size(); ++ i ) {
  //   if ( feature[i].size() != 30u ) { throw "gg"; }
  // }
  // cout << label[0] << endl;
  // cout << label.size() << endl;

  // this method is highly inefficient during large file
  // //std::string s;
  // //inputstream >> s;
  // //cout << s << endl;
  // std::string str((std::istreambuf_iterator<char>(inputstream)),
  //                 std::istreambuf_iterator<char>());
  // //cout << str << endl;

  auto now = std::chrono::high_resolution_clock::now();
  auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now-timerBegin).count()
    * 1e-9;
  cout << totalTimeSec << endl;

  // using rdbuf is very efficient

  // timerBegin = std::chrono::high_resolution_clock::now();
  // std::ifstream inputstream("feature_exclude.json");
  // inputstream = std::ifstream("feature_exclude.json");

  // std::stringstream buffer;
  // buffer << inputstream.rdbuf();
  // str = std::string(buffer.str());

  // now = std::chrono::high_resolution_clock::now();
  // totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now - timerBegin).count()
  //   * 1e-9;
  // cout << totalTimeSec << endl;
}
