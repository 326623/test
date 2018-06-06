#include <opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>

using std::vector;
using std::pair;
using std::string;
using std::cout;
using std::endl;
using std::cerr;
using std::ostream;
using namespace cv;


#define SPLIT_NUM 5 // 4
#define X_OFFSET 0
#define Y_OFFSET 1
#define C_OFFSET 2
#define ALL_OFFSET 3
#define LABEL_COLUMN 1
const double DATA_SET_WEIGHT[] = {311, 333, 404, 158, 594, 227, 272, 199, 193, 835};
const int NUMBER_CLASSES = 10;
const string TRAIN_PATH[3] = {"../models/action_train/", "_split_", ".json"};
const string TEST_PATH[3] = {"../models/action_test/", "_split_", ".json"};
const string DATA_SET_PATH[3] = {"../models/action/", "_split_", ".json"};
typedef pair<int, int> row_col;

int load_data(const std::string &file_name, const std::string PATH[],
              std::vector<float> *pFeatureVec, std::vector<int> *pLabelsVec) {

  try {
    vector<FileStorage> fs;

    // patch up pieces of json file
    int feature_length = 0;
    int row = 0;
    for (int i = 0; i < SPLIT_NUM; ++ i) {
      string json_file = PATH[0] + file_name + PATH[1] + std::to_string(i) + PATH[2];
      FileStorage tmp_fs(json_file, 0);
      fs.push_back(tmp_fs);
      cout << json_file << " -- lines number: " << tmp_fs[file_name].size() << endl;
      row += tmp_fs[file_name].size();
    }

    // initialize feature Vector
    feature_length = fs[0]["feature_length"];
    vector<float> featureVec;
    vector<int> labelsVec;

    // parse json file for read in
    for (int i = 0; i < SPLIT_NUM; ++ i) {
      FileNode root = fs[i][file_name];
      for (int j = 0; j < root.size(); ++ j) {
        labelsVec.push_back(root[j]["label"].real());
        //cout << root[j]["label"].real() << endl;

        vector<float> recev = vector<float>(feature_length, 0.0f);

        for (int counter = 0, rc = 0; counter < root[j]["joints"].size(); ++ counter) {
          if(counter % ALL_OFFSET != C_OFFSET) {
            recev[rc++] = root[j]["joints"][counter];
          }
        }

        //cout << recev.size() << endl;

        // vector append
        featureVec.insert(featureVec.end(), recev.begin(), recev.end());
      }
    }

    //assert(featureVec[0] == 0.646 && "feature read error");

    *pFeatureVec = featureVec;
    *pLabelsVec = labelsVec;
    feature_size->first = row;
    feature_size->second = feature_length;
    label_size->first = row;
    label_size->second = 1;
  }

  catch(const Exception& e) {
    cout << "error: " << e.what() << std::endl;
  }
  return 0;
}

int training_svm_and_save() {

}

int main() {

}
