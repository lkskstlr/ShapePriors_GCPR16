// C/C++ includes
#include <algorithm>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <time.h>
#include <tuple>
#include <unordered_map>

// OpenCV includes
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// Ceres includes
#include <ceres/ceres.h>

// VIZ includes
#include <viz/viz.h>

// Own includes
#include "io.h"
#include "optimization/pose.h"
#include "optimization/shape.h"
#include "tracking.h"
#include "utils.h"

using std::cout;
using std::endl;

struct parameters {
  std::string image_2;
  std::string detection;
  std::string disparity;
  std::string calibration;
  std::string groundplane;
  std::string result;

  parameters(int argc, char **argv) {
    if (argc != 6 + 1) {
      std::cerr << "Not enough parameters! "
                   "Params: image_2 detection "
                   "disparity calibration "
                   "groundplane result"
                << std::endl;
      std::exit(1);
    }
    image_2 = std::string(argv[1]);
    detection = std::string(argv[2]);
    disparity = std::string(argv[3]);
    calibration = std::string(argv[4]);
    groundplane = std::string(argv[5]);
    result = std::string(argv[6]);
  }

  void print(void) {
    cout << "--- Params ---" << endl;
    cout << "image_2: " << image_2 << endl;
    cout << "detection: " << detection << endl;
    cout << "disparity: " << disparity << endl;
    cout << "calibration: " << calibration << endl;
    cout << "groundplane: " << groundplane << endl;
    cout << "result: " << result << endl;
    cout << "--- ------ ---" << endl << endl;
  }
};

std::string index_to_str(int index) {
  char buffer[7];
  snprintf(buffer, sizeof(buffer), "%06d", index);
  return std::string(buffer);
}

std::string get_path(std::string folder, int index, std::string ext) {
  std::string out = folder;
  if (folder.back() != '/')
    out += "/";
  if (index < 0)
    out += "default";
  else
    out += index_to_str(index);
  out += ext;

  return out;
}

std::vector<int> find_indices(std::string dir) {
  struct dirent *entry = NULL;
  DIR *dp = NULL;

  dp = opendir(dir.c_str());
  std::vector<int> indices;
  const std::regex image_regex("([0-9]{6})\\.txt");
  if (dp) {
    while ((entry = readdir(dp))) {
      if (std::regex_match(entry->d_name, image_regex) == 1) {
        indices.push_back(std::stoi(std::string(entry->d_name).substr(0, 6)));
      }
    }
  }

  closedir(dp);

  std::sort(indices.begin(), indices.end());
  return indices;
}

Eigen::Vector3d get_size(const Eigen::MatrixXd &values) {
  double bin_size = 0.1;
  int dx = 60;
  int dy = 40;
  int dz = 60;

  int min_dx = std::numeric_limits<int>::max();
  int min_dy = std::numeric_limits<int>::max();
  int min_dz = std::numeric_limits<int>::max();
  int max_dx = std::numeric_limits<int>::min();
  int max_dy = std::numeric_limits<int>::min();
  int max_dz = std::numeric_limits<int>::min();

  // Convert 1D-index to 3D-index
  for (int i = 0; i < dx * dy * dz; i++) {
    int x = ((i / dy / dz) % dx) - 30;
    int y = (i / dz) % dy - 30;
    int z = (i % dz) - 30;
    float value = values(i, 0);
    if (value < 0) {
      // Inside
      min_dx = std::min(min_dx, x);
      min_dy = std::min(min_dy, y);
      min_dz = std::min(min_dz, z);
      max_dx = std::max(max_dx, x);
      max_dy = std::max(max_dy, y);
      max_dz = std::max(max_dz, z);
    }
  }

  const double width = (max_dx - min_dx + 1) * bin_size;
  const double height = (max_dy - min_dy + 1) * bin_size;
  const double length = (max_dz - min_dz + 1) * bin_size;

  using std::cout;
  using std::endl;

  return Eigen::Vector3d(width, height, length);
}

inline bool file_exists(const std::string &name) {
  ifstream f(name.c_str());
  return f.good();
}

std::string double_to_string(double x) {
  char buffer[100];
  snprintf(buffer, sizeof(buffer), "%.2f", x);

  return std::string(buffer);
}

void writeDetectionsToFile(std::string &path,
                           std::vector<std::shared_ptr<gvl::Detection>> &dets,
                           std::vector<Eigen::Vector3d> &sizes,
                           std::vector<double> &scores,
                           std::vector<bool> &valid) {
  int sum_valid = 0;
  for (unsigned int i = 0; i < dets.size(); i++) {
    if (valid[i]) {
      sum_valid += 1;
    }
  }

  if (sum_valid == 0) {
    return;
  }

  std::ofstream file;
  file.open(path.c_str());
  if (file.is_open()) {

    for (unsigned int i = 0; i < dets.size(); i++) {
      if (!valid[i]) {
        continue;
      }

      /*
      1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                        'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                        'Misc' or 'DontCare'
      1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                        truncated refers to the object leaving image boundaries.
            Truncation 2 indicates an ignored object (in particular
            in the beginning or end of a track) introduced by manual
            labeling.
      1    occluded     Integer (0,1,2,3) indicating occlusion state:
                        0 = fully visible, 1 = partly occluded
                        2 = largely occluded, 3 = unknown
      1    alpha        Observation angle of object, ranging [-pi..pi]
      4    bbox         2D bounding box of object in the image (0-based index):
                        contains left, top, right, bottom pixel coordinates
      3    dimensions   3D object dimensions: height, width, length (in meters)
      3    location     3D object location x,y,z in camera coordinates (in
      meters)
      1    rotation_y   Rotation ry around Y-axis in camera coordinates
      [-pi..pi]
      1    score        Only for results: Float, indicating confidence in
                        detection, needed for p/r curves, higher is better.*/

      auto &det = dets[i];

      std::string line = "Car -1 -1 -10 "; // type truncated occludedalpha
      line += double_to_string(det->boundingbox->left) + " ";
      line += double_to_string(det->boundingbox->top) + " ";
      line += double_to_string(det->boundingbox->right) + " ";
      line += double_to_string(det->boundingbox->bottom) + " ";
      // line += double_to_string(2) + " ";   // height
      // line += double_to_string(2.) + " ";  // width
      // line += double_to_string(4.2) + " "; // length
      line += double_to_string(sizes[i](1)) + " ";                // height
      line += double_to_string(sizes[i](0)) + " ";                // width
      line += double_to_string(sizes[i](2)) + " ";                // length
      line += double_to_string(det->translation[0]) + " ";        // tx
      line += double_to_string(det->translation[1]) + " ";        // ty
      line += double_to_string(det->translation[2]) + " ";        // tz
      line += double_to_string(det->rotation_y + M_PI / 2) + " "; // ry
      line += double_to_string(scores[i]);                        // score
      file << line << std::endl;
    }

  } else {
    std::cout << "Unable to open file: " << path << std::endl;
    return;
  }

  file.close();
}

int main(int argc, char **argv) {
  // Read parameters
  parameters params(argc, argv);
  bool print_params = true;
  if (print_params)
    params.print();
  auto indices = find_indices(params.detection);

  cout << "Found " << indices.size() << " images" << endl;

  int total_instances = 0;
  int total_valid = 0;

  for (auto index : indices) {
    // Read calibration, poses
    auto calib_path = get_path(params.calibration, index, ".txt");
    // cout << "\t\tReading calib" << endl;
    std::vector<Eigen::Matrix4d> projection_matrices =
        gvl::readCalibrationFile(calib_path);

    // // Build K matrix
    Eigen::Matrix3d K = projection_matrices.at(0).block<3, 3>(0, 0);
    double f = projection_matrices.at(0)(0, 0);
    double baseline = projection_matrices.at(1)(0, 3) / -f;
    bool fix_baseline = true;
    if (fix_baseline) {
      baseline += projection_matrices.at(0)(0, 3) / f;
    }

    // // Read SVD decompositions
    Eigen::MatrixXd &S = gvl::ShapeSpacePCA::instance().S;
    Eigen::MatrixXd &V = gvl::ShapeSpacePCA::instance().V;
    Eigen::MatrixXd &mean = gvl::ShapeSpacePCA::instance().mean;
    gvl::readMatrixFromBinaryFile(path_S, S);
    gvl::readMatrixFromBinaryFile(path_V, V);
    gvl::readMatrixFromBinaryFile(path_mean_shape, mean);

    // All detections
    std::vector<std::shared_ptr<gvl::Detection>>
        all_detections; // all 3dop detections
    std::vector<Eigen::Vector3d> all_sizes;
    std::vector<double> all_scores;
    std::vector<bool> all_valid;

    //**********************************************************************************************
    // Iterate over frames in window to accumulate data
    //**********************************************************************************************

    // Set up pathes for current frame
    auto detection_path = get_path(params.detection, index, ".txt");
    auto disparity_path = get_path(params.disparity, index, ".png");
    auto image_2_path = get_path(params.image_2, index, ".png");
    auto result_path = get_path(params.result, index, ".txt");
    auto groundplane_path = get_path(params.groundplane, index, ".txt");
    if (!file_exists(groundplane_path)) {
      groundplane_path = get_path(params.groundplane, -1, ".txt");
    }

    // Read detections
    std::vector<gvl::BoundingBox> detected_bboxes =
        gvl::readDetectionsFromFile(detection_path);

    // Read images and groundplane
    cv::Mat disparity = cv::imread(disparity_path, CV_LOAD_IMAGE_ANYDEPTH);
    if (disparity.data == NULL) {
      cout << "Error: " << disparity_path << std::endl;
      return 1;
    }
    cv::Mat image_2 = cv::imread(image_2_path);
    if (image_2.data == NULL) {
      cout << "Error: " << image_2_path << std::endl;
      return 0;
    }
    Eigen::Vector4d groundplane =
        gvl::readGroundplaneFromFile(groundplane_path);

    // // Generate scene-pointcloud
    cv::Mat vertices;
    auto pointcloud = std::make_shared<gvl::Pointcloud>();
    gvl::computeVerticesFromDisparity(disparity, K, baseline, vertices);
    gvl::computePointcloudFromVerticesAndColor(vertices, image_2, *pointcloud);

    // Filter and transform pointcloud
    auto pointcloud_filtered(std::make_shared<gvl::Pointcloud>());
    gvl::remove_road_plane_from_pointcloud(*pointcloud, groundplane,
                                           *pointcloud_filtered);

    //**********************************************************************************************
    // Initialize detections from all bounding boxes in this frame
    //**********************************************************************************************

    all_detections.resize(detected_bboxes.size());
    all_sizes.resize(detected_bboxes.size());
    all_scores.resize(detected_bboxes.size());
    all_valid.resize(detected_bboxes.size());
    for (unsigned int d = 0; d < detected_bboxes.size(); d++) {
      all_detections.at(d) = std::make_shared<gvl::Detection>();
      auto &bb = detected_bboxes.at(d);
      auto &det = all_detections.at(d);
      det->boundingbox = std::make_shared<gvl::BoundingBox>(bb);
      det->frame_id = -1;
      det->translation = Eigen::Vector3d(bb.x, bb.y, bb.z);
      det->rotation_y = bb.rotation_y;
      det->pose = gvl::computePoseFromRotTransScale(det->rotation_y,
                                                    det->translation, 1.0);
      det->z = Eigen::VectorXd::Zero(r, 1);
      det->shape = mean;
      gvl::extract_points_to_detection(image_2, *pointcloud_filtered, *det,
                                       det->pose, 2.5);
    }

    //**********************************************************************************************
    // Opimtization
    //**********************************************************************************************

    for (unsigned int i = 0; i < all_detections.size(); i++) {
      all_valid.at(i) = false;

      auto &det = all_detections[i];

      // Optimization - iterate alternativly until convergence
      double total_cost_curr = DBL_MAX / 2;
      double total_cost_prev = DBL_MAX;
      double delta = 0.1;
      int it = 0;
      while (total_cost_prev - total_cost_curr > delta && it < 1) {
        total_cost_prev = total_cost_curr;
        double pose_cost = gvl::optimize_pose(*det, false, &it);
        double shape_cost = gvl::optimize_shape(*det, false, &it);
        total_cost_curr = pose_cost + shape_cost;
      }

      if (!det->opt_successful) {
        cout << "\t\tDetection " << i << " unsuccessful" << endl;
        continue;
      }

      Eigen::Vector3d size = get_size(det->shape);
      all_sizes.at(i) = size;
      all_scores.at(i) = 1.0 - total_cost_curr;
      all_valid.at(i) = true;
    }

    int sum_valid = 0;
    for (unsigned int i = 0; i < all_valid.size(); i++)
      if (all_valid[i])
        sum_valid += 1;

    total_instances += all_valid.size();
    total_valid += sum_valid;

    // Write Detections to file
    writeDetectionsToFile(result_path, all_detections, all_sizes, all_scores,
                          all_valid);
  }
  cout << "Valid: " << total_valid << ", Total: " << total_instances << endl;

  return 0;
}
