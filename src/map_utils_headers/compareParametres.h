#ifndef COMPARE_PARAMETRES_H
#define COMPARE_PARAMETRES_H

#include <opencv2/core.hpp>

class Parameters{
public:
    Parameters() = default;
    Parameters(cv::CommandLineParser &parser){
        first_filename = parser.get<cv::String>("first_map");
        second_filename = parser.get<cv::String>("second_map");
        base_orb_tests_path = parser.get<cv::String>("base_orb_tests_path");
        base_desc_tests_path = parser.get<cv::String>("base_desc_tests_path");
        count_of_features = parser.get<size_t>("count_of_features");
        scale_factor = parser.get<float>("scale_factor");
        ratio_thresh = parser.get<float>("ratio_thresh");
        test_id = parser.get<int>("test_id");
        min_dist = parser.get<float>("min_dist");
        max_dist = parser.get<float>("max_dist");
    }
    std::string first_filename;
    std::string second_filename;
    std::string base_orb_tests_path;
    std::string base_desc_tests_path;
    size_t count_of_features;
    float scale_factor;
    float ratio_thresh;
    int test_id;
    float min_dist;
    float max_dist;
};

#endif