#ifndef COMPARE_PARAMETRES_H
#define COMPARE_PARAMETRES_H

#include <opencv2/core.hpp>

class Parameters{
public:
    Parameters() = default;
    // Parameters(cv::CommandLineParser &parser){
    //     first_filename = parser.get<cv::String>("first_map");
    //     second_filename = parser.get<cv::String>("second_map");
    //     base_orb_tests_path = parser.get<cv::String>("base_orb_tests_path");
    //     base_desc_tests_path = parser.get<cv::String>("base_desc_tests_path");
    //     count_of_features = parser.get<size_t>("count_of_features");
    //     scale_factor = parser.get<float>("scale_factor");
    //     ratio_thresh = parser.get<float>("ratio_thresh");
    //     test_id = parser.get<int>("test_id");
    //     min_dist = parser.get<float>("min_dist");
    //     max_dist = parser.get<float>("max_dist");
    //     cluster_tolerance = parser.get<float>("cluster_tolerance");
    //     read_imgs = parser.get<bool>("read_imgs");
    // }

    Parameters(std::string first_map, std::string second_map, 
        int count_of_features=100, float scale_factor=1.2f, float ratio_thresh=0.7f,
        float max_dist=5.0f, size_t test_id=0
    ){
        this->first_filename = first_map;
        this->second_filename = second_map;
        this->count_of_features = count_of_features;
        this->scale_factor = scale_factor;
        this->ratio_thresh = ratio_thresh;
        this->max_dist = max_dist;
        this->out_file_id = test_id;
    }

    std::string first_filename;
    std::string second_filename;
    size_t count_of_features;
    float scale_factor;
    float ratio_thresh;
    float max_dist;
    size_t out_file_id;
};

#endif