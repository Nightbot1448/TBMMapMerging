#ifndef TBM_MAP_MERGE_HPP
#define TBM_MAP_MERGE_HPP

#include <iostream>
#include <memory>
#include <cmath>
#include <string>

//#include "../map_utils_headers/compareParametres.h"
#include "../map_utils_headers/compute_descriptors.h"

#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>


class Algorithm_parameters {
public:
    Algorithm_parameters() = default;
    Algorithm_parameters(cv::CommandLineParser &parser){
        first_filename = parser.get<cv::String>("first_map");
        second_filename = parser.get<cv::String>("second_map");
        merged_map = parser.get<cv::String>("merged_map");
        count_of_features = parser.get<size_t>("count_of_features");
        scale_factor = parser.get<float>("scale_factor");
        ratio_thresh = parser.get<float>("ratio_thresh");
        show_maps = parser.get<bool>("show_maps");
    }
    std::string first_filename;
    std::string second_filename;
    std::string merged_map;
    size_t count_of_features;
    float scale_factor;
    float ratio_thresh;
    bool show_maps;
};

class TBM_map_merge {
public:
    Algorithm_parameters parameters;
    std::shared_ptr<UnboundedPlainGridMap> first_map;
    std::shared_ptr<UnboundedPlainGridMap> second_map;
    std::array<cv::Mat,4> first_parts_maps, second_parts_maps;

    std::vector<cv::KeyPoint> kp_first_conc, kp_second_conc;
    cv::Mat d_first_occ, d_first_emp, d_first_unk,
            d_second_occ,d_second_emp,d_second_unk;

    std::vector<cv::DMatch> good_matches_conc;

public:
    TBM_map_merge(const Algorithm_parameters& p);

    std::shared_ptr<UnboundedPlainGridMap> merge();

protected:
    void detectAndCompute();
    cv::Mat get_first_transform();


    double compute_average_distance(const std::vector<cv::Point2f> &first,
            const std::vector<cv::Point2f> &second, const cv::Mat &transform);
    
private:
    std::shared_ptr<UnboundedPlainGridMap> compute_merged(cv::Mat transform);
};

TBM_map_merge::TBM_map_merge(const Algorithm_parameters& p) : parameters(p) {
    auto gmp = MapValues::gmp;
    first_map = std::make_shared<UnboundedPlainGridMap>(std::make_shared<VinyDSCell>(), gmp);
    second_map = std::make_shared<UnboundedPlainGridMap>(std::make_shared<VinyDSCell>(), gmp);
    std::ifstream in(parameters.first_filename);
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
    first_map->load_state(file_content);
    first_map->crop_by_bounds();
    auto rotated = first_map->rotate(0.785398163);
    first_map.swap(rotated);
    first_parts_maps = first_map->get_maps_grs_ofu();
    in.close();
    in = std::ifstream(parameters.second_filename);
    file_content = std::vector<char>((std::istreambuf_iterator<char>(in)),
                                        std::istreambuf_iterator<char>());
    second_map->load_state(file_content);
    second_map->crop_by_bounds();
    second_parts_maps = second_map->get_maps_grs_ofu();
    in.close();
}

void TBM_map_merge::detectAndCompute(){
    cv::Ptr<cv::ORB> detector = cv::ORB::create( parameters.count_of_features, parameters.scale_factor );

    detector->detect(first_parts_maps.at(0), kp_first_conc);
    detector->compute(first_parts_maps.at(1), kp_first_conc, d_first_occ);
    detector->compute(first_parts_maps.at(2), kp_first_conc, d_first_emp);
    detector->compute(first_parts_maps.at(3), kp_first_conc, d_first_unk);

    detector->detect(second_parts_maps.at(0), kp_second_conc);
    detector->compute(second_parts_maps.at(1), kp_second_conc, d_second_occ);
    detector->compute(second_parts_maps.at(2), kp_second_conc, d_second_emp);
    detector->compute(second_parts_maps.at(3), kp_second_conc, d_second_unk);
}

cv::Mat TBM_map_merge::get_first_transform()
{
    std::vector<cv::Point2f> first, second;
    for (auto &match: good_matches_conc){
        const cv::Point2f &kp_first = kp_first_conc.at(match.queryIdx).pt;
        const cv::Point2f &kp_second = kp_second_conc.at(match.trainIdx).pt;
        first.push_back(kp_first);
        second.push_back(kp_second);
    }
    cv::Mat transform = cv::estimateAffine2D(first, second);
    compute_average_distance(first, second, transform);
    return transform;
}

std::shared_ptr<UnboundedPlainGridMap> TBM_map_merge::compute_merged(cv::Mat transform){
    DiscretePoint2D changed_sz;
    auto new_map = first_map->apply_transform(transform, changed_sz);
    auto merged_map_disjunctive = new_map->full_merge_disjunctive(second_map);
    return merged_map_disjunctive;
}

std::shared_ptr<UnboundedPlainGridMap> TBM_map_merge::merge() {
    detectAndCompute();
    
    cv::Mat concatenate_d_first = concatenateDescriptor(d_first_occ,
                                                        d_first_emp,
                                                        d_first_unk);
    cv::Mat concatenate_d_second = concatenateDescriptor(d_second_occ,
                                                         d_second_emp,
                                                         d_second_unk);

    good_matches_conc = get_good_matches(concatenate_d_first, concatenate_d_second, parameters.ratio_thresh);

    cv::Mat transform = get_first_transform();
    if (transform.dims == 2){
        auto merged = compute_merged(transform);
        if (parameters.show_maps){
            cv::Mat first_map_img = first_map->convert_to_grayscale_img();
            cv::Mat second_map_img = second_map->convert_to_grayscale_img();
            cv::Mat result_map_img = merged->convert_to_grayscale_img();
            int k=0;
            while (k != 27){
                cv::imshow("first map", first_map_img);
                cv::imshow("second map", second_map_img);
                cv::imshow("merged map", result_map_img);
                k = cv::waitKey();
            }
        }
        return merged;
    }
    return std::shared_ptr<UnboundedPlainGridMap>(nullptr);
}

double TBM_map_merge::compute_average_distance(const std::vector<cv::Point2f> &first,
                                               const std::vector<cv::Point2f> &second,
                                               const cv::Mat &transform){
    auto type_ = transform.type();
    cv::Mat mat(3,1, type_);
    cv::Mat second_;
    std::vector<double> distances;
    
    for(size_t i = 0; i < first.size(); ++i){
        std::vector<double> data_{first.at(i).x, first.at(i).y, 1};
        std::memcpy(mat.data, data_.data(), data_.size()*sizeof(double));
        cv::Mat transformed_point = transform * mat;
        cv::Mat(second.at(i)).convertTo(second_, type_);
        double norm = cv::norm(transformed_point, second_);
        distances.push_back(norm);

    }
    
    double avg_value = std::pow(
        std::accumulate(
            distances.begin(),
            distances.end(), 1.0, std::multiplies<double>()
        ), 1.0/distances.size()
    );
    auto minmax = std::minmax_element(distances.begin(), distances.end());
    std::cout << *(minmax.first) << ' ' << *(minmax.second) << ' ' << avg_value << std::endl;
    size_t file_base_sz = this->parameters.merged_map.find_last_of('.');
    std::string distances_filename = this->parameters.merged_map.substr(0, file_base_sz) + "_distances.txt";
    std::ofstream output_file(distances_filename);
    std::ostream_iterator<double> output_iterator(output_file, "\n");
    std::copy(distances.begin(), distances.end(), output_iterator);
    return avg_value;
}



#endif //TBM_MAP_MERGE_HPP