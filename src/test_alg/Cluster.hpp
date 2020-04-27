#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include <memory>

#include "../map_utils_headers/compareParametres.h"
#include "../map_utils_headers/compute_descriptors.h"

#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"

class Cluster {
public:
    Parameters parameters;
    std::unique_ptr<UnboundedPlainGridMap> first_map;
    std::unique_ptr<UnboundedPlainGridMap> second_map;
    std::array<cv::Mat,4> first_parts_maps, second_parts_maps;

    std::vector<cv::KeyPoint> kp_first_prob, kp_first_conc, 
                              kp_second_prob, kp_second_conc;
    cv::Mat d_first_prob, d_second_prob,
            d_first_occ, d_first_emp, d_first_unk,
            d_second_occ,d_second_emp,d_second_unk;

    std::vector<cv::DMatch> good_matches_prob, good_matches_conc;

public:
    Cluster(Parameters p);
    
    bool get_good_distance_vec_size(size_t &prob, size_t &conc);
    void detectAndCompute();

    void get_translation_of_keypoints(
        std::vector<cv::KeyPoint> &kp_first,
        std::vector<cv::KeyPoint> &kp_second, 
        std::vector<cv::DMatch> &good_matches
        );

    void action();
};

Cluster::Cluster(Parameters p) : parameters(p) {
    auto gmp = MapValues::gmp;
    first_map.reset(new UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp));
    second_map.reset(new UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp));
    std::ifstream in(parameters.first_filename);
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
    first_map->load_state(file_content);
    first_parts_maps = first_map->get_maps_grs_ofu();
    
    in = std::ifstream(parameters.second_filename);
    file_content = std::vector<char>((std::istreambuf_iterator<char>(in)),
                                        std::istreambuf_iterator<char>());
    second_map->load_state(file_content);
    second_parts_maps = second_map->get_maps_grs_ofu();
}

void Cluster::detectAndCompute(){
    cv::Ptr<cv::ORB> detector = cv::ORB::create( parameters.count_of_features, parameters.scale_factor );

    detector->detectAndCompute(first_parts_maps.at(0), cv::noArray(), 
            kp_first_prob, d_first_prob);
    
    kp_first_conc = kp_first_prob;
    detector->compute(first_parts_maps.at(1), kp_first_conc, d_first_occ);
    detector->compute(first_parts_maps.at(2), kp_first_conc, d_first_emp);
    detector->compute(first_parts_maps.at(3), kp_first_conc, d_first_unk);

    detector->detectAndCompute(second_parts_maps.at(0), cv::noArray(), 
            kp_second_prob, d_second_prob);

    kp_second_conc = kp_second_prob;
    detector->compute(second_parts_maps.at(1), kp_second_conc, d_second_occ);
    detector->compute(second_parts_maps.at(2), kp_second_conc, d_second_emp);
    detector->compute(second_parts_maps.at(3), kp_second_conc, d_second_unk);
}

void Cluster::action() {
    std::cout << "begin action" << std::endl;
    good_matches_prob = get_good_matches(d_first_prob, d_second_prob, parameters.ratio_thresh);
    
    cv::Mat concatinate_d_first = concatinateDescriptor(d_first_occ, d_first_emp, d_first_unk);
    cv::Mat concatinate_d_second = concatinateDescriptor(d_second_occ, d_second_emp, d_second_unk);

    good_matches_conc = get_good_matches(concatinate_d_first, concatinate_d_second, parameters.ratio_thresh);
    std::cout << "end action" << std::endl;
    // get_translation_of_keypoints(kp_first_conc, kp_second_conc, good_matches_conc);
}

void Cluster::get_translation_of_keypoints(
        std::vector<cv::KeyPoint> &kp_first,
        std::vector<cv::KeyPoint> &kp_second, 
        std::vector<cv::DMatch> &good_matches
        )
{
    try{
        for(size_t match_id = 0; match_id < good_matches.size(); match_id++) {
            cv::DMatch &match = good_matches[match_id];
        }
    }
    catch(std::out_of_range &ex){
        std::cout << ex.what() << std::endl;
    }
}

#endif //CLUSTER_HPP