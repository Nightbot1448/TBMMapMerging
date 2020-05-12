#ifndef DESCRIPTORS_COMPORATOR_H
#define DESCRIPTORS_COMPORATOR_H

#include "compute_descriptors.h"
#include "compareParametres.h"
#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"
#include <opencv2/xfeatures2d.hpp>

class Algorythm {
public:
    std::vector<cv::KeyPoint> kp_first, kp_second;
    cv::Mat d_occ_first, d_emp_first, d_unk_first,
            d_occ_second, d_emp_second, d_unk_second;
    std::vector<cv::DMatch> good_matches;
    cv::Mat img_matches;
    size_t good_dist_sz;
};

class DescriptorsComparator{
private:
    bool good_distance_vector_size(Algorythm &alg);
public:
    DescriptorsComparator(Parameters p);
    void compareDescriptors();

    void detectAndCompute(cv::Ptr<cv::Feature2D> detector, Algorythm &alg);
    void detectAndComputeORB();
    void detectAndComputeBRISK();
    void detectAndComputeSIFT();
    void detectAndComputeSURF();

    void conc_and_good_matches(Algorythm &alg);
    

public:
    Parameters parameters;
    std::unique_ptr<UnboundedPlainGridMap> first_map;
    std::unique_ptr<UnboundedPlainGridMap> second_map;
    std::array<cv::Mat,4> first_parts_maps, second_parts_maps;

    Algorythm orb;
    Algorythm brisk;
    Algorythm sift;
    Algorythm surf;

};

DescriptorsComparator::DescriptorsComparator(Parameters p) : parameters(p){
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

void DescriptorsComparator::detectAndCompute(cv::Ptr<cv::Feature2D> detector, Algorythm &alg){
    detector->detect(first_parts_maps.at(0), alg.kp_first, cv::noArray());
    detector->compute(first_parts_maps.at(1), alg.kp_first, alg.d_occ_first);
    detector->compute(first_parts_maps.at(2), alg.kp_first, alg.d_emp_first);
    detector->compute(first_parts_maps.at(3), alg.kp_first, alg.d_unk_first);

    detector->detect(second_parts_maps.at(0), alg.kp_second, cv::noArray());
    detector->compute(second_parts_maps.at(1), alg.kp_second, alg.d_occ_second);
    detector->compute(second_parts_maps.at(2), alg.kp_second, alg.d_emp_second);
    detector->compute(second_parts_maps.at(3), alg.kp_second, alg.d_unk_second);    
}

void DescriptorsComparator::detectAndComputeORB() {
    cv::Ptr<cv::ORB> orb_detector = cv::ORB::create( parameters.count_of_features, parameters.scale_factor );
    detectAndCompute(orb_detector, orb);
}

void DescriptorsComparator::detectAndComputeBRISK() {
    cv::Ptr<cv::BRISK> brisk_detector = cv::BRISK::create(60, 4, 1.f);
    detectAndCompute(brisk_detector, brisk);
}

void DescriptorsComparator::detectAndComputeSIFT() {
    cv::Ptr<cv::xfeatures2d::SIFT> sift_detector = cv::xfeatures2d::SIFT::create();
    detectAndCompute(sift_detector, sift);
}

void DescriptorsComparator::detectAndComputeSURF() {
    cv::Ptr<cv::xfeatures2d::SURF> surf_detector = cv::xfeatures2d::SURF::create(400);
    detectAndCompute(surf_detector, surf);
}

void DescriptorsComparator::conc_and_good_matches(Algorythm &alg){
    cv::Mat conc_first = concatenateDescriptor(alg.d_occ_first, alg.d_emp_first,
                                               alg.d_unk_first);
    cv::Mat conc_second = concatenateDescriptor(alg.d_occ_second,
                                                alg.d_emp_second,
                                                alg.d_unk_second);
    alg.good_matches = get_good_matches(conc_first, conc_second, parameters.ratio_thresh);
    bool res = good_distance_vector_size(alg);
    if (res)
        drawMatches(first_parts_maps.at(0), alg.kp_first, second_parts_maps.at(0), alg.kp_second, 
                alg.good_matches, alg.img_matches, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
}

void DescriptorsComparator::compareDescriptors() {
    detectAndComputeORB();
    detectAndComputeBRISK();
    detectAndComputeSIFT();
    detectAndComputeSURF();

    conc_and_good_matches(orb);
    conc_and_good_matches(brisk);
    conc_and_good_matches(surf);
    conc_and_good_matches(sift);
    

    if(parameters.test_id>=0){
        std::string test_folder = parameters.base_desc_tests_path+std::to_string(parameters.test_id)+"/";
        cv::imwrite( test_folder + "orb.jpg", orb.img_matches);
        cv::imwrite( test_folder + "brisk.jpg", brisk.img_matches);
        cv::imwrite( test_folder + "sift.jpg", sift.img_matches);
        cv::imwrite( test_folder + "surf.jpg", surf.img_matches);
        std::ofstream out(test_folder+ "stat");
        out << parameters.first_filename << std::endl 
            << parameters.second_filename << std::endl 
            << orb.good_matches.size() << ' ' << brisk.good_matches.size() << ' ' 
                << sift.good_matches.size() << ' ' << surf.good_matches.size() << std::endl
            << orb.good_dist_sz << ' ' << brisk.good_dist_sz << ' ' 
                << sift.good_dist_sz << ' ' << surf.good_dist_sz << std::endl;
        out.close();
    }
}

bool DescriptorsComparator::good_distance_vector_size(Algorythm &alg){
    std::vector<double> distances;
    try{
        for(size_t match_id = 0; match_id < alg.good_matches.size(); match_id++){
            cv::DMatch &match = alg.good_matches[match_id];
            distances.push_back(Euclid_distance(alg.kp_first.at(match.queryIdx),
                                                alg.kp_second.at(
                                                        match.trainIdx)));
        }

        std::vector<double> mean, stdDev;
        cv::meanStdDev(distances, mean, stdDev);

        std::cout << mean.at(0) << ' ' << stdDev.at(0) << std::endl;

        float   min_dist = parameters.min_dist,
                max_dist = parameters.max_dist;

        
        distances.erase(std::remove_if(distances.begin(), distances.end(), 
                            [min_dist, max_dist](const double &dist){ return dist < min_dist || dist > max_dist;}),
                            distances.end());
        alg.good_dist_sz = distances.size();
        return true;
    }
    catch(std::out_of_range &e){
        std::cout << e.what() << std::endl;
        return false;
    }
    alg.good_dist_sz = 0;
    return true;
}

// bool DescriptorsComparator::get_good_distance_vec_size( size_t &orb_sz, size_t &brisk_sz, size_t &sift_sz, size_t &surf_sz){
//     bool orb_res = good_distance_vector_size(orb, orb_sz);
//     bool brisk_res = good_distance_vector_size(brisk, brisk_sz);
//     bool sift_res = good_distance_vector_size(sift, sift_sz);
//     bool surf_res = good_distance_vector_size(surf, surf_sz);
//     return orb_res && brisk_res && sift_res && surf_res;
// }

#endif