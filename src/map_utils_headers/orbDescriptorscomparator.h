#ifndef ORB_DESCRIPTORS_COMPORATOR_H
#define ORB_DESCRIPTORS_COMPORATOR_H

#include <cstdlib>
#include "compute_descriptors.h"
#include "compareParametres.h"
#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"

// class my_function_object {
// private:
//     float threshold;
// public:
//     my_function_object(float threshold) :
//             threshold(threshold) {
//     }
// 
//     bool meets_criteria(float value) const {
//         return value <= threshold;
//     }
// 
//     float operator()(float a, float b) const {
//         return meets_criteria(b) ? a + b : a;
//     }
// };

class OrbDescriptorsComparator{
private:
    // bool remove_outlier(std::vector<double> &distances, size_t &good_dist_sz);
    // bool good_distance_vec_size(std::vector<cv::KeyPoint> &kp_first,std::vector<cv::KeyPoint> &kp_second, 
    //             std::vector<cv::DMatch> &good_matches, size_t &good_dist_sz);
    size_t get_correct_match_count(
        std::vector<cv::KeyPoint> &kp_first, std::vector<cv::KeyPoint> &kp_second,
        std::vector<cv::DMatch> &good_matches, std::vector<double> &distances
    );

    void write_results_to_file(std::vector<size_t> correct, std::vector<std::vector<double>> distances);

    // cv::Mat get_distances_between_matched_keypoints(
    //     std::vector<cv::KeyPoint> &kp_first, std::vector<cv::KeyPoint> &kp_second,
    //     std::vector<cv::DMatch> &good_matches, cv::Mat &transform
    // );
   
public:
    OrbDescriptorsComparator(Parameters p);
    void compareDescriptors();
    
    // bool get_good_distance_vec_size(size_t &prob, size_t &conc, size_t &land, size_t &lor, size_t &lxor);
    void detectAndCompute();
    

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

    std::vector<cv::DMatch> good_matches_prob, good_matches_conc, good_matches_land, good_matches_lor, good_matches_lxor;

};

OrbDescriptorsComparator::OrbDescriptorsComparator(Parameters p) : parameters(p){
    // std::cout << "start constructor" << std::endl;
    auto gmp = MapValues::gmp;
    first_map.reset(new UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp));
    second_map.reset(new UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp));
    std::ifstream in(parameters.first_filename);
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
    // std::cout << "1: start load state" << std::endl;
    first_map->load_state(file_content);
    // std::cout << "1: end load state, getting imgs" << std::endl;
    first_parts_maps = first_map->get_maps_grs_ofu();
    
    in = std::ifstream(parameters.second_filename);
    file_content = std::vector<char>((std::istreambuf_iterator<char>(in)),
                                        std::istreambuf_iterator<char>());
    second_map->load_state(file_content);
    second_parts_maps = second_map->get_maps_grs_ofu();
}

void OrbDescriptorsComparator::detectAndCompute(){
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

void OrbDescriptorsComparator::write_results_to_file(std::vector<size_t> correct, std::vector<std::vector<double>> distances){
    
    size_t first_base_backslash_1 = parameters.first_filename.find_last_of('/');
    size_t first_base_backslash_0 = parameters.first_filename.find_last_of('/', first_base_backslash_1-1);
    std::string first_filename = parameters.first_filename.substr(first_base_backslash_0+1, first_base_backslash_1 - first_base_backslash_0 - 1);
    size_t second_base_backslash_1 = parameters.second_filename.find_last_of('/');
    size_t second_base_backslash_0 = parameters.second_filename.find_last_of('/', second_base_backslash_1-1);
    std::string second_filename = parameters.second_filename.substr(second_base_backslash_0+1, second_base_backslash_1 - second_base_backslash_0 - 1);
    
    std::string out_file = std::string("/home/dmo/Documents/Study/diplom_paper/data/compareORB/") + std::to_string(parameters.out_file_id) + ".txt";
    std::ofstream out_stream;
    out_stream.open(out_file);
    
    out_stream << first_filename << std::endl;
    out_stream << second_filename << std::endl;
    out_stream << parameters.count_of_features << std::endl;
    out_stream << parameters.ratio_thresh << std::endl;
    
    out_stream << "prob: " << correct.at(0) << "/" <<  good_matches_prob.size() << std::endl;
    for(auto el: distances.at(0))
        out_stream << el << ' ';
    out_stream << std::endl;

    out_stream << "conc: " << correct.at(1) << "/" <<  good_matches_conc.size() << std::endl;
    for(auto el: distances.at(1))
        out_stream << el << ' ';
    out_stream << std::endl;

    out_stream << "land: " << correct.at(2) << "/" <<  good_matches_land.size() << std::endl;
    for(auto el: distances.at(2))
        out_stream << el << ' ';
    out_stream << std::endl;

    out_stream << "lor: "  << correct.at(3) << "/" <<  good_matches_lor.size()  << std::endl;
    for(auto el: distances.at(3))
        out_stream << el << ' ';
    out_stream << std::endl;
    
    out_stream << "lxor: " << correct.at(4) << "/" <<  good_matches_lxor.size() << std::endl;
    for(auto el: distances.at(4))
        out_stream << el << ' ';
    out_stream << std::endl;
    out_stream.close();
}

void OrbDescriptorsComparator::compareDescriptors() {
    detectAndCompute();
    
    good_matches_prob = get_good_matches(d_first_prob, d_second_prob, parameters.ratio_thresh);
    
    cv::Mat concatinate_d_first = concatenateDescriptor(d_first_occ,
                                                        d_first_emp,
                                                        d_first_unk);
    cv::Mat concatinate_d_second = concatenateDescriptor(d_second_occ,
                                                         d_second_emp,
                                                         d_second_unk);
    
    cv::Mat land_d_first = landDescriptor(d_first_occ, d_first_emp, d_first_unk);
    cv::Mat land_d_second = landDescriptor(d_second_occ, d_second_emp, d_second_unk);

    cv::Mat lor_d_first = lorDescriptor(d_first_occ, d_first_emp, d_first_unk);
    cv::Mat lor_d_second = lorDescriptor(d_second_occ, d_second_emp, d_second_unk);

    cv::Mat lxor_d_first = lxorDescriptor(d_first_occ, d_first_emp, d_first_unk);
    cv::Mat lxor_d_second = lxorDescriptor(d_second_occ, d_second_emp, d_second_unk);
    
    good_matches_conc = get_good_matches(concatinate_d_first, concatinate_d_second, parameters.ratio_thresh);
    good_matches_land = get_good_matches(land_d_first, land_d_second, parameters.ratio_thresh);
    good_matches_lor =  get_good_matches(lor_d_first, lor_d_second, parameters.ratio_thresh);
    good_matches_lxor = get_good_matches(lxor_d_first, lxor_d_second, parameters.ratio_thresh);

    std::vector<double> distances_prob, distances_conc, distances_land, distances_lor, distances_lxor;
    
    size_t correct_count_prob = good_matches_prob.size() ? get_correct_match_count(kp_first_conc, kp_second_conc, good_matches_prob, distances_prob) : 0;
    size_t correct_count_conc = good_matches_conc.size() ? get_correct_match_count(kp_first_conc, kp_second_conc, good_matches_conc, distances_conc) : 0;
    size_t correct_count_land = good_matches_land.size() ? get_correct_match_count(kp_first_conc, kp_second_conc, good_matches_land, distances_land) : 0;
    size_t correct_count_lor = good_matches_lor.size() ? get_correct_match_count(kp_first_conc, kp_second_conc, good_matches_lor, distances_lor) : 0;
    size_t correct_count_lxor = good_matches_lxor.size() ? get_correct_match_count(kp_first_conc, kp_second_conc, good_matches_lxor, distances_lxor) : 0;

    write_results_to_file(std::vector<size_t>{correct_count_prob, correct_count_conc, correct_count_land, correct_count_lor, correct_count_lxor},
    std::vector<std::vector<double>>{distances_prob, distances_conc, distances_land, distances_lor, distances_lxor}
        );

    {
    // size_t prob_good_dist_sz, conc_good_dist_sz;//, land_good_dist_sz, lor_good_dist_sz, lxor_good_dist_sz;
    // if(get_good_distance_vec_size(prob_good_dist_sz, conc_good_dist_sz))//, 
            // land_good_dist_sz, lor_good_dist_sz,lxor_good_dist_sz)
    // {
        //-- Draw matches
        // cv::Mat img_matches_prob, img_matches_conc, img_matches_land, img_matches_lor, img_matches_lxor;
        // drawMatches(first_parts_maps.at(0), kp_first_prob, second_parts_maps.at(0), kp_second_prob, 
        //             good_matches_prob, img_matches_prob, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
        //             std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        // drawMatches(first_parts_maps.at(0), kp_first_conc, second_parts_maps.at(0), kp_second_conc, 
        //             good_matches_conc, img_matches_conc, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
        //             std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        
        // drawMatches(first_parts_maps.at(0), kp_first_conc, second_parts_maps.at(0), kp_second_conc, 
        //             good_matches_land, img_matches_land, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
        //             std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        // drawMatches(first_parts_maps.at(0), kp_first_conc, second_parts_maps.at(0), kp_second_conc, 
        //             good_matches_lor, img_matches_lor, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
        //             std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        // drawMatches(first_parts_maps.at(0), kp_first_conc, second_parts_maps.at(0), kp_second_conc, 
        //             good_matches_lxor, img_matches_lxor, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
        //             std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        // if(parameters.test_id>=0){
        //     std::string test_folder = parameters.base_orb_tests_path+std::to_string(parameters.test_id)+"/";
        //     cv::imwrite( test_folder + "prob.jpg", img_matches_prob);
        //     cv::imwrite( test_folder + "conc.jpg", img_matches_conc);
        //     // cv::imwrite( test_folder + "land.jpg", img_matches_land);
        //     // cv::imwrite( test_folder + "lor.jpg",  img_matches_lor);
        //     // cv::imwrite( test_folder + "lxor.jpg", img_matches_lxor);
        //     std::ofstream out(test_folder+ "stat");
        //     out << parameters.first_filename << std::endl 
        //         << parameters.second_filename << std::endl 
        //         << parameters.count_of_features << std::endl 
        //         << parameters.ratio_thresh << std::endl
        //         << good_matches_prob.size() << ' ' << good_matches_conc.size() << std::endl// << ' ' << good_matches_land.size()
        //             // << ' ' << good_matches_lor.size() << ' ' << good_matches_lxor.size() << std::endl
        //         << prob_good_dist_sz << ' ' << conc_good_dist_sz << std::endl;// << ' ' << land_good_dist_sz
        //             // << ' ' << lor_good_dist_sz << ' ' << lxor_good_dist_sz << std::endl;
        //     out.close();

        // }
    // }
}

}

// bool OrbDescriptorsComparator::remove_outlier(std::vector<double> &distances, size_t &good_dist_sz){
//     if(distances.size()) {
//         try{
//             std::vector<double> meanC, stdDevC;
//             cv::meanStdDev(distances,meanC, stdDevC);
//             double mean = meanC.at(0);
//             double stdDev = stdDevC.at(0);
//             std::cout << mean << ' ' << stdDev << std::endl
//                     << "Distance size before filter 3*sigma: " << distances.size() << std::endl;
//             if(stdDev>5){
//                 float   min_dist = mean - 3*stdDev,
//                         max_dist = mean + 3*stdDev;
//                 distances.erase(std::remove_if(distances.begin(), distances.end(), 
//                                 [min_dist, max_dist](const double &dist){ return dist < min_dist || dist > max_dist;}),
//                                 distances.end());    
//             }
//             std::cout << "Distances: " << std::endl;
//             for( auto &distance: distances)
//                 std::cout << distance << ' ';
//             std::cout << std::endl;
// 
//             std::cout << "Distance size after filter 3*sigma: " << distances.size() << std::endl;
// 
//             float   min_dist = parameters.min_dist,
//                     max_dist = parameters.max_dist;
//             distances.erase(std::remove_if(distances.begin(), distances.end(), 
//                                 [min_dist, max_dist](const double &dist){ return dist < min_dist || dist > max_dist;}),
//                                 distances.end());
//             good_dist_sz = distances.size();
//             std::cout << "good dist sz: " << good_dist_sz << std::endl;
//             return true;
//         }
//         catch(std::out_of_range &e){
//             std::cout << e.what() << std::endl;
//             return false;
//         }
//     }
//     else{
//         std::cout << "distance size is zero" << std::endl;
//     }
//     good_dist_sz = 0;
//     return true;
// }

// bool OrbDescriptorsComparator::good_distance_vec_size(
//         std::vector<cv::KeyPoint> &kp_first,
//         std::vector<cv::KeyPoint> &kp_second, 
//         std::vector<cv::DMatch> &good_matches, 
//         size_t &good_dist_sz)
// {
//     std::vector<double> distance;
//     try{
//         for(size_t match_id = 0; match_id < good_matches.size(); match_id++){
//             cv::DMatch &match = good_matches[match_id];
//             distance.push_back(Euclid_distance(kp_first.at(match.queryIdx),
//                                                kp_second.at(match.trainIdx)));
//         }
//     }
//     catch(std::out_of_range &ex){
//         std::cout << ex.what() << std::endl;
//         return false;
//     }
// 
//     return remove_outlier(distance, good_dist_sz);
// }

// bool OrbDescriptorsComparator::get_good_distance_vec_size(
//         size_t &prob_good_dist_sz, 
//         size_t &conc_good_dist_sz 
//         size_t &land_good_dist_sz,
//         size_t &lor_good_dist_sz,
//         size_t &lxor_good_dist_sz
//         )
// {
//     bool prob = good_distance_vec_size( kp_first_prob, kp_second_prob, good_matches_prob, prob_good_dist_sz );
//     bool conc = good_distance_vec_size( kp_first_conc, kp_second_conc, good_matches_conc, conc_good_dist_sz );
//     bool land = good_distance_vec_size( kp_first_conc, kp_second_conc, good_matches_land, land_good_dist_sz );
//     bool lor  = good_distance_vec_size( kp_first_conc, kp_second_conc, good_matches_lor,  lor_good_dist_sz  );
//     bool lxor = good_distance_vec_size( kp_first_conc, kp_second_conc, good_matches_lxor, lxor_good_dist_sz );
//     return prob && conc && land && lor && lxor;
// }


size_t OrbDescriptorsComparator::get_correct_match_count(
    std::vector<cv::KeyPoint> &kp_first, std::vector<cv::KeyPoint> &kp_second,
    std::vector<cv::DMatch> &good_matches, std::vector<double> &distances
){
    std::vector<cv::Point2f> first, second, first_transformed;
    for (auto &match: good_matches){
        const cv::Point2f &kp1 = kp_first.at(match.queryIdx).pt;
        const cv::Point2f &kp2 = kp_second.at(match.trainIdx).pt;
        first.push_back(kp1);
        second.push_back(kp2);
    }
    cv::Mat transform = cv::estimateAffine2D(first, second);
    if (transform.dims == 2){
        cv::transform(first, first_transformed, transform);

        double good_dist_count = 0;
        for (size_t i=0; i<first_transformed.size(); i++){
            double distance = Euclid_distance(first_transformed.at(i), second.at(i));
            if (distance <= parameters.max_dist){
                good_dist_count++;
            }
            distances.push_back(distance);
        }
        return good_dist_count;
    }
    return 0;
    
}

#endif //ORB_DESCRIPTORS_COMPORATOR_H