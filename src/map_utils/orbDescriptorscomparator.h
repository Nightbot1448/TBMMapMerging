#ifndef ORB_DESCRIPTORS_COMPORATOR_H
#define ORB_DESCRIPTORS_COMPORATOR_H

#include <cstdlib>
#include "compute_descriptors.h"
#include "compareParametres.h"
#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"

class OrbDescriptorsComparator{
private:
    bool good_distance_vector_size(std::vector<double> &distances, size_t &good_dist_sz);
public:
    OrbDescriptorsComparator(Parameters p);
    void compareDescriptors();
    
    bool get_good_distance_vec_size(size_t &prob, size_t &conc, size_t &land, size_t &lor, size_t &lxor);
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

void OrbDescriptorsComparator::compareDescriptors() {
    detectAndCompute();
    
    good_matches_prob = get_good_matches(d_first_prob, d_second_prob, parameters.ratio_thresh);
    
    cv::Mat concatinate_d_first = concatinateDescriptor(d_first_occ, d_first_emp, d_first_unk);
    cv::Mat concatinate_d_second = concatinateDescriptor(d_second_occ, d_second_emp, d_second_unk);
    
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


    size_t prob_good_dist_sz, conc_good_dist_sz, land_good_dist_sz, lor_good_dist_sz, lxor_good_dist_sz;
    if(get_good_distance_vec_size(prob_good_dist_sz, conc_good_dist_sz, 
            land_good_dist_sz, lor_good_dist_sz,lxor_good_dist_sz))
    {
        //-- Draw matches
        cv::Mat img_matches_prob, img_matches_conc, img_matches_land, img_matches_lor, img_matches_lxor;
        drawMatches(first_parts_maps.at(0), kp_first_prob, second_parts_maps.at(0), kp_second_prob, 
                    good_matches_prob, img_matches_prob, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        drawMatches(first_parts_maps.at(0), kp_first_conc, second_parts_maps.at(0), kp_second_conc, 
                    good_matches_conc, img_matches_conc, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        
        drawMatches(first_parts_maps.at(0), kp_first_conc, second_parts_maps.at(0), kp_second_conc, 
                    good_matches_land, img_matches_land, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        drawMatches(first_parts_maps.at(0), kp_first_conc, second_parts_maps.at(0), kp_second_conc, 
                    good_matches_lor, img_matches_lor, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        drawMatches(first_parts_maps.at(0), kp_first_conc, second_parts_maps.at(0), kp_second_conc, 
                    good_matches_lxor, img_matches_lxor, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        if(parameters.test_id>=0){
            std::string test_folder = parameters.base_orb_tests_path+std::to_string(parameters.test_id)+"/";
            cv::imwrite( test_folder + "prob.jpg", img_matches_prob);
            cv::imwrite( test_folder + "conc.jpg", img_matches_conc);
            cv::imwrite( test_folder + "land.jpg", img_matches_land);
            cv::imwrite( test_folder + "lor.jpg",  img_matches_lor);
            cv::imwrite( test_folder + "lxor.jpg", img_matches_lxor);
            std::ofstream out(test_folder+ "stat");
            out << parameters.first_filename << std::endl 
                << parameters.second_filename << std::endl 
                << parameters.count_of_features << std::endl 
                << parameters.ratio_thresh << std::endl
                << good_matches_prob.size() << ' ' << good_matches_conc.size() << ' ' << good_matches_land.size()
                    << ' ' << good_matches_lor.size() << ' ' << good_matches_lxor.size() << std::endl
                << prob_good_dist_sz << ' ' << conc_good_dist_sz << ' ' << land_good_dist_sz
                    << ' ' << lor_good_dist_sz << ' ' << lxor_good_dist_sz << std::endl;
            out.close();

        }
    }
}

bool OrbDescriptorsComparator::good_distance_vector_size(std::vector<double> &distances, size_t &good_dist_sz){
    if(distances.size()) {
        try{
            float   min_dist = parameters.min_dist,
                    max_dist = parameters.max_dist;
            distances.erase(std::remove_if(distances.begin(), distances.end(), 
                                [min_dist, max_dist](const double &dist){ return dist < min_dist || dist > max_dist;}),
                                distances.end());
            good_dist_sz = distances.size();
            return true;
        }
        catch(std::out_of_range &e){
            std::cout << e.what() << std::endl;
            return false;
        }
    }
    good_dist_sz = 0;
    return true;
}

bool OrbDescriptorsComparator::get_good_distance_vec_size(
        size_t &prob_good_dist_sz, 
        size_t &conc_good_dist_sz, 
        size_t &land_good_dist_sz,
        size_t &lor_good_dist_sz,
        size_t &lxor_good_dist_sz)
{
    std::vector<double> distance_prob, distance_conc, distance_land,distance_lor,distance_lxor;
    try{
        for(size_t match_id = 0; match_id < good_matches_prob.size(); match_id++){
            cv::DMatch &match = good_matches_prob[match_id];
            distance_prob.push_back(Euclud_distance(kp_first_prob.at(match.queryIdx),kp_second_prob.at(match.trainIdx)));
        }
        for(size_t match_id = 0; match_id < good_matches_conc.size(); match_id++){
            cv::DMatch &match = good_matches_conc[match_id];
            distance_conc.push_back(Euclud_distance(kp_first_conc.at(match.queryIdx),kp_second_conc.at(match.trainIdx)));
        }
        for(size_t match_id = 0; match_id < good_matches_land.size(); match_id++){
            cv::DMatch &match = good_matches_land[match_id];
            distance_land.push_back(Euclud_distance(kp_first_conc.at(match.queryIdx),kp_second_conc.at(match.trainIdx)));
        }
        for(size_t match_id = 0; match_id < good_matches_lor.size(); match_id++){
            cv::DMatch &match = good_matches_lor[match_id];
            distance_lor.push_back(Euclud_distance(kp_first_conc.at(match.queryIdx),kp_second_conc.at(match.trainIdx)));
        }
        for(size_t match_id = 0; match_id < good_matches_lxor.size(); match_id++){
            cv::DMatch &match = good_matches_lxor[match_id];
            distance_lxor.push_back(Euclud_distance(kp_first_conc.at(match.queryIdx),kp_second_conc.at(match.trainIdx)));
        }
    }
    catch(std::out_of_range &ex){
        std::cout << ex.what() << std::endl;
        return false;
    }
    
    bool prob_res = good_distance_vector_size(distance_prob, prob_good_dist_sz);
    bool conc_res = good_distance_vector_size(distance_conc, conc_good_dist_sz);
    bool land_res = good_distance_vector_size(distance_land, land_good_dist_sz);
    bool lor_res  = good_distance_vector_size(distance_lor,  lor_good_dist_sz);
    bool lxor_res = good_distance_vector_size(distance_lxor, lxor_good_dist_sz);
    
    return prob_res && conc_res && land_res && lor_res && lxor_res;
}

#endif //ORB_DESCRIPTORS_COMPORATOR_H