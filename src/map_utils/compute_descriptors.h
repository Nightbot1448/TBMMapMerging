#ifndef COMPUTE_DESCRIPTORS_HPP
#define COMPUTE_DESCRIPTORS_HPP

#include <fstream>
#include <iterator>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"

#include <thread>
#include <future>
#include <cmath>

#include <set>

constexpr size_t THREAD_NUM = 3;

struct _thread_change_descriptor {
    const cv::Mat &img_first;
    const cv::Mat &img_second;

    std::vector<cv::KeyPoint> kp_first, kp_second;
    cv::Mat d_first_occ, d_first_emp, d_first_unk,
            d_second_occ, d_second_emp, d_second_unk;

    float count_of_features;
    float ratio_thresh;
    float scale_factor;
    _thread_change_descriptor( const cv::Mat &img1, const cv::Mat &img2, 
                    std::vector<cv::KeyPoint> kp_first, std::vector<cv::KeyPoint> kp_second,
                    cv::Mat d_first_occ, cv::Mat d_first_emp, cv::Mat d_first_unk,
                    cv::Mat d_second_occ, cv::Mat d_second_emp, cv::Mat d_second_unk,
                    float cof, float rt, float sf) : 
        img_first(img1), img_second(img2), kp_first(kp_first), kp_second(kp_second),
        d_first_occ(d_first_occ), d_first_emp(d_first_emp), d_first_unk(d_first_unk),
        d_second_occ(d_second_occ), d_second_emp(d_second_emp), d_second_unk(d_second_unk),
        count_of_features(cof), ratio_thresh(rt), scale_factor(sf){}
};

struct _thread_change_descriptor_return {
    cv::Mat new_d_first;
    cv::Mat new_d_second;
    std::vector<cv::DMatch> good_matches;
    _thread_change_descriptor_return( cv::Mat d1, cv::Mat d2, std::vector<cv::DMatch> gm):
             new_d_first(d1), new_d_second(d2), good_matches(gm){}
};

std::vector<cv::DMatch> get_good_matches(cv::Mat &first_descriptor, cv::Mat &second_descriptor, float ratio_thresh) 
{
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);

    std::vector<std::vector<cv::DMatch>> knn_matches;

    try{
        matcher->knnMatch( first_descriptor, second_descriptor, knn_matches, 2 );
    }
    catch(cv::Exception &e){
        std::cout << e.what() << std::endl;
    }

    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    return good_matches;
}

double Euclud_distance(cv::KeyPoint &f, cv::KeyPoint &s){
    return std::sqrt(std::pow(f.pt.x-s.pt.x,2)+std::pow(f.pt.y-s.pt.y,2));
}

_thread_change_descriptor_return new_descriptors_concatinate(_thread_change_descriptor info)
{
    cv::Mat new_d_first(info.count_of_features, 32*3, CV_8U), new_d_second(info.count_of_features, 32*3, CV_8U);

    for (size_t row_id = 0; row_id < info.count_of_features; row_id++) {
		uchar *dst_first = new_d_first.ptr(row_id);
        uchar *dst_second = new_d_second.ptr(row_id);
		std::memcpy( dst_first,     info.d_first_occ.ptr(row_id), 32);
        std::memcpy( dst_first+32,  info.d_first_emp.ptr(row_id), 32);
        std::memcpy( dst_first+64,  info.d_first_unk.ptr(row_id), 32);

        std::memcpy(dst_second,     info.d_second_occ.ptr(row_id), 32);
        std::memcpy(dst_second+32,  info.d_second_emp.ptr(row_id), 32);
        std::memcpy(dst_second+64,  info.d_second_unk.ptr(row_id), 32);
	}
    auto good_matches = get_good_matches( new_d_first, new_d_second, info.ratio_thresh );
    return _thread_change_descriptor_return(new_d_first, new_d_second, good_matches);
}

void parallel_new_descriptors_concatinate(_thread_change_descriptor info, std::promise<_thread_change_descriptor_return> *ret)
{
    ret->set_value(new_descriptors_concatinate(info));
}


void new_descriptors_land(_thread_change_descriptor info, std::promise<_thread_change_descriptor_return> *ret)
{
    cv::Mat new_d_first(info.count_of_features, 32, CV_8U), new_d_second(info.count_of_features, 32, CV_8U);

    for (size_t row_id = 0; row_id < info.count_of_features; row_id++) {
        size_t int_sz = sizeof(int);
        for(size_t col_id=0; col_id < 32/int_sz; col_id++){ 
            *new_d_first.ptr<int>(row_id, col_id) = (*info.d_first_occ.ptr<int>(row_id, col_id)) 
                                                  & (*info.d_first_emp.ptr<int>(row_id, col_id)) 
                                                  & (*info.d_first_unk.ptr<int>(row_id, col_id));
            *new_d_second.ptr<int>(row_id, col_id)= (*info.d_second_occ.ptr<int>(row_id, col_id)) 
                                                  & (*info.d_second_emp.ptr<int>(row_id, col_id)) 
                                                  & (*info.d_second_unk.ptr<int>(row_id, col_id));
        }
	}
    auto good_matches = get_good_matches( new_d_first, new_d_second, info.ratio_thresh );
    _thread_change_descriptor_return to_return(new_d_first, new_d_second, good_matches);
    ret->set_value(to_return);
}

void new_descriptors_lor(_thread_change_descriptor info, std::promise<_thread_change_descriptor_return> *ret)
{
    cv::Mat new_d_first(info.count_of_features, 32, CV_8U), new_d_second(info.count_of_features, 32, CV_8U);

    for (size_t row_id = 0; row_id < info.count_of_features; row_id++) {
        size_t int_sz = sizeof(int);
        for(size_t col_id=0; col_id < 32/int_sz; col_id++){ 
            *new_d_first.ptr<int>(row_id, col_id) = (*info.d_first_occ.ptr<int>(row_id, col_id)) 
                                                  | (*info.d_first_emp.ptr<int>(row_id, col_id)) 
                                                  | (*info.d_first_unk.ptr<int>(row_id, col_id));
            *new_d_second.ptr<int>(row_id, col_id)= (*info.d_second_occ.ptr<int>(row_id, col_id)) 
                                                  | (*info.d_second_emp.ptr<int>(row_id, col_id)) 
                                                  | (*info.d_second_unk.ptr<int>(row_id, col_id));
        }
	}
    auto good_matches = get_good_matches( new_d_first, new_d_second, info.ratio_thresh );
    _thread_change_descriptor_return to_return(new_d_first, new_d_second, good_matches);
    ret->set_value(to_return);
}

void new_descriptors_lxor(_thread_change_descriptor info, std::promise<_thread_change_descriptor_return> *ret)
{
    cv::Mat new_d_first(info.count_of_features, 32, CV_8U), new_d_second(info.count_of_features, 32, CV_8U);

    for (size_t row_id = 0; row_id < info.count_of_features; row_id++) {
        size_t int_sz = sizeof(int);
        for(size_t col_id=0; col_id < 32/int_sz; col_id++){ 
            *new_d_first.ptr<int>(row_id, col_id) = (*info.d_first_occ.ptr<int>(row_id, col_id)) 
                                                  ^ (*info.d_first_emp.ptr<int>(row_id, col_id)) 
                                                  ^ (*info.d_first_unk.ptr<int>(row_id, col_id));
            *new_d_second.ptr<int>(row_id, col_id)= (*info.d_second_occ.ptr<int>(row_id, col_id)) 
                                                  ^ (*info.d_second_emp.ptr<int>(row_id, col_id)) 
                                                  ^ (*info.d_second_unk.ptr<int>(row_id, col_id));
        }
	}
    auto good_matches = get_good_matches( new_d_first, new_d_second, info.ratio_thresh );
    _thread_change_descriptor_return to_return(new_d_first, new_d_second, good_matches);
    ret->set_value(to_return);
}

bool get_good_distance_vec_size(std::vector<cv::KeyPoint> &kp_first_prob, std::vector<cv::KeyPoint> &kp_second_prob,
                                std::vector<cv::KeyPoint> &kp_first_conc, std::vector<cv::KeyPoint> &kp_second_conc,
                                std::vector<cv::DMatch> &gm_prob, std::vector<cv::DMatch> &gm_conc, 
                                size_t &prob_good_dist_sz, size_t &conc_good_dist_sz){
    std::vector<double> distance_prob, distance_conc;
    try{
        for(size_t match_id = 0; match_id < gm_prob.size(); match_id++){
            cv::DMatch &match = gm_prob[match_id];
            distance_prob.push_back(Euclud_distance(kp_first_prob.at(match.queryIdx),kp_second_prob.at(match.trainIdx)));
        }
        for(size_t match_id = 0; match_id < gm_conc.size(); match_id++){
            cv::DMatch &match = gm_conc[match_id];
            distance_conc.push_back(Euclud_distance(kp_first_conc.at(match.queryIdx),kp_second_conc.at(match.trainIdx)));
        }
    }
    catch(std::out_of_range &ex){
        std::cout << ex.what() << std::endl;
        return false;
    }

    cv::Scalar mean_prob_s, stddev_prob_s, mean_conc_s, stddev_conc_s;
    if(distance_prob.size()) {
        try{
            cv::meanStdDev(distance_prob, mean_prob_s, stddev_prob_s);
            double  mean_prob = mean_prob_s[0],
                    stddev_prob = stddev_prob_s[0];
            double  prob_min = mean_prob - 1.5*stddev_prob,
                    prob_max = mean_prob + 1.5*stddev_prob;
            distance_prob.erase(std::remove_if(distance_prob.begin(), distance_prob.end(), 
                                [prob_min, prob_max](const double &dist){ return dist < prob_min || dist > prob_max;}),
                                distance_prob.end());
            prob_good_dist_sz = distance_prob.size();
        }
        catch(std::out_of_range &e){
            std::cout << e.what() << std::endl;
            return false;
        }
    }
    if(distance_conc.size()){
        try{
            cv::meanStdDev(distance_conc, mean_conc_s, stddev_conc_s);
            double  mean_conc = mean_conc_s[0],
                    stddev_conc = stddev_conc_s[0] > 5 ? stddev_conc_s[0]*2 : stddev_conc_s[0]*2;
            double  conc_min = mean_conc - stddev_conc,
                    conc_max = mean_conc + stddev_conc;
            distance_conc.erase( std::remove_if(distance_conc.begin(), distance_conc.end(), 
                                [conc_min, conc_max](const double &dist){ return dist < conc_min || dist > conc_max;}),
                                distance_conc.end());
            conc_good_dist_sz = distance_conc.size();
        }
        catch(std::out_of_range &e){
            std::cout << e.what() << std::endl;
            return false;
        }
    }
    return true;
}

#endif // COMPUTE_DESCRIPTORS_HPP