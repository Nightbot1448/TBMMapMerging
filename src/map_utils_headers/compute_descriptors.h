#ifndef COMPUTE_DESCRIPTORS_H
#define COMPUTE_DESCRIPTORS_H

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
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
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

double Euclid_distance(cv::Point2f &f, cv::Point2f &s){
    return std::sqrt(std::pow(f.x-s.x,2)+std::pow(f.y-s.y,2));
}

double Euclid_distance(cv::KeyPoint &f, cv::KeyPoint &s){
    return std::sqrt(std::pow(f.pt.x-s.pt.x,2)+std::pow(f.pt.y-s.pt.y,2));
}

cv::Mat concatenateDescriptor(cv::Mat &d_occ, cv::Mat &d_emp, cv::Mat &d_unk)
{
    size_t rows = static_cast<size_t>(d_occ.size[0]);
    size_t cols = static_cast<size_t>(d_occ.size[1]);
    cv::Mat new_descriptor(rows, cols*3, CV_8U);
    for (size_t row_id = 0; row_id < rows; row_id++) {
		uchar *dst = new_descriptor.ptr(row_id);
        std::memcpy( dst,       d_occ.ptr(row_id), cols);
        std::memcpy( dst+cols,  d_emp.ptr(row_id), cols);
        std::memcpy( dst+cols*2,d_unk.ptr(row_id), cols);
	}
    return new_descriptor;
}

_thread_change_descriptor_return new_descriptors_concatinate(_thread_change_descriptor info)
{
    cv::Mat new_d_first = concatenateDescriptor(info.d_first_occ,
                                                info.d_first_emp,
                                                info.d_first_unk);
    cv::Mat new_d_second = concatenateDescriptor(info.d_second_occ,
                                                 info.d_second_emp,
                                                 info.d_second_unk);
    auto good_matches = get_good_matches( new_d_first, new_d_second, info.ratio_thresh );
    return _thread_change_descriptor_return(new_d_first, new_d_second, good_matches);
}

void parallel_new_descriptors_concatinate(_thread_change_descriptor info, std::promise<_thread_change_descriptor_return> *ret)
{
    ret->set_value(new_descriptors_concatinate(info));
}


cv::Mat landDescriptor(cv::Mat &d_occ, cv::Mat &d_emp, cv::Mat &d_unk){
    size_t rows = static_cast<size_t>(d_occ.size[0]);
    size_t cols = static_cast<size_t>(d_occ.size[1]);
    cv::Mat new_descriptor(rows, cols, CV_8U);
    size_t int_sz = sizeof(int);
    for (size_t row_id = 0; row_id < rows; row_id++) {
        for(size_t col_id=0; col_id < cols/int_sz; col_id++){ 
            *new_descriptor.ptr<int>(row_id, col_id) = (*d_occ.ptr<int>(row_id, col_id))
                                                     & (*d_emp.ptr<int>(row_id, col_id))
                                                     & (*d_unk.ptr<int>(row_id, col_id));
        }
	}
    return new_descriptor;
}

void new_descriptors_land(_thread_change_descriptor info, std::promise<_thread_change_descriptor_return> *ret)
{
    cv::Mat new_d_first = landDescriptor(info.d_first_occ, info.d_first_emp, info.d_first_unk);
    cv::Mat new_d_second = landDescriptor(info.d_second_occ, info.d_second_emp, info.d_second_unk);
    auto good_matches = get_good_matches( new_d_first, new_d_second, info.ratio_thresh );
    _thread_change_descriptor_return to_return(new_d_first, new_d_second, good_matches);
    ret->set_value(to_return);
}

cv::Mat lorDescriptor(cv::Mat &d_occ, cv::Mat &d_emp, cv::Mat &d_unk)
{
    size_t rows = static_cast<size_t>(d_occ.size[0]);
    size_t cols = static_cast<size_t>(d_occ.size[1]);
    cv::Mat new_descriptor(rows, cols, CV_8U);
    size_t int_sz = sizeof(int);
    for (size_t row_id = 0; row_id < rows; row_id++) {
        for(size_t col_id=0; col_id < cols/int_sz; col_id++){ 
            *new_descriptor.ptr<int>(row_id, col_id) = (*d_occ.ptr<int>(row_id, col_id))
                                                     | (*d_emp.ptr<int>(row_id, col_id))
                                                     | (*d_unk.ptr<int>(row_id, col_id));
        }
	}
    return new_descriptor;
}

void new_descriptors_lor(_thread_change_descriptor info, std::promise<_thread_change_descriptor_return> *ret)
{
    cv::Mat new_d_first  = lorDescriptor(info.d_first_occ, info.d_first_emp, info.d_first_unk);
    cv::Mat new_d_second = lorDescriptor(info.d_second_occ, info.d_second_emp, info.d_second_unk);   
    auto good_matches = get_good_matches( new_d_first, new_d_second, info.ratio_thresh );
    _thread_change_descriptor_return to_return(new_d_first, new_d_second, good_matches);
    ret->set_value(to_return);
}

cv::Mat lxorDescriptor(cv::Mat &d_occ, cv::Mat &d_emp, cv::Mat &d_unk)
{
    size_t rows = static_cast<size_t>(d_occ.size[0]);
    size_t cols = static_cast<size_t>(d_occ.size[1]);

    cv::Mat new_descriptor(rows, cols, CV_8U);
    size_t int_sz = sizeof(int);

    for (size_t row_id = 0; row_id < rows; row_id++) {
        for(size_t col_id=0; col_id < cols/int_sz; col_id++){ 
            *new_descriptor.ptr<int>(row_id, col_id) = (*d_occ.ptr<int>(row_id, col_id))
                                                     ^ (*d_emp.ptr<int>(row_id, col_id))
                                                     ^ (*d_unk.ptr<int>(row_id, col_id));
        }
	}
    return new_descriptor;
}

void new_descriptors_lxor(_thread_change_descriptor info, std::promise<_thread_change_descriptor_return> *ret)
{
    cv::Mat new_d_first  = lxorDescriptor(info.d_first_occ, info.d_first_emp, info.d_first_unk);
    cv::Mat new_d_second = lxorDescriptor(info.d_second_occ, info.d_second_emp, info.d_second_unk);   
    auto good_matches = get_good_matches( new_d_first, new_d_second, info.ratio_thresh );
    _thread_change_descriptor_return to_return(new_d_first, new_d_second, good_matches);
    ret->set_value(to_return);
}

#endif // COMPUTE_DESCRIPTORS_HPP