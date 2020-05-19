#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include <iostream>
#include <memory>
#include <cmath>

#include "../map_utils_headers/compareParametres.h"
#include "../map_utils_headers/compute_descriptors.h"

#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"

#include "icp.h"

#include <opencv2/video/tracking.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/registration/icp.h>

cv::Mat merge_img(const cv::Mat &first, const cv::Mat &second);

class Cluster {
public:
    Parameters parameters;
    std::shared_ptr<UnboundedPlainGridMap> first_map;
    std::shared_ptr<UnboundedPlainGridMap> second_map;
    std::array<cv::Mat,4> first_parts_maps, second_parts_maps;

//    std::vector<cv::KeyPoint> kp_first_prob, kp_second_prob,
    std::vector<cv::KeyPoint> kp_first_conc, kp_second_conc;
//    cv::Mat d_first_prob, d_second_prob,
    cv::Mat d_first_occ, d_first_emp, d_first_unk,
            d_second_occ,d_second_emp,d_second_unk;

    std::vector<cv::DMatch> good_matches_prob, good_matches_conc;

public:
    Cluster(const Parameters& p);
    
    bool get_good_distance_vec_size(size_t &prob, size_t &conc);
    void detectAndCompute();

    pcl::PointCloud<pcl::PointXYZ>::Ptr get_translation_of_keypoints(
        std::vector<cv::KeyPoint> &kp_first,
        std::vector<cv::KeyPoint> &kp_second, 
        std::vector<cv::DMatch> &good_matches
    );

    std::vector<pcl::PointIndices> get_keypoints_translations_clusters(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoints_translations
    );

    void print_clusters(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoints_translations,
        const std::vector<pcl::PointIndices>& cluster_indices
    );

    void check_need_rotation(std::vector<pcl::PointIndices> &indices);
    void check_need_rotation_v2(std::vector<pcl::PointIndices> &cluster_indices);
    void check_need_rotation_v3(std::vector<pcl::PointIndices> &cluster_indices);
    void action();

    void get_first_transform();
};

Cluster::Cluster(const Parameters& p) : parameters(p) {
    if(p.read_imgs) {
    //     first_parts_maps.at(0) = cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_1_grs.jpg");
    //     first_parts_maps.at(1) = cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_1_occ.jpg");
    //     first_parts_maps.at(2) = cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_1_emp.jpg");
    //     first_parts_maps.at(3) = cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_1_unk.jpg");
        
        auto gmp = MapValues::gmp;
        first_map = std::make_shared<UnboundedPlainGridMap>(std::make_shared<VinyDSCell>(), gmp);
        second_map = std::make_shared<UnboundedPlainGridMap>(std::make_shared<VinyDSCell>(), gmp);
        std::ifstream in(parameters.first_filename);
        std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
        first_map->load_state(file_content);
        first_parts_maps = first_map->get_maps_grs_ofu();

        second_parts_maps.at(0) =
                cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_2_grs_edited.jpg", cv::IMREAD_GRAYSCALE);
        second_parts_maps.at(1) =
                cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_2_occ_edited.jpg", cv::IMREAD_GRAYSCALE);
        second_parts_maps.at(2) =
                cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_2_emp_edited.jpg", cv::IMREAD_GRAYSCALE);
        second_parts_maps.at(3) =
                cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_2_unk_edited.jpg", cv::IMREAD_GRAYSCALE);
    }
    else {
        auto gmp = MapValues::gmp;
        first_map = std::make_shared<UnboundedPlainGridMap>(std::make_shared<VinyDSCell>(), gmp);
        second_map = std::make_shared<UnboundedPlainGridMap>(std::make_shared<VinyDSCell>(), gmp);
        std::ifstream in(parameters.first_filename);
        std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
        first_map->load_state(file_content);

        first_map->crop_by_bounds();
        first_parts_maps = first_map->get_maps_grs_ofu();
        in.close();
        in = std::ifstream(parameters.second_filename);
        file_content = std::vector<char>((std::istreambuf_iterator<char>(in)),
                                            std::istreambuf_iterator<char>());
        second_map->load_state(file_content);
        second_map->crop_by_bounds();
        std::cout << "first map: {" << first_map->width() << ' ' << first_map->height() << "}; " << first_map->origin() << std::endl;
        std::cout << "second map: {" << second_map->width() << ' ' << second_map->height() << "}; " << second_map->origin() << std::endl;
        second_parts_maps = second_map->get_maps_grs_ofu();
        in.close();
    }
}

void Cluster::detectAndCompute(){
    cv::Ptr<cv::ORB> detector = cv::ORB::create( parameters.count_of_features, parameters.scale_factor );

//    detector->detectAndCompute(first_parts_maps.at(0), cv::noArray(),
//            kp_first_prob, d_first_prob);

//    kp_first_conc = kp_first_prob;
    detector->detect(first_parts_maps.at(0), kp_first_conc);
    detector->compute(first_parts_maps.at(1), kp_first_conc, d_first_occ);
    detector->compute(first_parts_maps.at(2), kp_first_conc, d_first_emp);
    detector->compute(first_parts_maps.at(3), kp_first_conc, d_first_unk);

//    detector->detectAndCompute(second_parts_maps.at(0), cv::noArray(),
//            kp_second_prob, d_second_prob);
//    kp_second_conc = kp_second_prob;
    detector->detect(second_parts_maps.at(0), kp_second_conc);
    detector->compute(second_parts_maps.at(1), kp_second_conc, d_second_occ);
    detector->compute(second_parts_maps.at(2), kp_second_conc, d_second_emp);
    detector->compute(second_parts_maps.at(3), kp_second_conc, d_second_unk);
}

//void Cluster::get_first_transform()
//{
//    std::vector<cv::Point2f> first, second;
//    for (auto &match: good_matches_conc){
//        const cv::Point2f &kp_first = kp_first_conc.at(match.queryIdx).pt;
//        const cv::Point2f &kp_second = kp_second_conc.at(match.trainIdx).pt;
////                std::cout << kp_first << ": " << kp_second << std::endl;
//        first.push_back(kp_first);
//        second.push_back(kp_second);
//    }
//    cv::Mat transform = cv::estimateRigidTransform(first, second, true);
//    std::cout << "first transform is:" << std::endl << transform << std::endl;
////    transform.at<double>(1,2) = - transform.at<double>(1,2);
////    std::cout << "new first transform is:" << std::endl << transform << std::endl;
//    DiscretePoint2D changed_sz;
//    auto new_map = first_map->apply_transform(transform, changed_sz);
//    auto merged_map = first_map->full_merge(second_map);
//
//    cv::Mat new_map_grs = new_map->convert_to_grayscale_img();
//    cv::Mat sec_grs = second_map->convert_to_grayscale_img();
//    cv::Mat merged_imgs = merge_img(new_map_grs, sec_grs);
//    cv::Mat merged_maps = merged_map->convert_to_grayscale_img();
//    int k=0;
//    while(k!= 27){
//        cv::imshow("new map", new_map_grs);
//        cv::imshow("sec map", sec_grs);
//        cv::imshow("merged imgs", merged_imgs);
//        cv::imshow("merged maps", merged_maps);
//        k=cv::waitKey();
//    }
//}

void Cluster::action() {
    detectAndCompute();

    cv::Mat concatenate_d_first = concatenateDescriptor(d_first_occ,
                                                        d_first_emp,
                                                        d_first_unk);
    cv::Mat concatenate_d_second = concatenateDescriptor(d_second_occ,
                                                         d_second_emp,
                                                         d_second_unk);

    good_matches_conc = get_good_matches(concatenate_d_first, concatenate_d_second, parameters.ratio_thresh);
//    get_first_transform();
//     check_need_rotation(cluster_indices);
    // alg part
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_translations =
                get_translation_of_keypoints(kp_first_conc, kp_second_conc, good_matches_conc);
        std::vector<pcl::PointIndices> clusters_indices = get_keypoints_translations_clusters(keypoints_translations);
        check_need_rotation_v3(clusters_indices);
//        std::vector<cv::DMatch> matches_to_show;
//        for(const auto & cluster_indices : clusters_indices){
//            for (auto pit = cluster_indices.indices.begin (); pit != cluster_indices.indices.end(); ++pit){
//                std::cout << *pit << ' ';
//                if(cluster_indices.indices.size() > 1){ // dont't show clusters with only one vector translation
//                    matches_to_show.push_back(good_matches_conc.at(*pit));
//                }
//            }
//            std::cout << std::endl;
//        }
    }

//    if(parameters.test_id>=0){
//        print_clusters(keypoints_translations, cluster_indices);
//        cv::Mat img_matches_conc;
//
//        drawMatches(first_parts_maps.at(0), kp_first_conc, second_parts_maps.at(0), kp_second_conc,
//                        matches_to_show, img_matches_conc, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0),
//                        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//
//        // drawMatches(first_parts_maps.at(0), kp_first_conc, second_parts_maps.at(0), kp_second_conc,
//        //                 good_matches_conc, img_matches_conc, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0),
//        //                 std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//
//        while(true){
//            if (cv::waitKey() != 27)
//                cv::imshow( "conc", img_matches_conc);
//        }
//    }
}

void Cluster::print_clusters(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoints_translations,
    const std::vector<pcl::PointIndices>& cluster_indices
){
    std::cout << "count of clusters: " << cluster_indices.size() << std::endl;
    for (const auto & cluster_index : cluster_indices)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        for (int index : cluster_index.indices){
            cloud_cluster->points.push_back(keypoints_translations->points[index]);
        }
        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
    }
}

std::vector<pcl::PointIndices> Cluster::get_keypoints_translations_clusters(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoints_translations
){
    if (keypoints_translations->empty()){
        return std::vector<pcl::PointIndices>();
    }
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(keypoints_translations);
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(parameters.cluster_tolerance);
    ec.setMinClusterSize(1);
    ec.setMaxClusterSize(parameters.count_of_features);
    ec.setSearchMethod(tree);
    ec.setInputCloud(keypoints_translations);
    ec.extract(cluster_indices);
    return cluster_indices;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Cluster::get_translation_of_keypoints(
        std::vector<cv::KeyPoint> &kp_first,
        std::vector<cv::KeyPoint> &kp_second, 
        std::vector<cv::DMatch> &good_matches
        )
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_translations(new pcl::PointCloud<pcl::PointXYZ>);
    try{
        for(auto & match : good_matches) {
            keypoints_translations->push_back(
                pcl::PointXYZ(
                    kp_first[match.queryIdx].pt.x - kp_second[match.trainIdx].pt.x, 
                    kp_first[match.queryIdx].pt.y - kp_second[match.trainIdx].pt.y, 
                    0 ));
        }
    }
    catch(std::out_of_range &ex){
        std::cout << ex.what() << std::endl;
    }
    return keypoints_translations;
}

//void Cluster::check_need_rotation(std::vector<pcl::PointIndices> &cluster_indices){
//    std::cout << "check_need_rotation" << std::endl;
//    for(auto &cluster: cluster_indices) {
//        if(cluster.indices.size()>2) {
//            std::cout << "------------------------------" << std::endl;
//            pcl::PointCloud <pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud <pcl::PointXYZ>);
//            pcl::PointCloud <pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud <pcl::PointXYZ>);
//            for (auto &index: cluster.indices){
//                const cv::DMatch &match = good_matches_conc.at(index);
//                const cv::KeyPoint &kp_first = kp_first_conc.at(match.queryIdx);
//                const cv::KeyPoint &kp_second = kp_second_conc.at(match.trainIdx);
//                cloud_in->push_back(pcl::PointXYZ(kp_first.pt.x, kp_first.pt.y, 0));
//                cloud_out->push_back(pcl::PointXYZ(kp_second.pt.x, kp_second.pt.y, 0));
//            }
//
//            for(size_t i=0; i < cloud_in->size(); i++){
//                std::cout << '[' << i << "]: " << cloud_in->at(i) << "; " << cloud_out->at(i)
//                    << ": (" << cloud_in->at(i).x - cloud_out->at(i).x
//                    << ',' << cloud_in->at(i).y - cloud_out->at(i).y << ')' << std::endl;
//            }
//            pcl::IterativeClosestPoint < pcl::PointXYZ, pcl::PointXYZ > icp;
//            icp.setInputSource(cloud_in);
//            icp.setInputTarget(cloud_out);
//            icp.setEuclideanFitnessEpsilon(1e-3);
//            pcl::PointCloud < pcl::PointXYZ > final_;
//            icp.align(final_);
//            Eigen::Matrix4f finalTransformation = icp.getFinalTransformation();
//            // float angle = atan2(finalTransformation(1,0), finalTransformation(0,0));
//            cv::Mat transformMatrix(2,3,CV_32F);
//            transformMatrix.at<float>(0,0) = finalTransformation(0,0);
//            transformMatrix.at<float>(0,1) = finalTransformation(0,1);
//            transformMatrix.at<float>(1,0) = finalTransformation(1,0);
//            transformMatrix.at<float>(1,1) = finalTransformation(0,1);
//
//            // transformMatrix.at<float>(0,2) = finalTransformation(0,3);
//            // transformMatrix.at<float>(1,2) = finalTransformation(1,3);
//
//
//            // cv::Mat out;
//            // cv::warpAffine(first_parts_maps.at(0), out, transformMatrix,
//            //     cv::Size(first_parts_maps.at(0).rows, first_parts_maps.at(0).cols));
//
//
//            // while(true){
//            //     cv::imshow("original", first_parts_maps.at(0));
//            //     cv::imshow("warp", out);
//            //     if (cv::waitKey() == 27)
//            //         break;
//            // }
//
//            std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
//            std::cout << finalTransformation << std::endl;
//
//        }
//    }
//}

//void Cluster::check_need_rotation_v2(std::vector<pcl::PointIndices> &cluster_indices){
//    std::cout << "----------------------------------------------------" << std::endl;
//    std::cout << "----------------------------------------------------" << std::endl;
//    std::cout << "----------------------------------------------------" << std::endl;
//    std::cout << "check_need_rotation v2" << std::endl;
//    for(auto &cluster: cluster_indices) {
//        if(cluster.indices.size()>2) {
//            std::cout << "------------------------------" << std::endl;
//            Eigen::MatrixXd cloud_first = Eigen::MatrixXd::Zero(cluster.indices.size(), 3);
//            Eigen::MatrixXd cloud_second = Eigen::MatrixXd::Zero(cluster.indices.size(), 3);
//            auto &indices = cluster.indices;
//            for(size_t i=0; i < indices.size(); i++) {
//                const cv::DMatch &match = good_matches_conc.at(indices.at(i));
//                const cv::KeyPoint &kp_first = kp_first_conc.at(match.queryIdx);
//                const cv::KeyPoint &kp_second = kp_second_conc.at(match.trainIdx);
//                cloud_first.block<1,3>(i,0) = Eigen::Vector3d(kp_first.pt.x, kp_first.pt.y, 0);
//                cloud_second.block<1,3>(i,0) = Eigen::Vector3d(kp_second.pt.x, kp_second.pt.y, 0);
//                std::cout << '[' << i << "]: " << cloud_first.row(i) << "; " << cloud_second.row(i)
//                    << ": (" << cloud_first.row(i) - cloud_second.row(i) << ')' << std::endl;
//            }
//            // for (auto &index: cluster.indices){
//            //     const cv::DMatch &match = good_matches_conc.at(index);
//            //     const cv::KeyPoint &kp_first = kp_first_conc.at(match.queryIdx);
//            //     const cv::KeyPoint &kp_second = kp_second_conc.at(match.trainIdx);
//            //     cloud_first.block<1,3>(jj,0) = Eigen::Vector3d(kp_first.pt.x, kp_first.pt.y, 0);
//            //     cloud_second.block<1,3>(jj,0) = Eigen::Vector3d(kp_second.pt.x, kp_second.pt.y, 0);
//            //     std::cout << '[' << jj << "]: " << cloud_first.row(jj) << "; " << cloud_second.row(jj)
//            //         << ": (" << cloud_first.row(jj) - cloud_second.row(jj) << ')' << std::endl;
//            //     jj++;
//            // }
//            ICP_OUT icp_result = icp(cloud_second, cloud_first, 20,  0.000001);
//
//            std::cout << icp_result.trans << std::endl;
//        }
//    }
//}


void Cluster::check_need_rotation_v3(std::vector<pcl::PointIndices> &cluster_indices){
    std::vector<cv::Mat> imgs;
    cv::Mat img;
    DiscretePoint2D shift_map;
    for(auto &cluster: cluster_indices) {
        if(cluster.indices.size()>2) {
            std::cout << "------------------------------" << std::endl;
            std::vector<cv::Point2f> first, second;
            for (auto &index: cluster.indices){
                const cv::DMatch &match = good_matches_conc.at(index);
                const cv::Point2f &kp_first = kp_first_conc.at(match.queryIdx).pt;
                const cv::Point2f &kp_second = kp_second_conc.at(match.trainIdx).pt;
//                std::cout << kp_first << ": " << kp_second << std::endl;
                first.push_back(kp_first);
                second.push_back(kp_second);
            }
            cv::Mat transform = cv::estimateRigidTransform(first, second, true);
//            cv::Mat transform = cv::Mat(2,3, CV_64F);
//            std::vector<double> data_{0.6,0.4,0,-0.4,0.6,0};
//            std::vector<double> data_{
//                1, 0, 100,
//                0, 1, -57};
//            std::vector<double> data_{
//                0.9, -0.44, 0,
//                0.44, 0.9,  0};
//            std::memcpy(transform.data, data_.data(), data_.size()*sizeof(double));

//            std::vector<double> main_transform{
//                    0.999428014664722, -0.01302791791974173, 100.5138984921331,
//                    0.01129921694475718, 0.9902744729629283, 57.0081216044651};
//            std::memcpy(transform.data, main_transform.data(), main_transform.size()*sizeof(double));

            if(transform.dims) {
                auto transformed = first_map->apply_transform(transform, shift_map);
                std::cout << "shift_map: " << shift_map << std::endl;
                img = transformed->convert_to_grayscale_img();
                imgs.push_back(img);
                imgs.push_back(second_map->convert_to_grayscale_img());
                auto merged_maps = transformed->full_merge(second_map, shift_map);
                imgs.push_back(merged_maps->convert_to_grayscale_img());
//                cv::Size max_sz(std::max(img.rows, second_parts_maps.at(0).rows),
//                                std::max(img.cols, second_parts_maps.at(0).cols));
//  y inversed

                cv::Mat out(img.rows, img.cols, first_parts_maps.at(0).type());
//                transform.at<double>(0,2) += shift_map.x;
                cv::warpAffine(first_parts_maps.at(0), out, transform, out.size());
//                cv::Mat result_merging = merge_img(out ,second_parts_maps.at(0));
//                imgs.push_back(out);
            }
//            std::cout << "transform: "  << std::endl << transform << std::endl;
        }
    }
    int k=0;
    std::cout << "imgs size: " << imgs.size() << std::endl;
//    auto second_shifted3 = second_map->shift(DiscretePoint2D(0, shift_map.y/2));
//    cv::Mat tmp3 = second_shifted3->convert_to_grayscale_img();
//    cv::Mat result_merging = merge_img(img, tmp3);
//    imgs.push_back(result_merging);
    while(k != 27) {
        for (size_t id = 0; id < imgs.size(); id++) {
            cv::imshow(std::string("img") + std::to_string(id), imgs.at(id));
        }
//        cv::imshow("first", img);
//        cv::imshow("second01", tmp3);
        k = cv::waitKey();
    }
}

cv::Mat merge_img(const cv::Mat &first, const cv::Mat &second) {
    double alpha = 0.5;
    double beta = 1 - alpha;

    cv::Size max_sz(std::max(first.rows, second.rows),
                    std::max(first.cols, second.cols));
    cv::Mat out1 = cv::Mat::zeros(max_sz.width,max_sz.height,first.type());
    cv::Mat out2 = cv::Mat::zeros(max_sz.width,max_sz.height,second.type());
    first.copyTo(out1(cv::Rect(0,0, first.cols, first.rows)));
    second.copyTo(out2(cv::Rect(0,0, second.cols, second.rows)));
    cv::Mat result_merging;
    cv::addWeighted(out1, alpha, out2, beta, 0.0, result_merging);
    return result_merging;
}
#endif //CLUSTER_HPP