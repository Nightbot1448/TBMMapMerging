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
};

Cluster::Cluster(Parameters p) : parameters(p) {
    if(p.read_imgs) {
    //     first_parts_maps.at(0) = cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_1_grs.jpg");
    //     first_parts_maps.at(1) = cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_1_occ.jpg");
    //     first_parts_maps.at(2) = cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_1_emp.jpg");
    //     first_parts_maps.at(3) = cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_1_unk.jpg");
        
        auto gmp = MapValues::gmp;
        first_map = std::make_unique<UnboundedPlainGridMap>(std::make_shared<VinyDSCell>(), gmp);
        second_map = std::make_unique<UnboundedPlainGridMap>(std::make_shared<VinyDSCell>(), gmp);
        std::ifstream in(parameters.first_filename);
        std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
        first_map->load_state(file_content);
        first_parts_maps = first_map->get_maps_grs_ofu();

        second_parts_maps.at(0) = cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_2_grs_edited.jpg");
        second_parts_maps.at(1) = cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_2_occ_edited.jpg");
        second_parts_maps.at(2) = cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_2_emp_edited.jpg");
        second_parts_maps.at(3) = cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_2_unk_edited.jpg");
    }
    else {
        auto gmp = MapValues::gmp;
        first_map = std::make_unique<UnboundedPlainGridMap>(std::make_shared<VinyDSCell>(), gmp);
        second_map = std::make_unique<UnboundedPlainGridMap>(std::make_shared<VinyDSCell>(), gmp);
        std::ifstream in(parameters.first_filename);
        std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
        first_map->load_state(file_content);
        first_parts_maps = first_map->get_maps_grs_ofu();
        in.close();
        in = std::ifstream(parameters.second_filename);
        file_content = std::vector<char>((std::istreambuf_iterator<char>(in)),
                                            std::istreambuf_iterator<char>());
        second_map->load_state(file_content);
        second_parts_maps = second_map->get_maps_grs_ofu();
        in.close();
    }
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
    detectAndCompute();
    good_matches_prob = get_good_matches(d_first_prob, d_second_prob, parameters.ratio_thresh);
    
    cv::Mat concatinate_d_first = concatenateDescriptor(d_first_occ,
                                                        d_first_emp,
                                                        d_first_unk);
    cv::Mat concatinate_d_second = concatenateDescriptor(d_second_occ,
                                                         d_second_emp,
                                                         d_second_unk);

    good_matches_conc = get_good_matches(concatinate_d_first, concatinate_d_second, parameters.ratio_thresh);
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_translations =
        get_translation_of_keypoints(kp_first_conc, kp_second_conc, good_matches_conc);    
    std::vector<pcl::PointIndices> cluster_indices = get_keypoints_translations_clusters(keypoints_translations);
    
    // check_need_rotation(cluster_indices);
    check_need_rotation_v3(cluster_indices);
    std::vector<cv::DMatch> matches_to_show;
    for(const auto & cluster_indice : cluster_indices){
        for (auto pit = cluster_indice.indices.begin (); pit != cluster_indice.indices.end (); ++pit){
                std::cout << *pit << ' ';
                if(cluster_indice.indices.size() > 1){ // dont't show clusters with only one vector translation
                    matches_to_show.push_back(good_matches_conc.at(*pit));
                }
            }
            std::cout << std::endl;
    }

    if(parameters.test_id>=0){
        print_clusters(keypoints_translations, cluster_indices);
        cv::Mat img_matches_conc;

        drawMatches(first_parts_maps.at(0), kp_first_conc, second_parts_maps.at(0), kp_second_conc, 
                        matches_to_show, img_matches_conc, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
                        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        // drawMatches(first_parts_maps.at(0), kp_first_conc, second_parts_maps.at(0), kp_second_conc, 
        //                 good_matches_conc, img_matches_conc, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
        //                 std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        
        // double alpha = 0.5; 
        // double beta = 1 - alpha;
        // cv::Mat result_merging;
        // cv::Mat out1 = cv::Mat::zeros(1020,1440,first_parts_maps.at(0).type());
        // cv::Mat out2 = cv::Mat::zeros(1020,1440,first_parts_maps.at(0).type());        
        // first_parts_maps.at(0).copyTo(out1(cv::Rect(200,14,1000,1000)));
        // auto roi = cv::Rect(0,0,1440,1000);
        // second_parts_maps.at(0).copyTo(out2(roi));
        // cv::addWeighted(out1, alpha, out2, beta, 0.0, result_merging);
        
        while(true){
            cv::imshow( "conc", img_matches_conc);
            // cv::imshow("merged", result_merging);
            if (cv::waitKey() == 27)
                break;
        }
    }
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

void Cluster::check_need_rotation(std::vector<pcl::PointIndices> &cluster_indices){
    std::cout << "check_need_rotation" << std::endl;
    for(auto &cluster: cluster_indices) {
        if(cluster.indices.size()>2) {
            std::cout << "------------------------------" << std::endl;
            pcl::PointCloud <pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud <pcl::PointXYZ>);
            pcl::PointCloud <pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud <pcl::PointXYZ>);
            for (auto &index: cluster.indices){
                const cv::DMatch &match = good_matches_conc.at(index);
                const cv::KeyPoint &kp_first = kp_first_conc.at(match.queryIdx);
                const cv::KeyPoint &kp_second = kp_second_conc.at(match.trainIdx);
                cloud_in->push_back(pcl::PointXYZ(kp_first.pt.x, kp_first.pt.y, 0));
                cloud_out->push_back(pcl::PointXYZ(kp_second.pt.x, kp_second.pt.y, 0));
            }

            for(size_t i=0; i < cloud_in->size(); i++){
                std::cout << '[' << i << "]: " << cloud_in->at(i) << "; " << cloud_out->at(i) 
                    << ": (" << cloud_in->at(i).x - cloud_out->at(i).x 
                    << ',' << cloud_in->at(i).y - cloud_out->at(i).y << ')' << std::endl;
            }
            pcl::IterativeClosestPoint < pcl::PointXYZ, pcl::PointXYZ > icp;
            icp.setInputSource(cloud_in);
            icp.setInputTarget(cloud_out);
            icp.setEuclideanFitnessEpsilon(1e-3);
            pcl::PointCloud < pcl::PointXYZ > final_;
            icp.align(final_);
            Eigen::Matrix4f finalTransformation = icp.getFinalTransformation();
            // float angle = atan2(finalTransformation(1,0), finalTransformation(0,0));
            cv::Mat transformMatrix(2,3,CV_32F);
            transformMatrix.at<float>(0,0) = finalTransformation(0,0);
            transformMatrix.at<float>(0,1) = finalTransformation(0,1);
            transformMatrix.at<float>(1,0) = finalTransformation(1,0);
            transformMatrix.at<float>(1,1) = finalTransformation(0,1);
            
            // transformMatrix.at<float>(0,2) = finalTransformation(0,3);
            // transformMatrix.at<float>(1,2) = finalTransformation(1,3);
            

            // cv::Mat out;
            // cv::warpAffine(first_parts_maps.at(0), out, transformMatrix, 
            //     cv::Size(first_parts_maps.at(0).rows, first_parts_maps.at(0).cols));
            
            
            // while(true){
            //     cv::imshow("original", first_parts_maps.at(0));
            //     cv::imshow("warp", out);
            //     if (cv::waitKey() == 27)
            //         break;
            // }

            std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
            std::cout << finalTransformation << std::endl;

        }
    }
}

void Cluster::check_need_rotation_v2(std::vector<pcl::PointIndices> &cluster_indices){
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "check_need_rotation v2" << std::endl;
    for(auto &cluster: cluster_indices) {
        if(cluster.indices.size()>2) {
            std::cout << "------------------------------" << std::endl;
            Eigen::MatrixXd cloud_first = Eigen::MatrixXd::Zero(cluster.indices.size(), 3);
            Eigen::MatrixXd cloud_second = Eigen::MatrixXd::Zero(cluster.indices.size(), 3);
            auto &indices = cluster.indices;
            for(size_t i=0; i < indices.size(); i++) {
                const cv::DMatch &match = good_matches_conc.at(indices.at(i));
                const cv::KeyPoint &kp_first = kp_first_conc.at(match.queryIdx);
                const cv::KeyPoint &kp_second = kp_second_conc.at(match.trainIdx);
                cloud_first.block<1,3>(i,0) = Eigen::Vector3d(kp_first.pt.x, kp_first.pt.y, 0);
                cloud_second.block<1,3>(i,0) = Eigen::Vector3d(kp_second.pt.x, kp_second.pt.y, 0);
                std::cout << '[' << i << "]: " << cloud_first.row(i) << "; " << cloud_second.row(i) 
                    << ": (" << cloud_first.row(i) - cloud_second.row(i) << ')' << std::endl;
            }
            // for (auto &index: cluster.indices){
            //     const cv::DMatch &match = good_matches_conc.at(index);
            //     const cv::KeyPoint &kp_first = kp_first_conc.at(match.queryIdx);
            //     const cv::KeyPoint &kp_second = kp_second_conc.at(match.trainIdx);
            //     cloud_first.block<1,3>(jj,0) = Eigen::Vector3d(kp_first.pt.x, kp_first.pt.y, 0);
            //     cloud_second.block<1,3>(jj,0) = Eigen::Vector3d(kp_second.pt.x, kp_second.pt.y, 0);
            //     std::cout << '[' << jj << "]: " << cloud_first.row(jj) << "; " << cloud_second.row(jj) 
            //         << ": (" << cloud_first.row(jj) - cloud_second.row(jj) << ')' << std::endl;
            //     jj++;
            // }
            ICP_OUT icp_result = icp(cloud_second, cloud_first, 20,  0.000001);

            std::cout << icp_result.trans << std::endl;
        }
    }
}


void Cluster::check_need_rotation_v3(std::vector<pcl::PointIndices> &cluster_indices){
    std::vector<cv::Mat> imgs;
    double alpha = 0.5;
    double beta = 1 - alpha;
    for(auto &cluster: cluster_indices) {
        if(cluster.indices.size()>2) {
            std::cout << "------------------------------" << std::endl;
            std::vector<cv::Point2f> first, second;
            for (auto &index: cluster.indices){
                const cv::DMatch &match = good_matches_conc.at(index);
                const cv::Point2f &kp_first = kp_first_conc.at(match.queryIdx).pt;
                const cv::Point2f &kp_second = kp_second_conc.at(match.trainIdx).pt;
                std::cout << kp_first << ": " << kp_second << std::endl;
                first.push_back(kp_first);
                second.push_back(kp_second);
            }
            cv::Mat transform = cv::estimateRigidTransform(first, second, true);

            if(transform.size[0] != 0) {
                cv::Size max_sz(std::max(first_parts_maps.at(0).rows, second_parts_maps.at(0).rows), std::max(first_parts_maps.at(0).cols, second_parts_maps.at(0).cols));
                std::cout << "max: " << max_sz << "; " << max_sz.width << ' ' << max_sz.height << std::endl;
                cv::Mat out(max_sz.width, max_sz.height, first_parts_maps.at(0).type());
                cv::warpAffine(first_parts_maps.at(0), out, transform, out.size());
                cv::Mat out1 = cv::Mat::zeros(max_sz.width,max_sz.height,first_parts_maps.at(0).type());
                cv::Mat out2 = cv::Mat::zeros(max_sz.width,max_sz.height,first_parts_maps.at(0).type());
                std::cout << "sizes: " <<  out1.size << ' ' << out2.size << std::endl;
                out.copyTo(out1(cv::Rect(0,0,out.cols,out.rows)));
                auto roi = cv::Rect(0,0,second_parts_maps.at(0).cols,second_parts_maps.at(0).rows);
                second_parts_maps.at(0).copyTo(out2(roi));
                cv::Mat result_merging;
                std::cout << "sizes1: " <<  out1.size << ' ' << out2.size << std::endl;
                cv::addWeighted(out1, alpha, out2, beta, 0.0, result_merging);
                imgs.push_back(result_merging);
            }
            std::cout << transform << std::endl;
        }
    }
    int k=0;
    std::cout << "imgs size: " << imgs.size() << std::endl;
    while(k != 27) {
        for (size_t id = 0; id < imgs.size(); id++) {
            cv::imshow(std::string("img") + std::to_string(id), imgs.at(id));
        }
        k = cv::waitKey();
    }
}
#endif //CLUSTER_HPP