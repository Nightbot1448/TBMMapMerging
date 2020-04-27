#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include <iostream>
#include <memory>

#include "../map_utils_headers/compareParametres.h"
#include "../map_utils_headers/compute_descriptors.h"

#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

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

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> get_keypoints_translations_clusters(
        pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_translations
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
    detectAndCompute();
    good_matches_prob = get_good_matches(d_first_prob, d_second_prob, parameters.ratio_thresh);
    
    cv::Mat concatinate_d_first = concatinateDescriptor(d_first_occ, d_first_emp, d_first_unk);
    cv::Mat concatinate_d_second = concatinateDescriptor(d_second_occ, d_second_emp, d_second_unk);

    good_matches_conc = get_good_matches(concatinate_d_first, concatinate_d_second, parameters.ratio_thresh);
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_translations =
        get_translation_of_keypoints(kp_first_conc, kp_second_conc, good_matches_conc);    
    get_keypoints_translations_clusters(keypoints_translations);

    cv::Mat img_matches_conc;
    if(parameters.test_id>=0){
        drawMatches(first_parts_maps.at(0), kp_first_conc, second_parts_maps.at(0), kp_second_conc, 
                        good_matches_conc, img_matches_conc, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
                        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        cv::imshow( "conc", img_matches_conc);
        cv::waitKey();
    }
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> Cluster::get_keypoints_translations_clusters(pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_translations){
    std::cout << "action" << std::endl;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(keypoints_translations);
    std::cout << "tree" << std::endl;
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(parameters.cluster_tolerance);
    ec.setMinClusterSize(1);
    ec.setMaxClusterSize(parameters.count_of_features);
    ec.setSearchMethod(tree);
    ec.setInputCloud(keypoints_translations);
    ec.extract (cluster_indices);

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters_of_vectors;

    std::cout << "count of clusters: " << cluster_indices.size() << std::endl;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit){
            cloud_cluster->points.push_back(keypoints_translations->points[*pit]); //*
        }
        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        clusters_of_vectors.push_back(cloud_cluster);
        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
    }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Cluster::get_translation_of_keypoints(
        std::vector<cv::KeyPoint> &kp_first,
        std::vector<cv::KeyPoint> &kp_second, 
        std::vector<cv::DMatch> &good_matches
        )
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_translations(new pcl::PointCloud<pcl::PointXYZ>);
    try{
        std::cout << "call get_translation_of_keypoints" << std::endl 
            << "good matches size: " << good_matches.size() << std::endl;
        for(size_t match_id = 0; match_id < good_matches.size(); match_id++) {
            cv::DMatch &match = good_matches[match_id];
            keypoints_translations->push_back(
                pcl::PointXYZ(
                    kp_first[match.queryIdx].pt.x - kp_second[match.trainIdx].pt.x, 
                    kp_first[match.queryIdx].pt.x - kp_second[match.trainIdx].pt.x, 
                    0 ));
        }
        std::cout << "res size: " << keypoints_translations->size() << std::endl;
        for(auto &  pt: *keypoints_translations){
            std::cout << pt.x << ' ' << pt.y << ' ' << pt.z << "; dist: " << sqrtf(pt.x*pt.x + pt.y*pt.y) << std::endl;
        }
    }
    catch(std::out_of_range &ex){
        std::cout << ex.what() << std::endl;
    }
    return keypoints_translations;
}

#endif //CLUSTER_HPP