#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include <iostream>
#include <memory>

#include "../map_utils_headers/compareParametres.h"
#include "../map_utils_headers/compute_descriptors.h"

#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"

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
        pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_translations
    );

    void print_clusters(
        pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_translations, 
        std::vector<pcl::PointIndices> cluster_indices
    );

    void check_need_rotation(std::vector<pcl::PointIndices> &indices);

    void action();
};

Cluster::Cluster(Parameters p) : parameters(p) {
    if(p.read_imgs) {
    //     first_parts_maps.at(0) = cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_1_grs.jpg");
    //     first_parts_maps.at(1) = cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_1_occ.jpg");
    //     first_parts_maps.at(2) = cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_1_emp.jpg");
    //     first_parts_maps.at(3) = cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_1_unk.jpg");
        
        auto gmp = MapValues::gmp;
        first_map.reset(new UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp));
        second_map.reset(new UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp));
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
    std::vector<pcl::PointIndices> cluster_indices = get_keypoints_translations_clusters(keypoints_translations);
    
    check_need_rotation(cluster_indices);

    std::vector<cv::DMatch> matches_to_show;
    for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++){
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit){
                std::cout << *pit << ' ';
                if(it->indices.size() > 1){ // dont't show clusters with only one vector translation
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
        while(true){
            cv::imshow( "conc", img_matches_conc);
            if (cv::waitKey() == 27)
                break;
        }
    }
}

void Cluster::print_clusters(
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_translations, 
    std::vector<pcl::PointIndices> cluster_indices
){
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
        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
    }
}

std::vector<pcl::PointIndices> Cluster::get_keypoints_translations_clusters(
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_translations
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
        for(size_t match_id = 0; match_id < good_matches.size(); match_id++) {
            cv::DMatch &match = good_matches[match_id];
            keypoints_translations->push_back(
                pcl::PointXYZ(
                    kp_first[match.queryIdx].pt.x - kp_second[match.trainIdx].pt.x, 
                    kp_first[match.queryIdx].pt.y - kp_second[match.trainIdx].pt.y, 
                    0 ));
        }
        if(parameters.test_id>=0){
            std::cout << "keypoints translation: " << keypoints_translations->size() << std::endl;
            for(auto it = keypoints_translations->begin(); it != keypoints_translations->end(); it++){
                std::cout << '[' << (it - keypoints_translations->begin()) << "]: " << it->x << ' ' << it->y << ' ' << it->z << "; dist: " << sqrtf(it->x*it->x + it->y*it->y) << std::endl;
            }
        }
    }
    catch(std::out_of_range &ex){
        std::cout << ex.what() << std::endl;
    }
    return keypoints_translations;
}

void Cluster::check_need_rotation(std::vector<pcl::PointIndices> &cluster_indices){
    std::cout << "check_need_rotation" << std::endl;
    if(cluster_indices.size()>1) {
        pcl::PointCloud <pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud <pcl::PointXYZ>);
        pcl::PointCloud <pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud <pcl::PointXYZ>);
        for (auto &index: cluster_indices.at(0).indices){
            const cv::DMatch &match = good_matches_conc.at(index);
            const cv::KeyPoint &kp_first = kp_first_conc.at(match.queryIdx);
            const cv::KeyPoint &kp_second = kp_second_conc.at(match.trainIdx);
            cloud_in->push_back(pcl::PointXYZ(kp_first.pt.x, kp_first.pt.y, 0));
            cloud_out->push_back(pcl::PointXYZ(kp_second.pt.x, kp_second.pt.y, 0));
        }

        for(int i=0; i < cloud_in->size(); i++){
            std::cout << cloud_in->at(i) << "; " << cloud_out->at(i) 
                << ": (" << cloud_in->at(i).x - cloud_out->at(i).x 
                << ',' << cloud_in->at(i).y - cloud_out->at(i).y << ')' << std::endl;
        }
        pcl::IterativeClosestPoint < pcl::PointXYZ, pcl::PointXYZ > icp;
        icp.setInputSource(cloud_in);
        icp.setInputTarget(cloud_out);
        icp.setEuclideanFitnessEpsilon(1e-3);
        pcl::PointCloud < pcl::PointXYZ > final_;
        icp.align(final_);
        std::cout << "final_" << std::endl;
        for(int i=0; i < final_.size(); i++){
            std::cout << final_.at(i) << std::endl;
        }

        std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
        std::cout << icp.getFinalTransformation() << std::endl;
    }
}

#endif //CLUSTER_HPP