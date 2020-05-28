#ifndef TBM_MAP_MERGE_HPP
#define TBM_MAP_MERGE_HPP

#include <iostream>
#include <memory>
#include <cmath>
#include <string>

//#include "../map_utils_headers/compareParametres.h"
#include "../map_utils_headers/compute_descriptors.h"

#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

class Algorithm_parameters {
public:
    Algorithm_parameters() = default;
    Algorithm_parameters(cv::CommandLineParser &parser){
        first_filename = parser.get<cv::String>("first_map");
        second_filename = parser.get<cv::String>("second_map");
        merged_map = parser.get<cv::String>("merged_map");
        count_of_features = parser.get<size_t>("count_of_features");
        scale_factor = parser.get<float>("scale_factor");
        ratio_thresh = parser.get<float>("ratio_thresh");
        show_maps = parser.get<bool>("show_maps");
//        cluster_tolerance = parser.get<float>("cluster_tolerance");
    }
    std::string first_filename;
    std::string second_filename;
    std::string merged_map;
    size_t count_of_features;
    float scale_factor;
    float ratio_thresh;
    bool show_maps;
//    float cluster_tolerance;
};

class TBM_map_merge {
public:
    Algorithm_parameters parameters;
    std::shared_ptr<UnboundedPlainGridMap> first_map;
    std::shared_ptr<UnboundedPlainGridMap> second_map;
    std::array<cv::Mat,4> first_parts_maps, second_parts_maps;

    std::vector<cv::KeyPoint> kp_first_conc, kp_second_conc;
    cv::Mat d_first_occ, d_first_emp, d_first_unk,
            d_second_occ,d_second_emp,d_second_unk;

    std::vector<cv::DMatch> good_matches_conc;

public:
    TBM_map_merge(const Algorithm_parameters& p);

    std::shared_ptr<UnboundedPlainGridMap> merge();

protected:

    void detectAndCompute();
    std::shared_ptr<UnboundedPlainGridMap> get_first_transform();

//    pcl::PointCloud<pcl::PointXYZ>::Ptr get_translation_of_keypoints(
//        std::vector<cv::KeyPoint> &kp_first,
//        std::vector<cv::KeyPoint> &kp_second,
//        std::vector<cv::DMatch> &good_matches
//    );
//
//    std::vector<pcl::PointIndices> get_keypoints_translations_clusters(
//        const pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoints_translations
//    );
//    void print_clusters(
//            const pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoints_translations,
//            const std::vector<pcl::PointIndices>& cluster_indices
//    );

    void rotate_parts(std::vector<pcl::PointIndices> &cluster_indices);
    double compute_average_distance(const std::vector<cv::Point2f> &first,
            const std::vector<cv::Point2f> &second, const cv::Mat &transform);
};

TBM_map_merge::TBM_map_merge(const Algorithm_parameters& p) : parameters(p) {
//    if(p.read_imgs) {
//        auto gmp = MapValues::gmp;
//        first_map = std::make_shared<UnboundedPlainGridMap>(std::make_shared<VinyDSCell>(), gmp);
//        second_map = std::make_shared<UnboundedPlainGridMap>(std::make_shared<VinyDSCell>(), gmp);
//        std::ifstream in(parameters.first_filename);
//        std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
//                                    std::istreambuf_iterator<char>());
//        first_map->load_state(file_content);
//        first_parts_maps = first_map->get_maps_grs_ofu();
//
//        second_parts_maps.at(0) =
//                cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_2_grs_edited.jpg", cv::IMREAD_GRAYSCALE);
//        second_parts_maps.at(1) =
//                cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_2_occ_edited.jpg", cv::IMREAD_GRAYSCALE);
//        second_parts_maps.at(2) =
//                cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_2_emp_edited.jpg", cv::IMREAD_GRAYSCALE);
//        second_parts_maps.at(3) =
//                cv::imread("/home/dmo/Documents/diplom/pictures/rotated_maps_ph/map_2_unk_edited.jpg", cv::IMREAD_GRAYSCALE);
//    }
//    else {
    auto gmp = MapValues::gmp;
    first_map = std::make_shared<UnboundedPlainGridMap>(std::make_shared<VinyDSCell>(), gmp);
    second_map = std::make_shared<UnboundedPlainGridMap>(std::make_shared<VinyDSCell>(), gmp);
    std::ifstream in(parameters.first_filename);
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
    first_map->load_state(file_content);
    first_map->crop_by_bounds();
    auto rotated = first_map->rotate(0.785398163);
    first_map.swap(rotated);

    first_parts_maps = first_map->get_maps_grs_ofu();
    in.close();
    in = std::ifstream(parameters.second_filename);
    file_content = std::vector<char>((std::istreambuf_iterator<char>(in)),
                                        std::istreambuf_iterator<char>());
    second_map->load_state(file_content);
    second_map->crop_by_bounds();
    second_parts_maps = second_map->get_maps_grs_ofu();
    in.close();
//    }
}

void TBM_map_merge::detectAndCompute(){
    cv::Ptr<cv::ORB> detector = cv::ORB::create( parameters.count_of_features, parameters.scale_factor );

    detector->detect(first_parts_maps.at(0), kp_first_conc);
    detector->compute(first_parts_maps.at(1), kp_first_conc, d_first_occ);
    detector->compute(first_parts_maps.at(2), kp_first_conc, d_first_emp);
    detector->compute(first_parts_maps.at(3), kp_first_conc, d_first_unk);

    detector->detect(second_parts_maps.at(0), kp_second_conc);
    detector->compute(second_parts_maps.at(1), kp_second_conc, d_second_occ);
    detector->compute(second_parts_maps.at(2), kp_second_conc, d_second_emp);
    detector->compute(second_parts_maps.at(3), kp_second_conc, d_second_unk);
}

std::shared_ptr<UnboundedPlainGridMap> TBM_map_merge::get_first_transform()
{
    std::vector<cv::Point2f> first, second;
    for (auto &match: good_matches_conc){
        const cv::Point2f &kp_first = kp_first_conc.at(match.queryIdx).pt;
        const cv::Point2f &kp_second = kp_second_conc.at(match.trainIdx).pt;
        first.push_back(kp_first);
        second.push_back(kp_second);
    }
    cv::Mat transform = cv::estimateRigidTransform(first, second, true);
//    compute_average_distance(first, second, transform);
    if(transform.dims == 2){
        DiscretePoint2D changed_sz;
        cv::Mat first_img = first_map->convert_to_grayscale_img();
        auto new_map = first_map->apply_transform(transform, changed_sz);
        auto merged_map_disjunctive = new_map->full_merge_disjunctive(second_map);
        return merged_map_disjunctive;

//        auto merged_map_conjunctive = new_map->full_merge_conjunctive(second_map);

//        cv::Mat merged_map_disj_grs = merged_map_disjunctive->convert_to_grayscale_img();
//        cv::Mat merged_map_conj_grs = merged_map_conjunctive->convert_to_grayscale_img();
//        cv::Mat sec_grs = second_map->convert_to_grayscale_img();
//        int k=0;
//        while(k!= 27){
//            cv::imshow("first map", first_img);
//            cv::imshow("second map", sec_grs);
//            cv::imshow("merged disj", merged_map_disj_grs);
//            cv::imshow("merged conj", merged_map_conj_grs);
////            cv::imwrite("/home/dmo/Documents/diplom/pictures/result_first_map.jpg", first_img);
////            cv::imwrite("/home/dmo/Documents/diplom/pictures/result_second_map.jpg", sec_grs);
////            cv::imwrite("/home/dmo/Documents/diplom/pictures/result_merged_conjunctive.jpg", merged_map_conj_grs);
////            cv::imwrite("/home/dmo/Documents/diplom/pictures/result_merged_disjunctive.jpg", merged_map_disj_grs);
//            k=cv::waitKey();
//        }
    }

}

std::shared_ptr<UnboundedPlainGridMap> TBM_map_merge::merge() {
    detectAndCompute();

    cv::Mat concatenate_d_first = concatenateDescriptor(d_first_occ,
                                                        d_first_emp,
                                                        d_first_unk);
    cv::Mat concatenate_d_second = concatenateDescriptor(d_second_occ,
                                                         d_second_emp,
                                                         d_second_unk);

    good_matches_conc = get_good_matches(concatenate_d_first, concatenate_d_second, parameters.ratio_thresh);
    auto merged = get_first_transform();
    if (parameters.show_maps){
        cv::Mat first_map_img = first_map->convert_to_grayscale_img();
        cv::Mat second_map_img = second_map->convert_to_grayscale_img();
        cv::Mat result_map_img = merged->convert_to_grayscale_img();
        int k=0;
        while (k != 27){
            cv::imshow("first map", first_map_img);
            cv::imshow("second map", second_map_img);
            cv::imshow("merged map", result_map_img);
            k = cv::waitKey();
        }
    }
    return merged;

//    alg part
//    --------------------------------------------
//    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_translations =
//            get_translation_of_keypoints(kp_first_conc, kp_second_conc, good_matches_conc);
//    std::vector<pcl::PointIndices> clusters_indices = get_keypoints_translations_clusters(keypoints_translations);
//    rotate_parts(clusters_indices);
//    --------------------------------------------
}

//void TBM_map_merge::print_clusters(
//    const pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoints_translations,
//    const std::vector<pcl::PointIndices>& cluster_indices
//){
//    for (const auto & cluster_index : cluster_indices)
//    {
//        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
//        for (int index : cluster_index.indices){
//            cloud_cluster->points.push_back(keypoints_translations->points[index]);
//        }
//        cloud_cluster->width = cloud_cluster->points.size ();
//        cloud_cluster->height = 1;
//        cloud_cluster->is_dense = true;
//    }
//}
//std::vector<pcl::PointIndices> TBM_map_merge::get_keypoints_translations_clusters(
//    const pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoints_translations
//){
//    if (keypoints_translations->empty()){
//        return std::vector<pcl::PointIndices>();
//    }
//    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
//    tree->setInputCloud(keypoints_translations);
//
//    std::vector<pcl::PointIndices> cluster_indices;
//    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
//    ec.setClusterTolerance(parameters.cluster_tolerance);
//    ec.setMinClusterSize(1);
//    ec.setMaxClusterSize(parameters.count_of_features);
//    ec.setSearchMethod(tree);
//    ec.setInputCloud(keypoints_translations);
//    ec.extract(cluster_indices);
//    return cluster_indices;
//}
//pcl::PointCloud<pcl::PointXYZ>::Ptr TBM_map_merge::get_translation_of_keypoints(
//        std::vector<cv::KeyPoint> &kp_first,
//        std::vector<cv::KeyPoint> &kp_second,
//        std::vector<cv::DMatch> &good_matches
//        )
//{
//    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_translations(new pcl::PointCloud<pcl::PointXYZ>);
//    try{
//        for(auto & match : good_matches) {
//            keypoints_translations->push_back(
//                pcl::PointXYZ(
//                    kp_first[match.queryIdx].pt.x - kp_second[match.trainIdx].pt.x,
//                    kp_first[match.queryIdx].pt.y - kp_second[match.trainIdx].pt.y,
//                    0 ));
//        }
//    }
//    catch(std::out_of_range &ex){
//        std::cout << ex.what() << std::endl;
//    }
//    return keypoints_translations;
//}

void TBM_map_merge::rotate_parts(std::vector<pcl::PointIndices> &cluster_indices){
    std::vector<cv::Mat> imgs;
    cv::Mat img;
    DiscretePoint2D shift_map;
    for(auto &cluster: cluster_indices) {
        if(cluster.indices.size()>2) {
            std::vector<cv::Point2f> first, second;
            for (auto &index: cluster.indices){
                const cv::DMatch &match = good_matches_conc.at(index);
                const cv::Point2f &kp_first = kp_first_conc.at(match.queryIdx).pt;
                const cv::Point2f &kp_second = kp_second_conc.at(match.trainIdx).pt;
                first.push_back(kp_first);
                second.push_back(kp_second);
            }
            cv::Mat transform = cv::estimateRigidTransform(first, second, true);

            if(transform.dims) {
                auto transformed = first_map->apply_transform(transform, shift_map);
                img = transformed->convert_to_grayscale_img();
                imgs.push_back(img);
                imgs.push_back(second_map->convert_to_grayscale_img());
                auto merged_maps = transformed->full_merge_disjunctive(second_map);
                imgs.push_back(merged_maps->convert_to_grayscale_img());
                cv::Mat out(img.rows, img.cols, first_parts_maps.at(0).type());
                cv::warpAffine(first_parts_maps.at(0), out, transform, out.size());
            }
        }
    }
    int k=0;
    while(k != 27) {
        for (size_t id = 0; id < imgs.size(); id++) {
            cv::imshow(std::string("img") + std::to_string(id), imgs.at(id));
        }
        k = cv::waitKey();
    }
}

double TBM_map_merge::compute_average_distance(const std::vector<cv::Point2f> &first,
                                               const std::vector<cv::Point2f> &second,
                                               const cv::Mat &transform){
    auto type_ = transform.type();
    cv::Mat mat(3,1, type_);
    cv::Mat second_;
    std::vector<double> distances;
    for(size_t i = 0; i < first.size(); ++i){
        std::vector<double> data_{first.at(i).x, first.at(i).y, 1};
        std::memcpy(mat.data, data_.data(), data_.size()*sizeof(double));
        cv::Mat transformed_point = transform * mat;
        cv::Mat(second.at(i)).convertTo(second_, type_);
        double norm = cv::norm(transformed_point, second_);
        distances.push_back(norm);

    }
    double avg_value = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
    auto minmax = std::minmax_element(distances.begin(), distances.end());
    std::cout << "min: " << *(minmax.first) << std::endl
            << "max: " << *(minmax.second) << std::endl;
    std::cout << "avg: " << avg_value << std::endl;
    std::ofstream output_file("/home/dmo/Documents/diplom/distances.txt");
    std::ostream_iterator<double> output_iterator(output_file, "\n");
    std::copy(distances.begin(), distances.end(), output_iterator);
    return avg_value;
}

//.

#endif //TBM_MAP_MERGE_HPP