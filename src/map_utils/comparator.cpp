#include "../map_utils_headers/compute_descriptors.h"
#include "../map_utils_headers/orbDescriptorscomparator.h"
#include "../map_utils_headers/descriptorsComarator.h"
// #include "orbDescriptorscomparator.h"
// #include "descriptorsComarator.h"
#include <chrono>

int main(int argc, char **argv) {
    const cv::String keys =
        "{first_map            | /home/dmo/Documents/diplom/dumps/compressed_dump_2_11.txt | first dump for load}"
        "{second_map           | /home/dmo/Documents/diplom/dumps/2_floor_31.txt | second dump for load}"
        "{base_orb_tests_path  | /home/dmo/Documents/diplom/orb_tests/ | base orb path }"
        "{base_desc_tests_path | /home/dmo/Documents/diplom/desc_tests/ | base orb/brisk path }"
        "{count_of_features    | 2000              | count of faetures to extract }"
        "{scale_factor         | 1.2               | scale factor }"
        "{ratio_thresh         | 0.7               | filter matches using the Lowe's ratio test }"
        "{test_id              | -1                | test id }"
        "{min_dist             | 0                 | min distance between keypoints }"
        "{max_dist             | 0                 | max distance between keypoints }"
        "{cluster_tolerance    | 0.02              | cluster tolerance for EuclideanClusterExtraction }"
        "{read_imgs            | false             | read maps or imgs }"
        ;
    cv::CommandLineParser parser(argc, argv, keys);
    Parameters p(parser);
    OrbDescriptorsComparator comparator(p);
    // DescriptorsComparator comparator(p);

    // std::chrono::time_point<std::chrono::system_clock> before_comp = std::chrono::system_clock::now();
    // comparator.compareDescriptors();
    // std::chrono::time_point<std::chrono::system_clock> after_comp = std::chrono::system_clock::now();
    // std::cout << "compare: " << std::chrono::duration_cast<std::chrono::milliseconds>(after_comp-before_comp).count() << std::endl;

    // cv::imshow("occ", second_parts_maps.at(1));
    // cv::imshow("emp", second_parts_maps.at(2));
    // cv::imshow("unk", second_parts_maps.at(3));
    // cv::imshow("inv_unk", out_);
    // cv::waitKey(0);

    

    return 0;
}
