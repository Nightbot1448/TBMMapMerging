#include "../map_utils_headers/compute_descriptors.h"
#include "../map_utils_headers/orbDescriptorscomparator.h"
// #include "orbDescriptorscomparator.h"
// #include "descriptorsComarator.h"
#include <chrono>

int main(int argc, char **argv) {
    // const cv::String keys =
    //     "{first_map            | /home/dmo/Documents/diplom/dumps/compressed_dump_2_11.txt | first dump for load}"
    //     "{second_map           | /home/dmo/Documents/diplom/dumps/2_floor_31.txt | second dump for load}"
    //     "{base_orb_tests_path  | /home/dmo/Documents/diplom/orb_tests/ | base orb path }"
    //     "{base_desc_tests_path | /home/dmo/Documents/diplom/desc_tests/ | base orb/brisk path }"
    //     "{count_of_features    | 2000              | count of faetures to extract }"
    //     "{scale_factor         | 1.2               | scale factor }"
    //     "{ratio_thresh         | 0.7               | filter matches using the Lowe's ratio test }"
    //     "{test_id              | -1                | test id }"
    //     "{min_dist             | 0                 | min distance between keypoints }"
    //     "{max_dist             | 0                 | max distance between keypoints }"
    //     "{cluster_tolerance    | 0.02              | cluster tolerance for EuclideanClusterExtraction }"
    //     "{read_imgs            | false             | read maps or imgs }"
    //     ;
    // cv::CommandLineParser parser(argc, argv, keys);

    size_t out_file_id = 0;
    std::vector<int> count_of_features{100, 500, 1000, 2500, 5000};
    std::vector<float> scale_factor{0.5, 0.6, 0.7, 0.8};
    std::string dump_base("/home/dmo/Documents/Study/diplom_paper/data/dump_files/");
    std::vector<std::string> dump_files{
        "fl_2_2011-01-18-06-37-58/7.tbm_map",
        "fl_2_2011-01-19-07-49-38/8.tbm_map",
        "fl_2_2011-01-20-07-18-45/7.tbm_map",
        "fl_2_2011-01-24-06-18-27/8.tbm_map",
        "fl_2_2011-01-25-06-29-26/8.tbm_map"
    };

    for(auto cof: count_of_features){
        for(auto sf: scale_factor){
            for(size_t first_map_id = 0; first_map_id < dump_files.size() - 1; first_map_id++ ){
                for(size_t second_map_id = first_map_id+1; second_map_id < dump_files.size(); second_map_id++ ){
                    Parameters p(
                        dump_base + dump_files[first_map_id],
                        dump_base + dump_files[second_map_id],
                        cof, 1.2f, sf, 1.0f, 1000+out_file_id
                    );
                    OrbDescriptorsComparator comparator(p);
                    std::chrono::time_point<std::chrono::system_clock> before_comp = std::chrono::system_clock::now();
                    comparator.compareDescriptors();
                    std::chrono::time_point<std::chrono::system_clock> after_comp = std::chrono::system_clock::now();
                    std::cout << out_file_id << "/ 200 [" << cof << ", " << sf << "]; Compare time: " << 
                        std::chrono::duration_cast<std::chrono::milliseconds>(after_comp-before_comp).count() << std::endl;
                    out_file_id++;
                }
            }
        }
    }

    

    // cv::imshow("occ", second_parts_maps.at(1));
    // cv::imshow("emp", second_parts_maps.at(2));
    // cv::imshow("unk", second_parts_maps.at(3));
    // cv::imshow("inv_unk", out_);
    // cv::waitKey(0);

    

    return 0;
}
