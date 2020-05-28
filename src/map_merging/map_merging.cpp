#include "../map_utils_headers/compute_descriptors.h"
#include "TBM_map_merge.hpp"
#include <chrono>

int main(int argc, char **argv) {

    const cv::String keys =
        "{help h usage ?    |                      | print this message}"
        "{first_map         |                      | path to first dump for load (required)}"
        "{second_map        |                      | path to second dump for load (required)}"
        "{merged_map        | ~/merged_map.tbm_map | path to save result of merging }"
        "{count_of_features | 1000                 | count of features to extract }"
        "{scale_factor      | 1.2                  | scale factor (float number > 1) }"
        "{ratio_thresh      | 0.7                  | filter matches using the Lowe's ratio test (float number < 1) }"
        "{show_maps         | false                | show imgs of input maps and result map (boolean); Esc to close imgs }"
//        "{cluster_tolerance    | 2.5                  | cluster tolerance for EuclideanClusterExtraction }"
        ;
    cv::CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    if(!parser.has("first_map")){
        std::cout << "Path to first map was not found" << std::endl
            << "run with --help to see parameters" << std::endl;
        return 0;
    }
    if(!parser.has("second_map")){
        std::cout << "Path to second map was not found" << std::endl
                << "run with --help to see parameters" << std::endl;
        return 0;
    }

    Algorithm_parameters p(parser);
    TBM_map_merge cl(p);

    std::chrono::time_point<std::chrono::system_clock> before_comp = std::chrono::system_clock::now();
    auto merged_map = cl.merge();
    std::chrono::time_point<std::chrono::system_clock> after_comp = std::chrono::system_clock::now();
    merged_map->save_state_to_file(p.merged_map);
    std::cout << "maps was merged"  << std::endl
        << "result filepath: " << p.merged_map << std::endl << "compute time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(after_comp-before_comp).count()
            << std::endl;

    return 0;
}