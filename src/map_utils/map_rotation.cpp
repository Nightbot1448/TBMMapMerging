#include <fstream>
#include <iterator>

#include <cmath>

#include <nav_msgs/OccupancyGrid.h>

#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


int main(int argc, char **argv) {
    
    const cv::String keys =
        "{path           |/home/dmo/Documents/diplom/dumps/compressed_dump_8.txt | path to file }"
        "{angle          | 45    | angle in deg }"
        "{axis_x         | 0     | axis point x coord }"
        "{axis_y         | 0     | axis point y coord }";

    cv::CommandLineParser parser(argc, argv, keys);
    float angle = parser.get<float>("angle");
    int axis_x = parser.get<int>("axis_x");
    int axis_y = parser.get<int>("axis_y");
    cv::String map_path = parser.get<cv::String>("path");
    
    auto gmp = MapValues::gmp;

    DiscretePoint2D axis(axis_x, axis_y);

    UnboundedPlainGridMap map = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    std::string input_file(map_path);
    std::ifstream in(map_path);
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
    map.load_state(file_content);
    
    cv::Mat orig_map = map.convert_to_grayscale_img();
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    std::shared_ptr<UnboundedPlainGridMap> rotated_pos = map.rotate(angle, { 200, -200});
    std::shared_ptr<UnboundedPlainGridMap> rotated_neg = map.rotate(angle, {-200,  200});
    end = std::chrono::system_clock::now();
    int elapsed_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>
                             (end-start).count();
    std::cout<< "time of 2 rotate: " << elapsed_milliseconds << std::endl;

    cv::imshow("original", orig_map);
    cv::imshow("rotated {-200, 200}", rotated_neg->convert_to_grayscale_img());
    cv::imshow("rotated {200, -200}", rotated_pos->convert_to_grayscale_img());
    
    int k = 0;
    while(k != 27){
        k = cv::waitKey(50);
    }
    return 0;
}