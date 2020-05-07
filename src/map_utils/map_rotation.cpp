#include <fstream>
#include <iterator>

#include <nav_msgs/OccupancyGrid.h>

#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


int main(int argc, char **argv) {
    
    const cv::String keys =
        "{path           |/home/dmo/Documents/diplom/dumps/compressed_dump_8.txt | path to file }"
        "{angle          | 45   | angle in deg }";

    cv::CommandLineParser parser(argc, argv, keys);
    float angle = parser.get<float>("angle");
    cv::String map_path = parser.get<cv::String>("path");
    
    auto gmp = MapValues::gmp;


    UnboundedPlainGridMap map = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    
    std::string input_file(map_path);


    int shift_x = -400, shift_y = 100;

    std::ifstream in(map_path);
    
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
    
    map.load_state(file_content);
    
    cv::Mat orig_map = map.convert_to_grayscale_img();
    cv::imshow("original", orig_map);

    UnboundedPlainGridMap map_second = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    map_second.clone_other_map_properties(map);

    int width = map.width();
    int height = map.height();
    DiscretePoint2D origin = map.origin();
    DiscretePoint2D pnt;
    DiscretePoint2D end_of_map = DiscretePoint2D(width,
                                                 height) - origin;
    
    DiscretePoint2D zero;
    std::cout << -origin <<std::endl;
    std::cout << end_of_map <<std::endl;
    std::cout << zero <<std::endl;
    
    // std::chrono::time_point<std::chrono::system_clock> start, end;
    // start = std::chrono::system_clock::now();

    // end = std::chrono::system_clock::now();
    // int elapsed_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>
    //                          (end-start).count();

    // std::cout<< "time: " << elapsed_milliseconds << std::endl;
    
    
    cv::imshow("changed", map.convert_to_grayscale_img());
    
    int k = 0;
    while(k != 27){
        k = cv::waitKey(50);
    }
    return 0;
}