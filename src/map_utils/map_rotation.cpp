#include <fstream>
#include <iterator>

#define _USE_MATH_DEFINES
#include <cmath>

#include <nav_msgs/OccupancyGrid.h>

#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


int main(int argc, char **argv) {
    
    const cv::String keys =
        "{path           |/home/dmo/Documents/diplom/dumps/compressed_dump_8.txt | path to file }"
        "{angle          | 180   | angle in deg }"
        "{axis_x         | 0     | axis point x coord }"
        "{axis_y         | 0     | axis point y coord }";

    cv::CommandLineParser parser(argc, argv, keys);
    float angle = parser.get<float>("angle");
    int axis_x = parser.get<int>("axis_x");
    int axis_y = parser.get<int>("axis_y");
    cv::String map_path = parser.get<cv::String>("path");
    
    auto gmp = MapValues::gmp;

    DiscretePoint2D axis(axis_x, axis_y);

    std::cout << "axis: " << axis << std::endl;

    UnboundedPlainGridMap map = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    
    std::string input_file(map_path);

    std::ifstream in(map_path);
    
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
    
    map.load_state(file_content);
    
    cv::Mat orig_map = map.convert_to_grayscale_img();
    cv::imshow("original", orig_map);

    UnboundedPlainGridMap map_second = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    map_second.clone_other_map_properties(map);
    // UnboundedPlainGridMap map_third = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    // map_third.clone_other_map_properties(map);
    // UnboundedPlainGridMap map_fourth = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    // map_fourth.clone_other_map_properties(map);
    // UnboundedPlainGridMap map_fifth = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    // map_fifth.clone_other_map_properties(map);

    int width = map.width();
    int height = map.height();
    DiscretePoint2D origin = map.origin();
    DiscretePoint2D pnt;
    DiscretePoint2D end_of_map = DiscretePoint2D(width,
                                                 height) - origin;
    
    std::cout << -origin <<std::endl;
    std::cout << end_of_map <<std::endl;
    
    double sin_angle = std::sin(angle * CV_PI / 180);
    double cos_angle = std::cos(angle * CV_PI / 180);
    
    // for(pnt.y = -origin.y; pnt.y < end_of_map.y; ++pnt.y) {
    //     for (pnt.x = -origin.x; pnt.x < end_of_map.x; ++pnt.x) {
    //         const GridCell &map_value = map[pnt];
    //         DiscretePoint2D new_cell_pnt;
    //         new_cell_pnt.x = std::round((pnt.x-axis.x) * cos_angle - (pnt.y - axis.y) * sin_angle);
    //         new_cell_pnt.y = std::round((pnt.x-axis.x) * sin_angle + (pnt.y - axis.y) * cos_angle);
    //         new_cell_pnt += axis;
    //         map_second.setCell(new_cell_pnt, new VinyDSCell(dynamic_cast<const VinyDSCell &>(map_value)));
            // new_cell_pnt.x = std::trunc((pnt.x-axis.x) * cos_angle - (pnt.y - axis.y) * sin_angle);
            // new_cell_pnt.y = std::trunc((pnt.x-axis.x) * sin_angle + (pnt.y - axis.y) * cos_angle);
            // new_cell_pnt += axis;
            // map_third.setCell(new_cell_pnt, new VinyDSCell(dynamic_cast<const VinyDSCell &>(map_value)));
            // new_cell_pnt.x = std::ceil((pnt.x-axis.x) * cos_angle - (pnt.y - axis.y) * sin_angle);
            // new_cell_pnt.y = std::ceil((pnt.x-axis.x) * sin_angle + (pnt.y - axis.y) * cos_angle);
            // new_cell_pnt += axis;
            // map_fourth.setCell(new_cell_pnt, new VinyDSCell(dynamic_cast<const VinyDSCell &>(map_value)));
            // new_cell_pnt.x = std::floor((pnt.x-axis.x) * cos_angle - (pnt.y - axis.y) * sin_angle);
            // new_cell_pnt.y = std::floor((pnt.x-axis.x) * sin_angle + (pnt.y - axis.y) * cos_angle);
            // new_cell_pnt += axis;
            // map_fifth.setCell(new_cell_pnt, new VinyDSCell(dynamic_cast<const VinyDSCell &>(map_value)));
    //     }
    // }

    map.cropp_by_bounds();

    cv::Mat cropped = map.convert_to_grayscale_img();
    cv::imshow("cropped", cropped);


    std::cout << "origin_size: " << map.width() << ' ' << map.height() << std::endl;
    // std::cout << "new_size: " << map_second.width() << ' ' << map_second.height() << std::endl;
    // std::chrono::time_point<std::chrono::system_clock> start, end;
    // start = std::chrono::system_clock::now();

    // end = std::chrono::system_clock::now();
    // int elapsed_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>
    //                          (end-start).count();

    // std::cout<< "time: " << elapsed_milliseconds << std::endl;
    
    
    // cv::imshow("round", map_second.convert_to_grayscale_img());
    // cv::imshow("trunc", map_third.convert_to_grayscale_img());
    // cv::imshow("ceil",  map_fourth.convert_to_grayscale_img());
    // cv::imshow("floor", map_fifth.convert_to_grayscale_img());
    
    int k = 0;
    while(k != 27){
        k = cv::waitKey(50);
    }
    return 0;
}