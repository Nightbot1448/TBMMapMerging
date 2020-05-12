#include <ros/ros.h>
#include <fstream>
#include <iterator>

#include <nav_msgs/OccupancyGrid.h>

#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"



int main(int  argc, char **argv) {
    const cv::String keys =
        "{path           |/home/dmo/Documents/diplom/dumps/compressed_dump_8.txt | path to file }"
        "{shift_x        | 100   | shift x }";
        "{shift_y        | 100   | shift y }";

    cv::CommandLineParser parser(argc, argv, keys);
    
    int shift_x = parser.get<int>("shift_x");
    int shift_y = parser.get<int>("shift_y");
    cv::String map_path = parser.get<cv::String>("path");
    
    auto gmp = MapValues::gmp;

    UnboundedPlainGridMap map = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    
    std::string input_file(map_path);
    std::ifstream in(map_path);
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
    map.load_state(file_content);
    
    cv::Mat orig_map = map.convert_to_grayscale_img();

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
    
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    for(pnt.y = -origin.y; pnt.y < end_of_map.y; ++pnt.y) {
        for (pnt.x = -origin.x; pnt.x < end_of_map.x; ++pnt.x) {
            const GridCell &map_value = map[pnt];
            DiscretePoint2D new_cell_pnt(pnt.x, (pnt.y + origin.y + height + shift_y )%height - origin.y );
            map_second.setCell(new_cell_pnt, new VinyDSCell(dynamic_cast<const VinyDSCell &>(map_value)));
        }
    }

    for(pnt.y = -origin.y; pnt.y < end_of_map.y; ++pnt.y) {
        for (pnt.x = -origin.x; pnt.x < end_of_map.x; ++pnt.x) {
            const GridCell &map_value = map_second[pnt];
            DiscretePoint2D new_cell_pnt((pnt.x + origin.x + width + shift_x )%width - origin.x, pnt.y );
            map.setCell(new_cell_pnt, new VinyDSCell(dynamic_cast<const VinyDSCell &>(map_value)));
        }
    }

    end = std::chrono::system_clock::now();
    int elapsed_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>
                             (end-start).count();

    std::cout<< "time: " << elapsed_milliseconds << std::endl;
    
    cv::imshow("original", orig_map);    
    cv::imshow("changed", map.convert_to_grayscale_img());
    int k = 0;
    while(k != 27){
        k = cv::waitKey(50);
    }

    return 0;
}