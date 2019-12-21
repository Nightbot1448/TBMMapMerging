#include <ros/ros.h>
#include <fstream>
#include <iterator>

#include <nav_msgs/OccupancyGrid.h>

#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"



int main(int  argc, char **argv) {
    ros::init(argc, argv, "map_translation");

    ros::NodeHandle nh;
    ros::Publisher pub =            nh.advertise<nav_msgs::OccupancyGrid>("map", 5);
    ros::Publisher pub_second_map = nh.advertise<nav_msgs::OccupancyGrid>("map_second", 5);
    std::string tf_frame = "odom_combined";
    auto gmp = MapValues::gmp;


    UnboundedPlainGridMap map = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    
    std::string input_file("/home/dmo/Documents/diplom/dumps/compressed_dump_8.txt");
    std::string output_file("/home/dmo/Documents/diplom/dumps/output_file.txt");

    int shift_x = 5, shift_y = 0; 

    nh.getParam("map_translation/input_file", input_file);
    nh.getParam("map_translation/output_file", output_file);
    nh.getParam("map_translation/shift_x", shift_x);
    nh.getParam("map_translation/shift_y", shift_y);


    std::ifstream in(input_file);
    
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
    
    map.load_state(file_content);
    
    UnboundedPlainGridMap map_second = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    map_second.clone_other_map_properties(map);

    
    nav_msgs::OccupancyGrid map_msg;
    map_msg.header.frame_id = tf_frame;
    map_msg.info.map_load_time = ros::Time::now();
    map_msg.info.width = map.width();
    map_msg.info.height = map.height();
    map_msg.info.resolution = map.scale();

    // // move map to the middle
    nav_msgs::MapMetaData &info = map_msg.info;
    DiscretePoint2D origin = map.origin();
    // info.origin.position.x = -info.resolution * origin.x;
    // info.origin.position.y = -info.resolution * origin.y;
    // info.origin.position.z = 0;
    map_msg.data.reserve(info.height * info.width);
    DiscretePoint2D pnt;
    DiscretePoint2D end_of_map = DiscretePoint2D(info.width,
                                                 info.height) - origin;
    
    DiscretePoint2D zero;
    std::cout << -origin <<std::endl;
    std::cout << end_of_map <<std::endl;
    std::cout << zero <<std::endl;
    std::cout << "{" << info.width << ',' << info.height << '}' << std::endl;
    
    nav_msgs::OccupancyGrid map_msg_second(map_msg);
    

    for (pnt.y = -origin.y; pnt.y < end_of_map.y; ++pnt.y) {
        for (pnt.x = -origin.x; pnt.x < end_of_map.x; ++pnt.x) {
            double value = static_cast<double>(map[pnt]);
            int cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
            map_msg.data.push_back(cell_value);
        }
    }
    
    if (shift_x >= 0 && shift_y >= 0){
        for (pnt.y = -origin.y+shift_y; pnt.y < end_of_map.y; ++pnt.y) {
            for (pnt.x = -origin.x+shift_x; pnt.x < end_of_map.x; ++pnt.x) {
                const GridCell &map_value = map[pnt];
                DiscretePoint2D new_cell_pnt(pnt.x-shift_x, pnt.y - shift_y);
                map_second.setCell(new_cell_pnt, new VinyDSCell(dynamic_cast<const VinyDSCell &>(map_value)));
                double value = static_cast<double>(map_value);
                int cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
                map_msg_second.data.push_back(cell_value);
            }

            for (int shift = 0; shift < shift_x; ++shift) {
                DiscretePoint2D get_(-origin.x+shift, pnt.y);
                const GridCell &map_value = map[get_];
                DiscretePoint2D new_cell_pnt(end_of_map.x-shift_x + shift, pnt.y - shift_y);
                map_second.setCell(new_cell_pnt, new VinyDSCell(dynamic_cast<const VinyDSCell &>(map_value)));
                double value = static_cast<double>(map_value);
                int cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
                map_msg_second.data.push_back(cell_value);
            }
        }
        
        for (int shift = 0; shift < shift_y; ++shift) {
            for (pnt.x = -origin.x; pnt.x < end_of_map.x; ++pnt.x) {
                DiscretePoint2D get_(pnt.x, -origin.y+shift);
                const GridCell &map_value = map[get_];
                DiscretePoint2D new_cell_pnt(pnt.x, end_of_map.y - shift_y + shift);
                map_second.setCell(new_cell_pnt, new VinyDSCell(dynamic_cast<const VinyDSCell &>(map_value)));
                double value = static_cast<double>(map_value);
                int cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
                map_msg_second.data.push_back(cell_value);
            }
        }
    }
    
    map_second.save_state_to_file(output_file);

    pub.publish(map_msg);
    ros::spinOnce();
    pub_second_map.publish(map_msg_second);
    ros::spinOnce();

    return 0;
}