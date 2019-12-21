#include <ros/ros.h>
#include <fstream>
#include <iterator>

#include <nav_msgs/OccupancyGrid.h>

#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"

void push_to_map_msg(nav_msgs::OccupancyGrid &msg, const GridCell &cell);

int main(int  argc, char **argv) {
    ros::init(argc, argv, "parts_of_map");

    ros::NodeHandle nh;
    ros::Publisher pub =        nh.advertise<nav_msgs::OccupancyGrid>("map", 5);
    ros::Publisher pub_part_1 = nh.advertise<nav_msgs::OccupancyGrid>("map_second", 5);
    ros::Publisher pub_part_2 = nh.advertise<nav_msgs::OccupancyGrid>("map_conj", 5); // just topic name
    std::string tf_frame = "odom_combined";
    auto gmp = MapValues::gmp;


    UnboundedPlainGridMap map = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    
    std::string input_file("/home/dmo/Documents/diplom/dumps/compressed_dump_8.txt");
    std::string first_output_file("/home/dmo/Documents/diplom/dumps/map_8_part_1.txt");
    std::string second_output_file("/home/dmo/Documents/diplom/dumps/map_8_part_2.txt");

    bool is_horizontal = false;
    int percent = 60;

    nh.getParam("/parts_of_map/input_file", input_file);
    nh.getParam("/parts_of_map/first_output_file", first_output_file);
    nh.getParam("/parts_of_map/second_output_file", second_output_file);
    nh.getParam("/parts_of_map/is_horizontal", is_horizontal);
    nh.getParam("/parts_of_map/percent", percent);

    std::cout << "input_file: " << input_file << std::endl
              << "first_output_file: " << first_output_file << std::endl
              << "second_output_file: " << second_output_file << std::endl
              << "is_horizontal: " << is_horizontal << std::endl
              << "percent: " << percent << std::endl;

    std::ifstream in(input_file);
    
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
    
    map.load_state(file_content);

    UnboundedPlainGridMap map_part_1 = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    UnboundedPlainGridMap map_part_2 = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    map_part_1.clone_other_map_properties(map);
    map_part_2.clone_other_map_properties(map);

    
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
    
    // DiscretePoint2D zero;
    // std::cout << -origin <<std::endl;
    // std::cout << end_of_map <<std::endl;
    // std::cout << zero <<std::endl;
    // std::cout << "{" << info.width << ',' << info.height << '}' << std::endl;
    
    nav_msgs::OccupancyGrid map_part_1_msg(map_msg);
    nav_msgs::OccupancyGrid map_part_2_msg(map_msg);


    if (is_horizontal) { // split map by vertical line
        int count_of_cells_in_part = map.width() * percent / 100,
            max_cell_x_part_1 = -origin.x + count_of_cells_in_part,
            min_cell_x_part_2 = end_of_map.x - count_of_cells_in_part;
        
        for (pnt.y = -origin.y; pnt.y < end_of_map.y; ++pnt.y) {
            for (pnt.x = -origin.x; pnt.x < end_of_map.x; ++pnt.x) {
                const GridCell &map_value = map[pnt];
                push_to_map_msg(map_msg, map_value);
                
                if(pnt.x <= max_cell_x_part_1)
                {
                    map_part_1.setCell(pnt, new VinyDSCell(dynamic_cast<const VinyDSCell &>(map_value)));
                }
                push_to_map_msg(map_part_1_msg, map_part_1[pnt]);
                
                if(pnt.x >= min_cell_x_part_2)
                {
                    map_part_2.setCell(pnt, new VinyDSCell(dynamic_cast<const VinyDSCell &>(map_value)));
                }
                push_to_map_msg(map_part_2_msg, map_part_2[pnt]);
            }
        }
    }
    else
    {
        int count_of_cells_in_part = map.height() * percent / 100,
            max_cell_y_part_1 = -origin.y + count_of_cells_in_part,
            min_cell_y_part_2 = end_of_map.y - count_of_cells_in_part;

        for (pnt.y = -origin.y; pnt.y < end_of_map.y; ++pnt.y) {
            if (pnt.y<=max_cell_y_part_1 && pnt.y>=min_cell_y_part_2){
                for (pnt.x = -origin.x; pnt.x < end_of_map.x; ++pnt.x) {
                    const GridCell &map_value = map[pnt];
                    push_to_map_msg(map_msg, map_value);
                    map_part_1.setCell(pnt, new VinyDSCell(dynamic_cast<const VinyDSCell &>(map_value)));
                    push_to_map_msg(map_part_1_msg, map_part_1[pnt]);
                    map_part_2.setCell(pnt, new VinyDSCell(dynamic_cast<const VinyDSCell &>(map_value)));
                    push_to_map_msg(map_part_2_msg, map_part_2[pnt]);
                }
            }
            else
            {
                for (pnt.x = -origin.x; pnt.x < end_of_map.x; ++pnt.x) {
                    const GridCell &map_value = map[pnt];
                    if(pnt.y<=max_cell_y_part_1){ // to part 1
                        
                        map_part_1.setCell(pnt, new VinyDSCell(dynamic_cast<const VinyDSCell &>(map_value)));
                        map_part_2.setCell(pnt, new VinyDSCell);
                    }
                    else { // to part 2
                        map_part_1.setCell(pnt, new VinyDSCell);
                        map_part_2.setCell(pnt, new VinyDSCell(dynamic_cast<const VinyDSCell &>(map_value)));
                    }
                    push_to_map_msg(map_msg, map_value);
                    push_to_map_msg(map_part_1_msg, map_part_1[pnt]);
                    push_to_map_msg(map_part_2_msg, map_part_2[pnt]);
                }
            }
        }
    }

    map_part_1.save_state_to_file(first_output_file);
    map_part_2.save_state_to_file(second_output_file);

    pub.publish(map_msg);
    ros::spinOnce();
    pub_part_1.publish(map_part_1_msg);
    ros::spinOnce();
    pub_part_2.publish(map_part_2_msg);
    ros::spinOnce();

    return 0;
}

void push_to_map_msg(nav_msgs::OccupancyGrid &msg, const GridCell &cell){
    double value = static_cast<double>(cell);
    int cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
    msg.data.push_back(cell_value);
}