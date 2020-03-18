#include <ros/ros.h>
#include <fstream>
#include <iterator>
#include <chrono>

#include <nav_msgs/OccupancyGrid.h>

#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"

#include <typeinfo>

int main(int  argc, char **argv) {
    ros::init(argc, argv, "conj_and_disj");

    ros::NodeHandle nh;
    ros::Publisher pub =            nh.advertise<nav_msgs::OccupancyGrid>("map", 5);
    ros::Publisher pub_second_map = nh.advertise<nav_msgs::OccupancyGrid>("map_second", 5);
    ros::Publisher pub_conj =       nh.advertise<nav_msgs::OccupancyGrid>("map_conj", 5);
    ros::Publisher pub_disj =       nh.advertise<nav_msgs::OccupancyGrid>("map_disj", 5);
    std::string tf_frame = "odom_combined";
    auto gmp = MapValues::gmp;

    UnboundedPlainGridMap map =         UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    UnboundedPlainGridMap map_second =  UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    UnboundedPlainGridMap map_conj =    UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    UnboundedPlainGridMap map_disj =    UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    map_conj.clone_other_map_properties(map);
    map_disj.clone_other_map_properties(map);

    std::string first_file("/home/dmo/Documents/diplom/dumps/compressed_dump_8.txt");
    std::string second_file("/home/dmo/Documents/diplom/dumps/compressed_dump_0.txt");

    nh.getParam("/conj_and_disj/first_file", first_file);
    nh.getParam("/conj_and_disj/second_file", second_file);

    std::ifstream in(first_file);
    std::ifstream in_second(second_file);
    
    std::chrono::time_point<std::chrono::system_clock> before_reading = std::chrono::system_clock::now();

    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
    std::vector<char> second_file_content((std::istreambuf_iterator<char>(in_second)),
                                   std::istreambuf_iterator<char>());

    map.load_state(file_content);
    map_second.load_state(second_file_content);

    std::chrono::time_point<std::chrono::system_clock> after_reading = std::chrono::system_clock::now();

    ROS_INFO("time for read: %ld", std::chrono::duration_cast<std::chrono::milliseconds>(after_reading-before_reading).count());

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
    
    nav_msgs::OccupancyGrid map_msg_second(map_msg);
    nav_msgs::OccupancyGrid map_msg_conj(map_msg);
    nav_msgs::OccupancyGrid map_msg_disj(map_msg);

    ROS_INFO("new map size: %d %d", map_conj.width(), map_conj.height());
    std::chrono::time_point<std::chrono::system_clock> before_merge = std::chrono::system_clock::now();
    
    for (pnt.y = -origin.y; pnt.y < end_of_map.y; ++pnt.y) {
        for (pnt.x = -origin.x; pnt.x < end_of_map.x; ++pnt.x) {
            const TBM &el0 = dynamic_cast<const VinyDSCell &>(map[pnt]).belief();
            const TBM &el1 = dynamic_cast<const VinyDSCell &>(map_second[pnt]).belief();
            TBM res = conjunctive(el0,el1);
            Occupancy occ = TBM_to_O(res);
            map_conj.setCell(pnt, new VinyDSCell(occ, res));
        }
    }
    std::chrono::time_point<std::chrono::system_clock> after_merge = std::chrono::system_clock::now();
    ROS_INFO("conj merge: %ld", std::chrono::duration_cast<std::chrono::milliseconds>(after_merge-before_merge).count());
    
    before_merge = std::chrono::system_clock::now();
    for (pnt.y = -origin.y; pnt.y < end_of_map.y; ++pnt.y) {
        for (pnt.x = -origin.x; pnt.x < end_of_map.x; ++pnt.x) {
            const TBM &el0 = dynamic_cast<const VinyDSCell &>(map[pnt]).belief();
            const TBM &el1 = dynamic_cast<const VinyDSCell &>(map_second[pnt]).belief();
            TBM res = disjunctive(el0,el1);
            Occupancy occ = TBM_to_O(res);
            map_disj.setCell(pnt, new VinyDSCell(occ, res));
        }
    }
    after_merge = std::chrono::system_clock::now();
    ROS_INFO("disj merge: %ld", std::chrono::duration_cast<std::chrono::milliseconds>(after_merge-before_merge).count());

    for (pnt.y = -origin.y; pnt.y < end_of_map.y; ++pnt.y) {
        for (pnt.x = -origin.x; pnt.x < end_of_map.x; ++pnt.x) {
            
            double value = static_cast<double>(map[pnt]);
            int cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
            map_msg.data.push_back(cell_value);

            value = static_cast<double>(map_second[pnt]);
            cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
            map_msg_second.data.push_back(cell_value);

            const TBM &el0 = dynamic_cast<const VinyDSCell &>(map[pnt]).belief();
            const TBM &el1 = dynamic_cast<const VinyDSCell &>(map_second[pnt]).belief();
            TBM res_conj = conjunctive(el0,el1);
            TBM res_disj = disjunctive(el0,el1);

            Occupancy o_conj = TBM_to_O(res_conj);
            Occupancy o_disj = TBM_to_O(res_disj);

            value = static_cast<double>(o_conj);
            cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
            map_msg_conj.data.push_back(cell_value);         

            value = static_cast<double>(o_disj);
            cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
            map_msg_disj.data.push_back(cell_value);         
        }
    }

    map_conj.save_state_to_file("/home/dmo/Documents/diplom/dumps/conj.txt");
    map_disj.save_state_to_file("/home/dmo/Documents/diplom/dumps/disj.txt");

    pub.publish(map_msg);
    ros::spinOnce();
    pub_second_map.publish(map_msg_second);
    ros::spinOnce();
    pub_conj.publish(map_msg_conj);
    ros::spinOnce();
    pub_disj.publish(map_msg_disj);
    ros::spinOnce();
    
    return 0;
}