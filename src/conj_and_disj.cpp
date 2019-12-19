#include <ros/ros.h>
#include <fstream>
#include <iterator>

#include <nav_msgs/OccupancyGrid.h>

#include "core/maps/plain_grid_map.h"
#include "slams/viny/viny_grid_cell.h"

#include <typeinfo>

int main(int  argc, char **argv) {
    ros::init(argc, argv, "load_map_ff");

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

    std::ifstream in("/home/dmo/Documents/diplom/dumps/compressed_dump_8.txt");
    std::ifstream in_second("/home/dmo/Documents/diplom/dumps/compressed_dump_0.txt");
    
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
    std::vector<char> second_file_content((std::istreambuf_iterator<char>(in_second)),
                                   std::istreambuf_iterator<char>());

    map.load_state(file_content);
    map_second.load_state(second_file_content);

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
    nav_msgs::OccupancyGrid map_msg_conj(map_msg);
    nav_msgs::OccupancyGrid map_msg_disj(map_msg);


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

            // if(res_conj != el0 && res_conj != el1 )
            // {
            //     std::cout << pnt << "; " << el0 << ", " << el1 << ", " << res_conj << std::endl;
            // }

            Occupancy o_conj = TBM_to_O(res_conj);
            Occupancy o_disj = TBM_to_O(res_disj);

            map_conj.setCell(pnt, VinyDSCell(o_conj, res_conj));
            map_disj.setCell(pnt, VinyDSCell(o_disj, res_disj));

            value = static_cast<double>(o_conj);
            cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
            map_msg_conj.data.push_back(cell_value);         

            value = static_cast<double>(o_disj);
            cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
            map_msg_disj.data.push_back(cell_value);         
            
        }
    }

    map_conj.save_state_to_file();
    map_disj.save_state_to_file();

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