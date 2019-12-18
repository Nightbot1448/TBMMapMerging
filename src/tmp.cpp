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
    // // ros::Publisher pub_disj = nh.advertise<nav_msgs::OccupancyGrid>("map_disj", 5);
    std::string tf_frame = "odom_combined";
    auto gmp = MapValues::gmp;

    UnboundedPlainGridMap map =         UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    UnboundedPlainGridMap map_second =  UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    UnboundedPlainGridMap map_conj =    UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    // UnboundedPlainGridMap map_disj = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);

    std::ifstream in("/home/dmo/Documents/diplom/dumps/compressed_dump_8.txt");
    std::ifstream in1("/home/dmo/Documents/diplom/dumps/compressed_dump_0.txt");
    
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
    std::vector<char> file_content1((std::istreambuf_iterator<char>(in1)),
                                   std::istreambuf_iterator<char>());

    map_second.load_state(file_content);
    map_conj.load_state(file_content1);

    nav_msgs::OccupancyGrid map_msg;
    map_msg.header.frame_id = tf_frame;
    map_msg.info.map_load_time = ros::Time::now();
    map_msg.info.width = map_second.width();
    map_msg.info.height = map_second.height();
    map_msg.info.resolution = map_second.scale();

    // // move map to the middle
    nav_msgs::MapMetaData &info = map_msg.info;
    DiscretePoint2D origin = map_second.origin();
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


    for (pnt.y = -origin.y; pnt.y < end_of_map.y; ++pnt.y) {
        for (pnt.x = -origin.x; pnt.x < end_of_map.x; ++pnt.x) {
            double value = static_cast<double>(map[pnt]);
            int cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
            map_msg.data.push_back(cell_value);

            value = static_cast<double>(map_second[pnt]);
            cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
            map_msg_second.data.push_back(cell_value);

            // const VinyDSCell &cell0 = dynamic_cast<const VinyDSCell &>(map_second[pnt]);
            const TBM &el0 = dynamic_cast<const VinyDSCell &>(map_second[pnt]).belief();
            const TBM &el1 = dynamic_cast<const VinyDSCell &>(map_conj[pnt]).belief();
            TBM res_conj = conjunctive(el0,el1);

            if(res_conj != el0 && res_conj != el1 )
            {
                std::cout << pnt << "; " << el0 << ", " << el1 << ", " << res_conj << std::endl;
            }
            value = static_cast<double>(TBM_to_O(res_conj));
            cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
            map_msg_conj.data.push_back(cell_value);         
            
        }
    }
    
    pub.publish(map_msg);
    ros::spinOnce();
    pub_second_map.publish(map_msg_second);
    ros::spinOnce();
    pub.publish(map_msg_conj);
    ros::spinOnce();
    ros::spinOnce();
    
    
//---------------------------------

    // map_msg1.header.frame_id = tf_frame;
    // map_msg1.info.map_load_time = ros::Time::now();
    // map_msg1.info.width = map_conj.width();
    // map_msg1.info.height = map_conj.height();
    // map_msg1.info.resolution = map_conj.scale();
    
    // // move map to the middle
    // nav_msgs::MapMetaData &info1 = map_msg1.info;
    // DiscretePoint2D origin1 = map_conj.origin();
    // // info1.origin.position.x = -info1.resolution * origin1.x;
    // // info1.origin.position.y = -info1.resolution * origin1.y;
    // // info1.origin.position.z = 0;
    // map_msg1.data.reserve(info1.height * info1.width);
    // DiscretePoint2D end_of_map_conj = DiscretePoint2D(info1.width, info1.height) - origin1;

    // for (pnt.y = -origin1.y; pnt.y < end_of_map_conj.y; ++pnt.y) {
    //     for (pnt.x = -origin1.x; pnt.x < end_of_map_conj.x; ++pnt.x) {
    //         double value = static_cast<double>(map_conj[pnt]);
    //         int cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
    //         map_msg1.data.push_back(cell_value);
    //     }
    // }
    // pub_conj.publish(map_msg1);
    // ros::spinOnce();

    return 0;
}