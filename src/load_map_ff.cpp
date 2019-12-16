#include <ros/ros.h>
#include <fstream>
#include <iterator>

#include "ros/occupancy_grid_publisher.h"
#include "ros/init_utils.h"

#include "core/maps/grid_map.h"
#include "core/maps/plain_grid_map.h"
#include "slams/viny/viny_grid_cell.h"

#include <typeinfo>

int main(int  argc, char **argv) {
    ros::init(argc, argv, "load_map_ff");

    ros::NodeHandle nh;
    ros::Publisher pub = nh.advertise<nav_msgs::OccupancyGrid>("map", 5);
    ros::Publisher pub0 = nh.advertise<nav_msgs::OccupancyGrid>("map0", 5);
    ros::Publisher pub1 = nh.advertise<nav_msgs::OccupancyGrid>("map1", 5);
    std::string tf_frame = "odom_combined";
    auto gmp = MapValues::gmp;

    UnboundedPlainGridMap map = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    UnboundedPlainGridMap map_0 = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    UnboundedPlainGridMap map_1 = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    std::ifstream in("/home/dmo/Documents/diplom/dumps/compressed_dump_8.txt");
    std::ifstream in1("/home/dmo/Documents/diplom/dumps/compressed_dump_0.txt");
    
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
    std::vector<char> file_content1((std::istreambuf_iterator<char>(in1)),
                                   std::istreambuf_iterator<char>());


    map_0.load_state(file_content);
    map_1.load_state(file_content1);


    nav_msgs::OccupancyGrid map_msg0;
    map_msg0.header.frame_id = tf_frame;
    map_msg0.info.map_load_time = ros::Time::now();
    map_msg0.info.width = map_0.width();
    map_msg0.info.height = map_0.height();
    map_msg0.info.resolution = map_0.scale();

    
    // // move map to the middle
    nav_msgs::MapMetaData &info = map_msg0.info;
    DiscretePoint2D origin = map_0.origin();
    // info.origin.position.x = -info.resolution * origin.x;
    // info.origin.position.y = -info.resolution * origin.y;
    info.origin.position.z = 0;
    map_msg0.data.reserve(info.height * info.width);
    DiscretePoint2D pnt;
    DiscretePoint2D end_of_map = DiscretePoint2D(info.width,
                                                 info.height) - origin;
    
    DiscretePoint2D zero;
    std::cout << -origin <<std::endl;
    std::cout << end_of_map <<std::endl;
    std::cout << zero <<std::endl;
    std::cout << "{" << info.width << ',' << info.height << '}' << std::endl;
    
    nav_msgs::OccupancyGrid map_msg(map_msg0);

    for (pnt.y = -origin.y; pnt.y < end_of_map.y; ++pnt.y) {
        for (pnt.x = -origin.x; pnt.x < end_of_map.x; ++pnt.x) {
            
            double value = static_cast<double>(map_0[pnt]);
            int cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
            map_msg0.data.push_back(cell_value);

            const VinyDSCell &cell0 = dynamic_cast<const VinyDSCell &>(map_0[pnt]);
            const TBM &el0 = cell0.belief();
            const VinyDSCell &cell1 = dynamic_cast<const VinyDSCell &>(map_1[pnt]);
            const TBM &el1 = cell1.belief();
            TBM res = conjunctive(el0,el1);

            if(res != el0 && res != el1 )
            {
                std::cout << pnt << "; " << el0 << ", " << el1 << ", " << res << std::endl;
            }
            value = static_cast<double>(TBM_to_O(res));
            cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
            map_msg.data.push_back(cell_value);         
            
        }
    }

    // map.save_state_to_file("/home/dmo/Documents/diplom/dumps/merged_test");
    // std::cout << "merged" << std::endl;
    pub.publish(map_msg);
    ros::spinOnce();
    pub0.publish(map_msg0);
    ros::spinOnce();
    
//---------------------------------

    nav_msgs::OccupancyGrid map_msg1;

    map_msg1.header.frame_id = tf_frame;
    map_msg1.info.map_load_time = ros::Time::now();
    map_msg1.info.width = map_1.width();
    map_msg1.info.height = map_1.height();
    map_msg1.info.resolution = map_1.scale();
    
    // move map to the middle
    nav_msgs::MapMetaData &info1 = map_msg1.info;
    DiscretePoint2D origin1 = map_1.origin();
    info1.origin.position.x = -info1.resolution * origin1.x;
    info1.origin.position.y = -info1.resolution * origin1.y;
    info1.origin.position.z = 0;
    map_msg1.data.reserve(info1.height * info1.width);
    DiscretePoint2D end_of_map_1 = DiscretePoint2D(info1.width, info1.height) - origin1;

    for (pnt.y = -origin1.y; pnt.y < end_of_map_1.y; ++pnt.y) {
        for (pnt.x = -origin1.x; pnt.x < end_of_map_1.x; ++pnt.x) {
            double value = static_cast<double>(map_1[pnt]);
            int cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
            map_msg1.data.push_back(cell_value);
        }
    }
    pub1.publish(map_msg1);
    ros::spinOnce();

    return 0;
}