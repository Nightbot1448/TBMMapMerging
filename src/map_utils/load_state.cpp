#include <ros/ros.h>
#include <fstream>
#include <iterator>
#include <string>

#include <nav_msgs/OccupancyGrid.h>

#include <opencv2/highgui.hpp>

#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"

void push_to_map_msg(nav_msgs::OccupancyGrid &msg, const GridCell &cell);

int main(int  argc, char **argv) {
    ros::init(argc, argv, "load_state");

    ros::NodeHandle nh;
    ros::Publisher pub = nh.advertise<nav_msgs::OccupancyGrid>("map", 5);
    std::string tf_frame = "odom_combined";
    auto gmp = MapValues::gmp;

    std::string filename_ = "/home/dmo/Documents/diplom/dumps/compressed_dump_8.txt";
    nh.getParam("/load_state/file", filename_);

    UnboundedPlainGridMap map = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    
    ROS_INFO("filename = %s", filename_.c_str());

    std::ifstream in(filename_);
    
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());

    map.load_state(file_content);


    nav_msgs::OccupancyGrid map_msg;
    map_msg.header.frame_id = tf_frame;
    map_msg.info.map_load_time = ros::Time::now();
    map_msg.info.width = map.width();
    map_msg.info.height = map.height();
    map_msg.info.resolution = map.scale();

    // move map to the middle
    nav_msgs::MapMetaData &info = map_msg.info;
    DiscretePoint2D origin = map.origin();
    map_msg.data.reserve(info.height * info.width);
    DiscretePoint2D pnt;
    DiscretePoint2D end_of_map = DiscretePoint2D(info.width,
                                                 info.height) - origin;
    
    DiscretePoint2D zero;
    std::cout << -origin <<std::endl;
    std::cout << end_of_map <<std::endl;
    std::cout << zero <<std::endl;
    std::cout << "{" << info.width << ',' << info.height << '}' << std::endl;
    
    for (pnt.y = -origin.y; pnt.y < end_of_map.y; ++pnt.y)
        for (pnt.x = -origin.x; pnt.x < end_of_map.x; ++pnt.x)
                push_to_map_msg(map_msg, map[pnt]);

    ros::Rate loop_rate(1000);
    cv::Mat map_img =  map.convert_to_grayscale_img();

    size_t pos_slash = filename_.find_last_of('/');
    size_t pos_point = filename_.find_last_of('.');

    std::string result_filename("/home/dmo/Documents/diplom/pictures/for_rep/desc_compare/");
    result_filename += filename_.substr(pos_slash+1, pos_point) + ".jpg";
    cv::imwrite(result_filename, map_img);

    // while(ros::ok())
    // {
    //     pub.publish(map_msg);
    //     cv::imshow("Map", map_img);
    //     cv::waitKey(1);
    //     ros::spinOnce();
    //     loop_rate.sleep();
    // }

    return 0;
}

void push_to_map_msg(nav_msgs::OccupancyGrid &msg, const GridCell &cell){
    double value = static_cast<double>(cell);
    int cell_value = value == -1 ? -1 : static_cast<int>(value * 100);
    msg.data.push_back(cell_value);
}