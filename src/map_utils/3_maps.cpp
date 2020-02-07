#include <fstream>
#include <iterator>

#include <opencv2/highgui.hpp>

#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"


int main(int  argc, char **argv) {
    auto gmp = MapValues::gmp;

    std::string filename_ = "/home/dmo/Documents/diplom/dumps/compressed_dump_2_11.txt";
    
    UnboundedPlainGridMap map = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    
    std::ifstream in(filename_);
    
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());

    map.load_state(file_content);
    // cv::Mat map_img =  map.convert_to_grayscale_img();
    std::array<cv::Mat,3> maps =  map.get_3_maps();

    cv::imshow("occupancy", maps.at(0));
    cv::imshow("empty", maps.at(1));
    cv::imshow("unknown", maps.at(2));
    cv::waitKey(10);
    return 0;
}
