#include <fstream>
#include <iterator>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "../core/maps/plain_grid_map.h"
#include "../slams/viny/viny_grid_cell.h"

#include <thread>
#include <future>

#include <set>

constexpr size_t THREAD_NUM = 3;

struct _thread_info {
    const cv::Mat &img_first;
    const cv::Mat &img_second;
    float count_of_features;
    float ratio_thresh;
    float scale_factor;
    _thread_info(const cv::Mat &img1, const cv::Mat &img2, float cof, float rt, float sf) : 
        img_first(img1), img_second(img2), count_of_features(cof), ratio_thresh(rt), scale_factor(sf){}
};

struct _thread_return {
    std::vector<cv::KeyPoint> kp_first;
    std::vector<cv::KeyPoint> kp_second;
    cv::Mat descriptors_first;
    cv::Mat descriptors_second;
    std::vector<cv::DMatch> good_matches;
    _thread_return(std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2, 
        cv::Mat d1, cv::Mat d2, std::vector<cv::DMatch> gm):
            kp_first(kp1), kp_second(kp2), descriptors_first(d1), descriptors_second(d2), good_matches(gm){}
};

void get_mathces(_thread_info info, std::promise<_thread_return> ret);

int main(int  argc, char **argv) {
    auto gmp = MapValues::gmp;
     const cv::String keys =
        "{first_map            | /home/dmo/Documents/diplom/dumps/compressed_dump_2_11.txt | first dump for load}"
        "{second_map           | /home/dmo/Documents/diplom/dumps/2_floor_31.txt | second dump for load}"
        "{count_of_features    | 500               | count of faetures to extract  }"
        "{scale_factor         | 1.2               | scale factor }"
        "{ratio_thresh         | 0.7               | filter matches using the Lowe's ratio test }"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
    std::string first_filename = parser.get<cv::String>("first_map");
    std::string second_filename = parser.get<cv::String>("second_map");
    int count_of_features = parser.get<int>("count_of_features");
    const float scale_factor = parser.get<float>("scale_factor");
    const float ratio_thresh = parser.get<float>("ratio_thresh");;


    std::ifstream in(first_filename);
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());

    UnboundedPlainGridMap first_map = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    first_map.load_state(file_content);
    std::array<cv::Mat,3> first_parts_maps = first_map.get_3_maps();
    
    in = std::ifstream(second_filename);
    file_content = std::vector<char>((std::istreambuf_iterator<char>(in)),
                                        std::istreambuf_iterator<char>());
    
    UnboundedPlainGridMap second_map = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    second_map.load_state(file_content);
    std::array<cv::Mat,3> second_parts_maps = second_map.get_3_maps();
    
    {
        // cv::imwrite("/home/dmo/Documents/diplom/pictures/map_1_occ.jpg", first_parts_maps.at(0));
        // cv::imwrite("/home/dmo/Documents/diplom/pictures/map_2_occ.jpg", second_parts_maps.at(0));
        // cv::imwrite("/home/dmo/Documents/diplom/pictures/map_1_emp.jpg", first_parts_maps.at(1));
        // cv::imwrite("/home/dmo/Documents/diplom/pictures/map_2_emp.jpg", second_parts_maps.at(1));
        // cv::imwrite("/home/dmo/Documents/diplom/pictures/map_1_unk.jpg", first_parts_maps.at(2));
        // cv::imwrite("/home/dmo/Documents/diplom/pictures/map_2_unk.jpg", second_parts_maps.at(2));
        // cv::imshow("occupancy_1", first_parts_maps.at(0));
        // cv::imshow("empty_1", first_parts_maps.at(1));
        // cv::imshow("unknown_1", first_parts_maps.at(2));
        // cv::imshow("occupancy_2", second_parts_maps.at(0));
        // cv::imshow("empty_2", second_parts_maps.at(1));
        // cv::imshow("unknown_2", second_parts_maps.at(2));
        // cv::waitKey(0);
    }

    std::promise<_thread_return> occ_promise, emp_promise, unk_promise;
    std::future<_thread_return> occ_future = occ_promise.get_future(),
                                emp_future = emp_promise.get_future(),
                                unk_future = unk_promise.get_future();

    _thread_info occ_ti(first_parts_maps.at(0), second_parts_maps.at(0), count_of_features, ratio_thresh, scale_factor);
    _thread_info emp_ti(first_parts_maps.at(1), second_parts_maps.at(1), count_of_features, ratio_thresh, scale_factor);
    _thread_info unk_ti(first_parts_maps.at(2), second_parts_maps.at(2), count_of_features, ratio_thresh, scale_factor);
    
    std::thread occ_thread(get_mathces, occ_ti, std::move(occ_promise));
    std::thread emp_thread(get_mathces, emp_ti, std::move(emp_promise));
    std::thread unk_thread(get_mathces, unk_ti, std::move(unk_promise));

    _thread_return occ_res = occ_future.get();
    _thread_return emp_res = emp_future.get();
    _thread_return unk_res = unk_future.get();
    occ_thread.join();
    emp_thread.join();
    unk_thread.join();

    std::cout << occ_res.good_matches.size() << std::endl
              << emp_res.good_matches.size() << std::endl
              << unk_res.good_matches.size() << std::endl;

    //-- Draw matches
    cv::Mat occ_img_matches, emp_img_matches, unk_img_matches;
    
    cv::drawMatches( first_parts_maps.at(0), occ_res.kp_first, second_parts_maps.at(0), occ_res.kp_second, 
                 occ_res.good_matches, occ_img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), 
                 std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cv::drawMatches( first_parts_maps.at(1), emp_res.kp_first, second_parts_maps.at(1), emp_res.kp_second, 
                 emp_res.good_matches, emp_img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), 
                 std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cv::drawMatches( first_parts_maps.at(2), unk_res.kp_first, second_parts_maps.at(2), unk_res.kp_second, 
                 unk_res.good_matches, unk_img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), 
                 std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cv::imshow("Occupancy", occ_img_matches );
    cv::imshow("Empty", emp_img_matches );
    cv::imshow("Unknown", unk_img_matches );

    cv::waitKey();
    return 0;
}

void get_mathces(_thread_info info, std::promise<_thread_return> ret){
    cv::Ptr<cv::ORB> detector = cv::ORB::create( info.count_of_features, info.scale_factor );

    std::vector<cv::KeyPoint> kp_first, kp_second;
	cv::Mat descriptors_first, descriptors_second;

	detector->detectAndCompute( info.img_first, cv::noArray(), kp_first, descriptors_first );
    detector->detectAndCompute( info.img_second, cv::noArray(), kp_second, descriptors_second );

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);

    std::vector< std::vector<cv::DMatch> > knn_matches;

    // std::cout << std::this_thread::get_id() << ' ' 
    //     << info.descriptors_first->size() << ' ' << info.descriptors_second->size() << std::endl;
    try{
        matcher->knnMatch( descriptors_first, descriptors_second, knn_matches, 2 );
    }
    catch(cv::Exception &e){
        std::cout << e.what() << std::endl;
    }

    //-- Filter matches using the Lowe's ratio test
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < info.ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    _thread_return to_return(kp_first,kp_second, descriptors_first, descriptors_second, good_matches);
    std::cout << good_matches.size() << std::endl;
    ret.set_value(to_return);
}


// struct KPComp {
//     bool operator()(const cv::KeyPoint &k1, const cv::KeyPoint &k2){
//         return k1.pt.y < k2.pt.y ? true : ( k1.pt.y > k2.pt.y ? false : k1.pt.x < k2.pt.x);
//     }
// };
// bool operator==(const cv::KeyPoint &k1, const cv::KeyPoint &k2){
//     return k1.pt == k2.pt;
// }

// template <typename T>
// void findDuplicates(std::vector<T> & vecOfElements, std::map<T, int, KPComp> & countMap)
// {
//     for (auto & elem : vecOfElements)
//     {
//         auto result = countMap.insert(std::pair<T, int>(elem, 1));
//         if (result.second == false)
//             result.first->second++;
//     }
//     for (auto it = countMap.begin() ; it != countMap.end() ;)
//     {
//         if (it->second == 1)
//             it = countMap.erase(it);
//         else
//             it++;
//     }
// }