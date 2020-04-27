#include "../map_utils_headers/descriptorsComarator.h"
#include <chrono>
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
    size_t count_of_features = parser.get<size_t>("count_of_features");
    const float scale_factor = parser.get<float>("scale_factor");
    const float ratio_thresh = parser.get<float>("ratio_thresh");


    std::ifstream in(first_filename);
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());

    UnboundedPlainGridMap first_map = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    first_map.load_state(file_content);
    std::array<cv::Mat,4> first_parts_maps = first_map.get_maps_grs_ofu();
    
    in = std::ifstream(second_filename);
    file_content = std::vector<char>((std::istreambuf_iterator<char>(in)),
                                        std::istreambuf_iterator<char>());
    
    UnboundedPlainGridMap second_map = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    second_map.load_state(file_content);
    std::array<cv::Mat,4> second_parts_maps = second_map.get_maps_grs_ofu();

    
    std::chrono::time_point<std::chrono::system_clock> before_act = std::chrono::system_clock::now();


    std::vector<cv::KeyPoint> kp_first, kp_second;
    cv::Mat d_first_prob, d_second_prob,
            d_first_occ, d_first_emp, d_first_unk,
            d_second_occ,d_second_emp,d_second_unk;

    
    cv::Ptr<cv::ORB> detector = cv::ORB::create( count_of_features, scale_factor );

    detector->detectAndCompute(first_parts_maps.at(0), cv::noArray(), kp_first, d_first_prob);
    detector->compute(first_parts_maps.at(1), kp_first, d_first_occ);
    detector->compute(first_parts_maps.at(2), kp_first, d_first_emp);
    detector->compute(first_parts_maps.at(3), kp_first, d_first_unk);
    
    detector->detectAndCompute(second_parts_maps.at(0), cv::noArray(), kp_second, d_second_prob);
    detector->compute(second_parts_maps.at(1), kp_second, d_second_occ);
    detector->compute(second_parts_maps.at(2), kp_second, d_second_emp);
    detector->compute(second_parts_maps.at(3), kp_second, d_second_unk);
    

    std::vector<cv::DMatch> good_matches_prob = get_good_matches(d_first_prob, d_second_prob, ratio_thresh);

    _thread_change_descriptor info(  first_parts_maps.at(0), second_parts_maps.at(0), kp_first, kp_second,
                        d_first_occ, d_first_emp, d_first_unk, d_second_occ, d_second_emp, d_second_unk,
                        count_of_features, ratio_thresh, scale_factor );

    std::promise<_thread_change_descriptor_return>  concatinate_promise, land_promise, lor_promise, lxor_promise;
    std::future<_thread_change_descriptor_return>   concatinate_future = concatinate_promise.get_future(),
                                                    land_future = land_promise.get_future(),
                                                    lor_future  = lor_promise.get_future(),
                                                    lxor_future  = lxor_promise.get_future();
    
    std::thread concatinate_thread(parallel_new_descriptors_concatinate, info, &concatinate_promise);
    std::thread land_thread(new_descriptors_land, info, &land_promise);
    std::thread lor_thread(new_descriptors_lor, info, &lor_promise);
    std::thread lxor_thread(new_descriptors_lxor, info, &lxor_promise);

    _thread_change_descriptor_return    concatinate_ret = concatinate_future.get(),
                                        land_ret = land_future.get(),
                                        lor_ret = lor_future.get(),
                                        lxor_ret = lxor_future.get();
    
    concatinate_thread.join();
    land_thread.join();
    lor_thread.join();
    lxor_thread.join();

    std::vector<cv::DMatch> good_matches_concatinate = concatinate_ret.good_matches,
                            good_matches_land = land_ret.good_matches,
                            good_matches_lor = lor_ret.good_matches,
                            good_matches_lxor = lxor_ret.good_matches;

	
    //-- Draw matches
    cv::Mat img_matches_prob, img_matches_conc, img_matches_land, img_matches_lor, img_matches_lxor;
    drawMatches(first_parts_maps.at(0), kp_first, second_parts_maps.at(0), kp_second, 
                good_matches_prob, img_matches_prob, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    drawMatches(first_parts_maps.at(0), kp_first, second_parts_maps.at(0), kp_second, 
                good_matches_concatinate, img_matches_conc, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
    drawMatches(first_parts_maps.at(0), kp_first, second_parts_maps.at(0), kp_second, 
                good_matches_land, img_matches_land, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    drawMatches(first_parts_maps.at(0), kp_first, second_parts_maps.at(0), kp_second, 
                good_matches_lor, img_matches_lor, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
    drawMatches(first_parts_maps.at(0), kp_first, second_parts_maps.at(0), kp_second, 
                good_matches_lxor, img_matches_lxor, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    

    std::chrono::time_point<std::chrono::system_clock> after_act = std::chrono::system_clock::now();
    std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(after_act-before_act).count() << std::endl;


    std::cout << "count of good matches prob: " << good_matches_prob.size() << std::endl;
    std::cout << "count of good matches conc: " << good_matches_concatinate.size() << std::endl;
    std::cout << "count of good matches land: " << good_matches_land.size() << std::endl;
    std::cout << "count of good matches lor: " << good_matches_lor.size() << std::endl;
    std::cout << "count of good matches lxor: " << good_matches_lxor.size() << std::endl;

    cv::imshow("Good Matches prob", img_matches_prob );
    cv::imshow("Good Matches conc", img_matches_conc );
    cv::imshow("Good Matches land", img_matches_land );
    cv::imshow("Good Matches lor",  img_matches_lor  );
    cv::imshow("Good Matches lxor", img_matches_lxor );
    
	int k = 0;
	while (k != 27)
    	k = cv::waitKey();


    {
        // cv::imwrite("/home/dmo/Documents/diplom/pictures/map_1_occ.jpg", first_parts_maps.at(0));
        // cv::imwrite("/home/dmo/Documents/diplom/pictures/map_2_occ.jpg", second_parts_maps.at(0));
        // cv::imwrite("/home/dmo/Documents/diplom/pictures/map_1_emp.jpg", first_parts_maps.at(1));
        // cv::imwrite("/home/dmo/Documents/diplom/pictures/map_2_emp.jpg", second_parts_maps.at(1));
        // cv::imwrite("/home/dmo/Documents/diplom/pictures/map_1_unk.jpg", first_parts_maps.at(2));
        // cv::imwrite("/home/dmo/Documents/diplom/pictures/map_2_unk.jpg", second_parts_maps.at(2));
        // cv::imshow("occupancy_1", first_parts_maps.at(1));
        // cv::imshow("empty_1", first_parts_maps.at(2));
        // cv::imshow("unknown_1", first_parts_maps.at(3));
        // cv::imshow("occupancy_2", second_parts_maps.at(1));
        // cv::imshow("empty_2", second_parts_maps.at(2));
        // cv::imshow("unknown_2", second_parts_maps.at(3));
        // cv::waitKey(0);
    }
    return 0;
}