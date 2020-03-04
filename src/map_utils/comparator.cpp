#include "compute_descriptors.h"
#include <cstdlib>

int main(int  argc, char **argv) {
    auto gmp = MapValues::gmp;
     const cv::String keys =
        "{first_map            | /home/dmo/Documents/diplom/dumps/compressed_dump_2_11.txt | first dump for load}"
        "{second_map           | /home/dmo/Documents/diplom/dumps/2_floor_31.txt | second dump for load}"
        "{base_tests_path      | /home/dmo/Documents/diplom/tests/ | base path }"
        "{count_of_features    | 500               | count of faetures to extract }"
        "{scale_factor         | 1.2               | scale factor }"
        "{ratio_thresh         | 0.7               | filter matches using the Lowe's ratio test }"
        "{test_id              | -1                | test id }"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
    std::string first_filename = parser.get<cv::String>("first_map");
    std::string second_filename = parser.get<cv::String>("second_map");
    std::string base_tests_path = parser.get<cv::String>("base_tests_path");
    size_t count_of_features = parser.get<size_t>("count_of_features");
    float scale_factor = parser.get<float>("scale_factor");
    float ratio_thresh = parser.get<float>("ratio_thresh");
    size_t test_id = parser.get<size_t>("test_id");

    // std::cout << '1' << std::endl;
    std::ifstream in(first_filename);
    std::vector<char> file_content((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());

    UnboundedPlainGridMap first_map = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    first_map.load_state(file_content);
    std::array<cv::Mat,4> first_parts_maps = first_map.get_maps_grs_ofu();
    
    // std::cout << '2' << std::endl;
    in = std::ifstream(second_filename);
    file_content = std::vector<char>((std::istreambuf_iterator<char>(in)),
                                        std::istreambuf_iterator<char>());
    
    UnboundedPlainGridMap second_map = UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp);
    second_map.load_state(file_content);
    std::array<cv::Mat,4> second_parts_maps = second_map.get_maps_grs_ofu();
    // std::cout << '3' << std::endl;

    // cv::imshow("occ", second_parts_maps.at(1));
    // cv::imshow("emp", second_parts_maps.at(2));
    // cv::imshow("unk", second_parts_maps.at(3));
    // cv::waitKey(0);

    std::vector<cv::KeyPoint> kp_first_prob, kp_first_conc, kp_second_prob, kp_second_conc;
    cv::Mat d_first_prob, d_second_prob,
            d_first_occ, d_first_emp, d_first_unk,
            d_second_occ,d_second_emp,d_second_unk;

    cv::Ptr<cv::ORB> detector = cv::ORB::create( count_of_features, scale_factor );

    // std::cout << '4' << std::endl;

    detector->detectAndCompute(first_parts_maps.at(0), cv::noArray(), kp_first_prob, d_first_prob);
    kp_first_conc = kp_first_prob;

    detector->compute(first_parts_maps.at(1), kp_first_conc, d_first_occ);
    detector->compute(first_parts_maps.at(2), kp_first_conc, d_first_emp);
    detector->compute(first_parts_maps.at(3), kp_first_conc, d_first_unk);

//    size_t kp_first_prob_sz = kp_first_prob.size();
//    size_t kp_first_conc_sz = kp_first_conc.size();
//    std::cout << "first map(prob, conc): " << kp_first_prob_sz << ' ' << kp_first_conc_sz << std::endl;

    detector->detectAndCompute(second_parts_maps.at(0), cv::noArray(),
            kp_second_prob, d_second_prob);

    kp_second_conc = kp_second_prob;
    detector->compute(second_parts_maps.at(1),
            kp_second_conc, d_second_occ);
    detector->compute(second_parts_maps.at(2),
            kp_second_conc, d_second_emp);
    detector->compute(second_parts_maps.at(3),
            kp_second_conc, d_second_unk);

//    size_t kp_second_prob_sz = kp_second_prob.size();
//    size_t kp_second_conc_sz = kp_second_conc.size();
//    std::cout << "second map(prob, conc): " << kp_second_prob_sz << ' ' << kp_second_conc_sz << std::endl;

    std::vector<cv::DMatch> good_matches_prob = get_good_matches(d_first_prob, d_second_prob, ratio_thresh);

    _thread_change_descriptor info(  first_parts_maps.at(0),
            second_parts_maps.at(0), kp_first_conc, kp_second_conc,
                        d_first_occ, d_first_emp, d_first_unk, d_second_occ,
                        d_second_emp, d_second_unk,
                        count_of_features, ratio_thresh, scale_factor );

    _thread_change_descriptor_return concatinate_ret = new_descriptors_concatinate(info);

    std::vector<cv::DMatch> good_matches_conc = concatinate_ret.good_matches;

    {
        //    kp_first_prob_sz = kp_first_prob.size();
        //    kp_second_prob_sz = kp_second_prob.size();
        //    kp_first_conc_sz = kp_first_conc.size();
        //    kp_second_conc_sz = kp_second_conc.size();

        //    size_t good_matches_prob_sz = good_matches_prob.size();
        //    size_t good_matches_conc_sz = good_matches_conc.size();

        //    std::cout   << kp_first_prob_sz << ' ' << kp_second_prob_sz << std::endl
        //                << kp_first_conc_sz << ' ' << kp_second_conc_sz << std::endl
        //                << good_matches_prob_sz << ' ' << good_matches_conc_sz << std::endl;
    }

    size_t prob_good_dist_sz, conc_good_dist_sz;
    if(get_good_distance_vec_size( kp_first_prob, kp_second_prob,
                                kp_first_conc, kp_second_conc, 
                                good_matches_prob, good_matches_conc, 
                                prob_good_dist_sz, conc_good_dist_sz))
    {
        //-- Draw matches
        cv::Mat img_matches_prob, img_matches_conc;//, img_matches_land, img_matches_lor, img_matches_lxor;
        drawMatches(first_parts_maps.at(0), kp_first_prob, second_parts_maps.at(0), kp_second_prob, 
                    good_matches_prob, img_matches_prob, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        drawMatches(first_parts_maps.at(0), kp_first_conc, second_parts_maps.at(0), kp_second_conc, 
                    good_matches_conc, img_matches_conc, cv::Scalar(0,255,0,0), cv::Scalar(255,0,0,0), 
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        
     
        std::cout << prob_good_dist_sz << ' ' << conc_good_dist_sz << std::endl;

        cv::imshow("prob", img_matches_prob);
        cv::imshow("conc", img_matches_conc);
        cv::waitKey(0);


        if(test_id>=0){
            std::string test_folder = base_tests_path+std::to_string(test_id)+"/";
            cv::imwrite( test_folder + "prob.jpg", img_matches_prob);
            cv::imwrite( test_folder + "conc.jpg", img_matches_conc);
            std::ofstream out(test_folder+ "stat");
            out << first_filename << std::endl 
                << second_filename << std::endl 
                << count_of_features << std::endl 
                << ratio_thresh << std::endl
                << good_matches_prob.size() << ' ' << good_matches_conc.size() << std::endl
                << prob_good_dist_sz << ' ' << conc_good_dist_sz << std::endl;
            out.close();

        }
    }
    return 0;
}