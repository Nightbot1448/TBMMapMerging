#ifndef SLAM_CTOR_CORE_PLAIN_GRID_MAP_H_INCLUDED
#define SLAM_CTOR_CORE_PLAIN_GRID_MAP_H_INCLUDED

#include <cmath>
#include <vector>
#include <memory>
#include <cassert>
#include <algorithm>
#include <array>

#include <typeinfo>
#include <iostream>

#include "grid_map.h"
#include "../../slams/viny/viny_grid_cell.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>


class PlainGridMap : public GridMap {
public:
	// TODO: cp, mv ctors, dtor
	PlainGridMap(std::shared_ptr<VinyDSCell> prototype,
							 const GridMapParams& params = MapValues::gmp)
		: GridMap{prototype, params}, _cells(GridMap::height()) {
		for (auto &row : _cells) {
			row.reserve(GridMap::width());
			for (int i = 0; i < GridMap::width(); i++) {
					row.push_back(prototype->clone_viny());
			}
		}
	}

	const VinyDSCell &operator[](const Coord& c) const override {
		auto coord = external2internal(c);
		assert(has_internal_cell(coord));
		return cell_internal(coord);
	}

protected: // fields

	const VinyDSCell& cell_internal(const Coord& ic) const {
		return *_cells[ic.y][ic.x];
	}

	std::vector<std::vector<std::unique_ptr<VinyDSCell>>> _cells;
};

/* Unbounded implementation */

class UnboundedPlainGridMap : public PlainGridMap {
private: // fields
	static constexpr double Expansion_Rate = 1.2;
public: // methods
	UnboundedPlainGridMap(std::shared_ptr<VinyDSCell> prototype,
												const GridMapParams &params = MapValues::gmp)
		: PlainGridMap{prototype, params}
			, _origin{GridMap::origin()}, _unknown_cell{prototype->clone_viny()} {}

	void clone_other_map_properties(const UnboundedPlainGridMap &other){
		set_width(other.width());
		set_height(other.height());
		set_scale(other.scale());
		// _origin = other._origin;
		// _unknown_cell = other.new_cell();
	}

	void update(const Coord &area_id, const AreaOccupancyObservation &aoo) override {
		ensure_inside(area_id);
		PlainGridMap::update(area_id, aoo);
	}

	// void setCell(const Coord &area_id, const VinyDSCell &cell){
	void setCell(const Coord &area_id, VinyDSCell *cell){
		ensure_inside(area_id);
		auto ic = external2internal(area_id);
		// *(_cells[ic.y][ic.x]) = cell;
		_cells[ic.y][ic.x].release();
		_cells[ic.y][ic.x].reset(cell);
	}

	void reset(const Coord &area_id, const VinyDSCell &new_area) {
		ensure_inside(area_id);
		auto ic = external2internal(area_id);
			_cells[ic.y][ic.x].reset(new_area.clone_viny().release());
		//PlainGridMap::reset(area_id, new_area);
	}

	const VinyDSCell &operator[](const Coord& ec) const override {
		auto ic = external2internal(ec);
		if (!PlainGridMap::has_internal_cell(ic)) { return *_unknown_cell; }
		return PlainGridMap::cell_internal(ic);
	}

	Coord origin() const override { return _origin; }

	bool has_cell(const Coord &) const override { return true; }

	std::vector<char> save_state() const override {
		auto w = width(), h = height();
		size_t map_size_bytes = w * h * _unknown_cell->serialize().size();

		Serializer s(sizeof(GridMapParams) + sizeof(Coord) + map_size_bytes);
		s << h << w << scale() << origin().x << origin().y;

		Serializer ms(map_size_bytes);
		for (auto &row : _cells) {
			for (auto &cell : row) {
				ms.append(cell->serialize());
			}
		}
	#ifdef COMPRESSED_SERIALIZATION
		s.append(ms.compressed());
	#else
		s.append(ms.result());
	#endif
		return s.result();
	}

public:
	virtual cv::Mat convert_to_grayscale_img() const {
		int w = width();
		int h = height();
		auto origin_ = origin();
		// TODO: remove inv_start && inv_before_end
		cv::Mat_<uchar> map_img(h,w);
		for(int i=0; i<h; i++){
			for(int j=0; j<w; j++){
				map_img(i,j) = static_cast<uchar>(
						(1 - operator[](Coord(j-origin_.x, i-origin_.y)).occupancy().prob_occ) * 255);
			}   
		}
		return map_img;
	}

	virtual std::array<cv::Mat, 3> get_3_maps() const {
		int w = width();
		int h = height();
		auto origin_ = origin();
		cv::Mat_<uchar> occ_map(h,w);
		cv::Mat_<uchar> emp_map(h,w);
		cv::Mat_<uchar> unk_map(h,w);
		
		for(int i=0; i<h; i++){
			for(int j=0; j<w; j++){
				auto cell_belief = operator[](Coord(j-origin_.x, i-origin_.y)).belief();
				occ_map(i, j) = static_cast<uchar>(cell_belief.occupied() * 255);
				emp_map(i, j) = static_cast<uchar>(cell_belief.empty() * 255);
				// HACK: inversed unknown map
				unk_map(i, j) = static_cast<uchar>((1-cell_belief.unknown()) * 255);
			}
		}
		
		std::array<cv::Mat, 3> result;
		result.at(0) = occ_map;
		result.at(1) = emp_map;
		result.at(2) = unk_map;
		return result;
	}

	virtual std::array<cv::Mat, 4> get_maps_grs_ofu() const {
		std::array<cv::Mat, 3> ofu = get_3_maps();
		std::array<cv::Mat, 4> result;
		result.at(0) = convert_to_grayscale_img();
		result.at(1) = ofu.at(0);
		result.at(2) = ofu.at(1);
		result.at(3) = ofu.at(2);
		return result;
	}

	void save_state_to_file(const std::string& _base_fname = "/home/dmo/Documents/diplom/dumps/tmp_") const override {
		auto w = width(), h = height();
		size_t map_size_bytes = w * h * _unknown_cell->serialize().size() * sizeof(char);

		Serializer s(sizeof(GridMapParams) + sizeof(Coord) + map_size_bytes);
		s << h << w << scale() << origin().x << origin().y;

		Serializer ms(map_size_bytes);
		for (auto &row : _cells) {
				for (auto &cell : row) {
					ms.append(cell->serialize());
				}
		}
#ifdef COMPRESSED_SERIALIZATION
		auto tmp = ms.compressed();
		s.append(tmp);
#else
		s.append(ms.result());
#endif
		static size_t _id = 0;
		// ROS_INFO("save %lu dump", _id);
		std::string res_name;
		if (_base_fname == "/home/dmo/Documents/diplom/dumps/tmp_")
			res_name = _base_fname + std::to_string(_id) + ".txt";
		else
			res_name = _base_fname;
		std::ofstream dst = std::ofstream{res_name, std::ios::out};
		std::vector<char> result = s.result();
		dst.write(result.data(), result.size() * sizeof(*(result.data())));
		dst.close();
		_id++;
	}

	void load_state(const std::vector<char>& data) override {
		decltype(width()) w, h;
		decltype(scale()) s;

		Deserializer d(data);
		d >> h >> w >> s >> _origin.x >> _origin.y;


		set_width(w);
		set_height(h);
		set_scale(s);
	#ifdef COMPRESSED_SERIALIZATION
		// ROS_INFO("start decompress");
		std::vector<char> map_data = Deserializer::decompress(
				data.data() + d.pos(), data.size() - d.pos(),
				w * h * _unknown_cell->serialize().size());
		size_t pos = 0;
		// ROS_INFO("finish decompress");
	#else
		const std::vector<char> &map_data = data;
		size_t pos = d.pos();
	#endif
		_cells.clear();
		_cells.resize(h);
		for (auto &row : _cells) {
			row.reserve(w);
			for (int i = 0; i < w; ++i) {
				auto cell = new_cell();
				pos = cell->deserialize(map_data, pos);
				row.push_back(std::move(cell));
			}
		}
		// std::cout << "Size: " << _cells.size() << 'x' << _cells.at(0).size() << std::endl;
	}

	virtual void crop_by_bounds() {
		int min_x = width(), min_y = height(), max_x = 0, max_y = 0;

		for(int row_id = 0; row_id < height(); row_id++){
			for(int cell_id = 0; cell_id < width(); cell_id++){
				auto &cell = _cells[row_id][cell_id];
				if(cell->belief().unknown() != 1.0f){
					if (min_x > cell_id)
						min_x = cell_id;
					if (max_x < cell_id)
						max_x = cell_id;
					if (min_y > row_id)
						min_y = row_id;
					if (max_y < row_id)
						max_y = row_id;
				}
			}
		}

		// std::cout << "result bounds: {" << min_x << ' ' << min_y << "}; {" << max_x << ' ' << max_y << '}' << std::endl; 

		min_x = std::floor(min_x/50.0f)*50;
		min_y = std::floor(min_y/50.0f)*50;
		max_x = std::ceil(max_x/50.0f)*50;
		max_y = std::ceil(max_y/50.0f)*50;

		int new_width = max_x - min_x;
		int new_height = max_y - min_y;
		Coord new_origin(new_width/2, new_height/2);
		for (int j=0; j<new_height; j++) {
			for (int i = 0; i < new_width; ++i) {
				_cells[j][i].reset(_cells[min_y+j][min_x+i].release());
			}
			_cells.at(j).resize(new_width);
		}
		_cells.resize(new_height);

		std::cout << "new size: " << new_width << ' ' << new_height << std::endl;
		std::cout << "new origin: " << new_origin << std::endl;

		set_height(new_height);
		set_width(new_width);
		_origin = new_origin;
	}

	/* angle in radian */
	virtual std::shared_ptr<UnboundedPlainGridMap> rotate(double angle, DiscretePoint2D axis = {0,0}){
		int width = this->width();
		int height = this->height();
		auto gmp = MapValues::gmp;
		auto gmp_rot = GridMapParams{
			int(std::ceil(width*std::abs(std::cos(angle)) + height*std::abs(std::sin(angle)))), 
			int(std::ceil(width*std::abs(std::sin(angle)) + height*std::abs(std::cos(angle)))),
			gmp.meters_per_cell};

		auto rotated_map = std::make_shared<UnboundedPlainGridMap>(UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp_rot));
		
		double inv_sin_angle = std::sin(-angle * CV_PI / 180);
		double inv_cos_angle = std::cos(-angle * CV_PI / 180);

		DiscretePoint2D new_origin = rotated_map->origin();
		DiscretePoint2D new_end_of_map = DiscretePoint2D(rotated_map->width(),
													rotated_map->height()) - new_origin;
		DiscretePoint2D pnt;
		for(pnt.y = -new_origin.y; pnt.y < new_end_of_map.y; ++pnt.y) {
			for (pnt.x = -new_origin.x; pnt.x < new_end_of_map.x; ++pnt.x) {
				DiscretePoint2D cell_pnt;
				cell_pnt.x = std::round((pnt.x-axis.x) * inv_cos_angle - (pnt.y - axis.y) * inv_sin_angle);
				cell_pnt.y = std::round((pnt.x-axis.x) * inv_sin_angle + (pnt.y - axis.y) * inv_cos_angle);
				cell_pnt += axis;
				const GridCell &map_value = this->operator[](cell_pnt);
				rotated_map->setCell(pnt, new VinyDSCell(dynamic_cast<const VinyDSCell &>(map_value)));
			}
		}
		return rotated_map;
	}

    virtual std::shared_ptr<UnboundedPlainGridMap> shift(DiscretePoint2D shift_ = {0,0}){
        int width = this->width();
        int height = this->height();
        auto gmp = MapValues::gmp;

        auto shifted_map = std::make_shared<UnboundedPlainGridMap>(UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp));
        DiscretePoint2D pnt;
        DiscretePoint2D end_of_map = DiscretePoint2D(width, height) - _origin;
        for(pnt.y = -_origin.y; pnt.y < end_of_map.y; ++pnt.y) {
            for (pnt.x = -_origin.x; pnt.x < end_of_map.x; ++pnt.x) {
                const GridCell &map_value = operator[](pnt);
                DiscretePoint2D new_cell_pnt(pnt.x + shift_.x, pnt.y + shift_.y );
                shifted_map->setCell(new_cell_pnt, new VinyDSCell(dynamic_cast<const VinyDSCell &>(map_value)));
            }
        }
        return shifted_map;
    }

//    virtual std::shared_ptr<UnboundedPlainGridMap> apply_transform(const cv::Mat &transform){
////	    this->crop_by_bounds();
//        cv::Mat inversed_transform;
//        cv::Mat original;
//        cv::invertAffineTransform(transform, inversed_transform);
////        cv::invertAffineTransform(inversed_transform, original);
//        std::cout << "transform" << std::endl << transform << std::endl;
//        std::cout << "inversed" << std::endl << inversed_transform << std::endl;
////        std::cout << "inversed^2" << std::endl << original << std::endl;
//
//        auto gmp = MapValues::gmp;
//        std::vector<int> new_bounds = get_transformed_bounds(transform);
//
//        std::cout << "new bounds: [" << new_bounds.at(0) << ", " <<new_bounds.at(1)
//            << "]; -> [" << new_bounds.at(2) << ", " << new_bounds.at(3) << ']' << std::endl;
//        GridMapParams gmp_modified{(int)std::ceil((new_bounds.at(2) - new_bounds.at(0))/100.0)*100,
//                                   (int)std::ceil((new_bounds.at(3) - new_bounds.at(1))/100.0)*100,
//                                   gmp.meters_per_cell};
//
//        auto transformed_map = std::make_shared<UnboundedPlainGridMap>(UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp_modified));
//        DiscretePoint2D new_origin = transformed_map->origin();
////        DiscretePoint2D new_origin = DiscretePoint2D((int)std::ceil(-new_bounds.at(0)/50.0)*50, (int)std::ceil(-new_bounds.at(1)/50.0)*50);
//        DiscretePoint2D new_end_of_map = DiscretePoint2D(transformed_map->width(),
//                                                         transformed_map->height()) - new_origin;
////        transformed_map->_origin = new_origin;
//        std::cout << "new size:" << transformed_map->width() << ' ' << transformed_map->height() << std::endl;
//        std::cout << "new origin:" << transformed_map->origin() << std::endl;
//        DiscretePoint2D pnt;
//        for(pnt.y = -new_origin.y; pnt.y < new_end_of_map.y; ++pnt.y) {
//            for (pnt.x = -new_origin.x; pnt.x < new_end_of_map.x; ++pnt.x) {
//                cv::Mat point{(double)pnt.x,(double)pnt.y,1.0};
//                cv::Mat base_pt = inversed_transform * point;
//                DiscretePoint2D cell_pnt((int)base_pt.at<double>(0), (int)base_pt.at<double>(1));
//                const VinyDSCell &map_value = this->operator[](cell_pnt);
//                transformed_map->setCell(pnt, new VinyDSCell(map_value));
//            }
//        }
//
//        return transformed_map;
//	}

    // mat 2x3 [R|t]
    virtual std::shared_ptr<UnboundedPlainGridMap> apply_transform(const cv::Mat &transform, DiscretePoint2D &changed_size){
        std::cout << "transform" << std::endl << transform << std::endl;
        auto gmp = MapValues::gmp;
        auto new_bounds = get_transformed_bounds(transform);
        auto gmp_mod = GridMapParams{new_bounds.at(2)-new_bounds.at(0), new_bounds.at(3)-new_bounds.at(1), gmp.meters_per_cell};
        auto transformed_map = std::make_shared<UnboundedPlainGridMap>(UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), gmp_mod));
        transformed_map->_origin = this->_origin+DiscretePoint2D(new_bounds.at(0), new_bounds.at(1));
        std::cout << "origin: " << transformed_map->origin() << std::endl;
        std::cout << "old sz: " << transformed_map->width() << ' ' << transformed_map->height() << std::endl;
        DiscretePoint2D inv_transformed_origin = -transformed_map->_origin;
        DiscretePoint2D pnt;
//        cv::Mat inverted_transform;
//        cv::invertAffineTransform(transform, inverted_transform);
        for(pnt.y = 0; pnt.y < height(); ++pnt.y) {
            for (pnt.x = 0; pnt.x < width(); ++pnt.x) {
                cv::Mat point{(double)pnt.x,(double)pnt.y,1.0};
                cv::Mat base_pt = transform * point;
                DiscretePoint2D cell_pnt((int)std::round(base_pt.at<double>(0)),
                        (int)std::round(base_pt.at<double>(1)));
                cell_pnt += inv_transformed_origin;
                const VinyDSCell &map_value = this->operator[](pnt-_origin);
                transformed_map->setCell(cell_pnt, new VinyDSCell(map_value));
            }
        }
        std::cout << "new origin: " << transformed_map->origin() << std::endl;
        std::cout << "new sz: " << transformed_map->width() << ' ' << transformed_map->height() << std::endl;
        changed_size = DiscretePoint2D(transformed_map->width()-width(), transformed_map->height() - height());
        return transformed_map;
    }

    std::shared_ptr<UnboundedPlainGridMap> full_merge(std::shared_ptr<UnboundedPlainGridMap> other, DiscretePoint2D changed_size){
        DiscretePoint2D pnt;
        std::cout << "this sz & origin: {" << this->width() << ' ' << this->height() << "}; " << this->_origin << std::endl;
        std::cout << "other sz & origin: {" << other->width() << ' ' << other->height() << "}; " << other->_origin << std::endl;
        std::pair<int,int> width_minmax = std::minmax(this->width(), other->width());
        std::pair<int,int> height_minmax = std::minmax(this->height(), other->height());
        auto gmp = MapValues::gmp;
        auto merged_gmp = GridMapParams{width_minmax.second, height_minmax.second, gmp.meters_per_cell};
        auto merged_map = std::make_shared<UnboundedPlainGridMap>(UnboundedPlainGridMap(std::make_shared<VinyDSCell>(), merged_gmp));
        merged_map->set_width(width_minmax.second);
        merged_map->set_height(height_minmax.second);
        merged_map->_origin = DiscretePoint2D(width_minmax.second/2, height_minmax.second/2);
        std::cout << "merged sz & origin: {" << merged_map->width() << ' ' << merged_map->height() << "}; " << merged_map->_origin << std::endl;
//        DiscretePoint2D merged_origin = merged_map->_origin;
        for (pnt.y = 0; pnt.y < height_minmax.first; ++pnt.y) {
            for (pnt.x = 0; pnt.x < width_minmax.first; ++pnt.x) {
//                std::cout << pnt.x << std::endl;
                const TBM &el0 = this->_cells[pnt.y][pnt.x]->belief();
                const TBM &el1 = other->_cells[pnt.y][pnt.x]->belief();
                TBM res = disjunctive(el0,el1);
                Occupancy occ = TBM_to_O(res);
                merged_map->_cells[pnt.y][pnt.x].reset(new VinyDSCell(occ, res));
            }
        }
//        std::cout << std::endl;
        return merged_map;
	}

private:
    std::vector<int> get_transformed_bounds(const cv::Mat &transform){
        std::vector<cv::Mat> bounds;
        double width_ = static_cast<double>(width());
        double height_ = static_cast<double>(height());
        cv::Mat mat(3,1,CV_64F);
        std::vector<double> data_{0, 0, 1};
        std::memcpy(mat.data, data_.data(), data_.size()*sizeof(double));
        bounds.push_back(mat.clone());
        data_ = std::vector<double>{0, height_, 1};
        std::memcpy(mat.data, data_.data(), data_.size()*sizeof(double));
        bounds.push_back(mat.clone());
        data_ = std::vector<double>{width_, 0, 1};
        std::memcpy(mat.data, data_.data(), data_.size()*sizeof(double));
        bounds.push_back(mat.clone());
        data_ = std::vector<double>{width_, height_, 1};
        std::memcpy(mat.data, data_.data(), data_.size()*sizeof(double));
        bounds.push_back(mat);
        std::vector<cv::Mat> new_bounds;
//        std::cout << "points: "<< std::endl;
        for(auto &point: bounds){
            new_bounds.push_back(transform * point);
//            std::cout << "{" << point.at<double>(0) << ", " << point.at<double>(1) << "} -> {"
//                    << new_bounds.back().at<double>(0) << ", " << new_bounds.back().at<double>(1) << "}" << std::endl;
        }
        double max_x=0, min_x=0, max_y=0, min_y=0;
        for(auto &point: new_bounds){
            double x = point.at<double>(0);
            double y = point.at<double>(1);
            if(x > max_x)
                max_x = x;
            if(x < min_x)
                min_x = x;
            if(y > max_y)
                max_y = y;
            if(y < min_y)
                min_y = y;
        }
        return std::vector<int>{static_cast<int>(std::floor(min_x/50.0)*50), static_cast<int>(std::floor(min_y/50.)*50),
                                static_cast<int>(std::ceil(max_x/50.)*50), static_cast<int>(std::ceil(max_y/50.)*50)};
	}

protected: // methods

	bool ensure_inside(const Coord &c) {
		auto coord = external2internal(c);
		if (PlainGridMap::has_internal_cell(coord)) return false;

		unsigned w = width(), h = height();
		unsigned prep_x = 0, app_x = 0, prep_y = 0, app_y = 0;
		std::tie(prep_x, app_x) = determine_cells_nm(0, coord.x, w);
		std::tie(prep_y, app_y) = determine_cells_nm(0, coord.y, h);

		unsigned new_w = prep_x + w + app_x, new_h = prep_y + h + app_y;
		#define UPDATE_DIM(dim, elem)                                    \
			if (dim < new_##dim && new_##dim < Expansion_Rate * dim) {     \
				double scale = prep_##elem / (new_##dim - dim);              \
				prep_##elem += (Expansion_Rate * dim - new_##dim) * scale;   \
				new_##dim = Expansion_Rate * dim;                            \
				app_##elem = new_##dim - (prep_##elem + dim);                \
			}

		UPDATE_DIM(w, x);
		UPDATE_DIM(h, y);
		#undef UPDATE_DIM

		// PERFORMANCE: _cells can be reused
		std::vector<std::vector<std::unique_ptr<VinyDSCell>>> new_cells{new_h};
		for (size_t y = 0; y != new_h; ++y) {
//        std::generate_n(std::back_inserter(new_cells[y]), new_w, [this](){ return this->_unknown_cell->clone(); });
				std::generate_n(std::back_inserter(new_cells[y]), new_w, [this](){ return this->_unknown_cell->clone_viny(); });
			if (y < prep_y || prep_y + h <= y) { continue; }

			std::move(_cells[y - prep_y].begin(), _cells[y - prep_y].end(),
								&new_cells[y][prep_x]);
		}

		std::swap(_cells, new_cells);
		set_height(new_h);
		set_width(new_w);
		_origin += Coord(prep_x, prep_y);

		assert(PlainGridMap::has_cell(c));
		return true;
	}

	std::tuple<unsigned, unsigned> determine_cells_nm(
		int min, int val, int max) const {
		assert(min <= max);
		unsigned prepend_nm = 0, append_nm = 0;
		if (val < min) {
			prepend_nm = min - val;
		} else if (max <= val) {
			append_nm = val - max + 1;
		}
		return std::make_tuple(prepend_nm, append_nm);
	}

private: // fields
	Coord _origin;
	std::shared_ptr<VinyDSCell> _unknown_cell;
};

#endif
