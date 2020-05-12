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

private:
	int inv_start = 300, inv_before_end = 200;
public:
	virtual cv::Mat convert_to_grayscale_img() const {
		int w = width();
		int h = height();
		auto origin_ = origin();
		
		cv::Mat_<uchar> map_img(h-inv_start-inv_before_end,w);
		for(int i=inv_start; i<h-inv_before_end; i++){
			for(int j=0; j<w; j++){
				map_img(h-i-inv_before_end-1,j) = static_cast<uchar>(
						(1 - operator[](Coord(j-origin_.x, i-origin_.y)).occupancy().prob_occ) * 255);
			}   
		}
		return map_img;
	}

	virtual std::array<cv::Mat, 3> get_3_maps() const {
		int w = width();
		int h = height();
		auto origin_ = origin();
		cv::Mat_<uchar> occ_map(h-inv_start-inv_before_end,w);
		cv::Mat_<uchar> emp_map(h-inv_start-inv_before_end,w);
		cv::Mat_<uchar> unk_map(h-inv_start-inv_before_end,w);
		
		for(int i=inv_start; i<h-inv_before_end; i++){
			for(int j=0; j<w; j++){
				auto cell_belief = operator[](Coord(j-origin_.x, i-origin_.y)).belief();
				occ_map(h - i - inv_before_end - 1, j) = static_cast<uchar>(cell_belief.occupied() * 255);
				emp_map(h - i - inv_before_end - 1, j) = static_cast<uchar>(cell_belief.empty() * 255);
				// HACK: inversed unknown map
				unk_map(h - i - inv_before_end - 1, j) = static_cast<uchar>((1-cell_belief.unknown()) * 255);
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

	void save_state_to_file(const std::string& _base_fname = "/home/dmo/Documents/diplom/dumps/tmp_") const /*override??*/ {
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
