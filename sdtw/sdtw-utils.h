// sdtw/sdtw-utils.h

// Author: David Harwath

#ifndef KALDI_SDTW_SDTW_UTILS_H
#define KALDI_SDTW_SDTW_UTILS_H

#include <iostream>
#include <map>
#include <utility>
#include <vector>

#include "base/kaldi-common.h"
#include "matrix/kaldi-matrix.h"
#include "util/common-utils.h"

namespace kaldi {

BaseFloat CosineSimilarity(const VectorBase<BaseFloat> &first,
												 	 const VectorBase<BaseFloat> &second);

BaseFloat KLSimilarity(const VectorBase<BaseFloat> &first,
											 const VectorBase<BaseFloat> &second);

BaseFloat DotProdSimilarity(const VectorBase<BaseFloat> &first,
														const VectorBase<BaseFloat> &second);

struct Line {
	std::pair<size_t, size_t> start;
	std::pair<size_t, size_t> end;
	Line(size_t start_row, size_t start_col, size_t end_row, size_t end_col) {
		start = std::make_pair<size_t, size_t>(start_row, start_col);
		end = std::make_pair<size_t, size_t>(end_row, end_col);
	};
};

class Path {
public:
	void Read(std::istream &in_stream, bool binary);
	void Write(std::ostream &out_stream, bool binary) const;
	std::vector<std::pair<size_t, size_t> > path_points;
	std::vector<BaseFloat> similarities;
	std::string first_id;
	std::string second_id;
};

typedef TableWriter<KaldiObjectHolder<Path> > PathWriter;

template<class T> class SparseMatrix {
	public:
		SparseMatrix() : mat_() {
			size_ = std::make_pair<size_t, size_t>(0, 0);
		}

		~SparseMatrix() {};

		std::vector<std::pair<size_t, size_t> > GetNonzeroElements() const {
			std::vector<std::pair<size_t, size_t> > retval;
			for (typename std::map<std::pair<size_t, size_t>, T>::const_iterator it = mat_.begin(); it != mat_.end(); ++it) {
				retval.push_back(it->first);
			}
			return retval;
		}

		void Clear() {
			mat_ = std::map<std::pair<size_t, size_t>, T>();
			size_ = std::make_pair<size_t, size_t>(0, 0);
		}

		void Clamp(const T &epsilon) {
			std::map<std::pair<size_t, size_t>, T> clamped_mat;
			for (typename std::map<std::pair<size_t, size_t>, T>::const_iterator it = mat_.begin(); it != mat_.end(); ++it) {
				if (it->second > epsilon) {
					clamped_mat[it->first] = it->second;
				}
			}
			mat_ = clamped_mat;
		}

		T Get(const std::pair<size_t, size_t> &coordinate) const {
			T retval = T(0);
			typename std::map<std::pair<size_t, size_t>, T>::const_iterator it = mat_.find(coordinate);
			if (it != mat_.end()) {
				retval = it->second;
			}
			return retval;
		}

		T Get(const size_t &row, const size_t &col) const {
			return Get(std::make_pair(row, col));
		}

		T GetSafe(const std::pair<size_t, size_t> &coordinate) const {
			T retval = T(0);
			if (coordinate.first >= 0 && coordinate.first < size_.first &&
					coordinate.second >= 0 && coordinate.second < size_.second) {
				typename std::map<std::pair<size_t, size_t>, T>::const_iterator it = mat_.find(coordinate);
				if (it != mat_.end()) {
					retval = it->second;
				}
			}
			return retval;
		}

		T GetSafe(const size_t &row, const size_t &col) const {
			return GetSafe(std::make_pair(row, col));
		}

		std::pair<size_t, size_t> GetSize() const {
			return size_;
		}

		bool SetSize(const size_t &num_row, const size_t &num_col) {
			return SetSize(std::make_pair(num_row, num_col));
		}

		bool SetSize(const std::pair<size_t, size_t> &size) {
			if (size.first >= 0 && size.second >= 0) {
				size_ = size;
				return true;
			}
			return false;
		}

		bool Set(const std::pair<size_t, size_t> &coordinate,
				 		 const T &value) {
			mat_[coordinate] = value;
			return true;
		}

		bool Set(const size_t &row, const size_t &col, const T &value) {
			mat_[std::make_pair(row, col)] = value;
			return true;
		}

		bool SetSafe(const std::pair<size_t, size_t> &coordinate,
					 			 const T &value) {
			if (coordinate.first >= 0 && coordinate.first < size_.first
					&& coordinate.second >= 0 && coordinate.second < size_.second) {
				if (value != 0) {
					mat_[coordinate] = value;
				}
				return true;
			}
			return false;
		}

		bool SetSafe(const size_t &row, const size_t &col, const T &value) {
			return SetSafe(std::make_pair(row, col), value);
		}

		bool IncrementSafe(const std::pair<size_t, size_t> &coordinate, const T &increment) {
			if (coordinate.first >= 0 && coordinate.first < size_.first &&
				coordinate.second >= 0 && coordinate.second < size_.second) {
				typename std::map<std::pair<size_t, size_t>, T>::const_iterator search = mat_.find(coordinate);
				if (search != mat_.end()) {
					mat_[coordinate] = mat_[coordinate] + increment;
				} else {
					mat_[coordinate] = increment;
				}
				return true;
			}
			return false;
		}

		bool IncrementSafe(const size_t &row, const size_t &col, const T &increment) {
			return IncrementSafe(std::make_pair(row, col), increment);
		}

		void Read(std::istream &in_stream, bool binary) {
			return;
		}

		void Write(std::ostream &out_stream, bool binary) const {
			WriteToken(out_stream, binary, "[");
			for (int32 row = 0; row < size_.first; ++row) {
				for (int32 col = 0; col < size_.second; ++col) {
					T val = GetSafe(std::make_pair(row, col));
					WriteBasicType(out_stream, binary, val);
				}
				WriteToken(out_stream, binary, ";");
			}
			WriteToken(out_stream, binary, "];");
		}

	private:
		std::pair<size_t, size_t> size_;
		std::map< std::pair<size_t, size_t>, T > mat_;
};

typedef TableWriter<KaldiObjectHolder<SparseMatrix<BaseFloat> > > SparseFloatMatrixWriter;
typedef TableWriter<KaldiObjectHolder<SparseMatrix<int32> > > SparseIntMatrixWriter;

}  // end namespace kaldi

#endif