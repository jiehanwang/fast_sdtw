// sdtw/sdtw-utils.h

// Author: David Harwath

#ifndef KALDI_SDTW_SDTW_UTILS_H
#define KALDI_SDTW_SDTW_UTILS_H

#include <iostream>
#include <map>
#include <utility>
#include <vector>

#include "base/kaldi-common.h"
#include "util/common-utils.h"

namespace kaldi {

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
	std::vector<std::pair<size_t, size_t> > path_points;
	std::vector<BaseFloat> similarities;
	std::string first_id;
	std::string second_id;

	void Read(std::istream &in_stream, bool binary) {
	  int32 path_length;
	  std::string id1, id2;
	  ExpectToken(in_stream, binary, "<ID1>");
	  ReadToken(in_stream, binary, &id1);
	  ExpectToken(in_stream, binary, "<ID2>");
	  ReadToken(in_stream, binary, &id2);
	  ExpectToken(in_stream, binary, "<LENGTH>");
	  ReadBasicType(in_stream, binary, &path_length);
	  similarities.clear();
	  path_points.clear();
	  similarities.reserve(path_length);
	  path_points.reserve(path_length);
	  for (int32 i = 0; i < path_length; ++i) {
	  	size_t first, second;
	  	BaseFloat similarity;
	  	ReadBasicType(in_stream, binary, &first);
	  	ReadBasicType(in_stream, binary, &second);
	  	ReadBasicType(in_stream, binary, &similarity);
	    similarities.push_back(similarity);
	    path_points.push_back(std::make_pair(first, second));
	  }
	}

	void Write(std::ostream &out_stream, bool binary) const {
	  int32 path_length = this->path_points.size();
	  KALDI_ASSERT(path_length == this->similarities.size());
	  if (path_length == 0) {
	    KALDI_WARN << "Trying to write empty Path object.";
	  }
	  WriteToken(out_stream, binary, "<ID1>");
	  WriteToken(out_stream, binary, first_id);
	  WriteToken(out_stream, binary, "<ID2>");
	  WriteToken(out_stream, binary, second_id);
	  WriteToken(out_stream, binary, "<LENGTH>");
	 	WriteBasicType(out_stream, binary, path_length);
	  for (int32 i = 0; i < path_length; ++i) {
	    size_t first = path_points[i].first;
	    size_t second = path_points[i].second;
	    BaseFloat similarity = similarities[i];
	    WriteBasicType(out_stream, binary, first);
	    WriteBasicType(out_stream, binary, second);
	    WriteBasicType(out_stream, binary, similarity);
	  }
	}
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
			auto it = mat_.find(coordinate);
			if (it != mat_.end()) {
				retval = *it;
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
				auto it = mat_.find(coordinate);
				if (it != mat_.end()) {
					retval = *it;
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

	private:
		std::pair<size_t, size_t> size_;
		std::map< std::pair<size_t, size_t>, T > mat_;
};

BaseFloat CosineSimilarity(const Vector<BaseFloat> &first,
													 const Vector<BaseFloat> &second){
	Vector<BaseFloat> f = first;
	Vector<BaseFloat> s = second;
	f.Scale(1.0 / f.Norm(2));
	s.Scale(1.0 / f.Norm(2));
	f.MulElements(s);
	return f.Sum();
}

BaseFloat KLSimilarity(const Vector<BaseFloat> &first,
											 const Vector<BaseFloat> &second){
	// TODO: Implement this.
	return 0.0;
}

BaseFloat DotProdSimilarity(const Vector<BaseFloat> &first,
														const Vector<BaseFloat> &second){
	Vector<BaseFloat> f = first;
	f.MulElements(second);
	return f.Sum();
}

}  // end namespace kaldi

#endif