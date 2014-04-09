// sdtw/stdw-utils.cc

// Author: David Harwath

#include <map>
#include <utility>
#include <vector>

#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include "sdtw/sdtw-utils.h"

namespace kaldi {

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

template<class T> std::vector<std::pair<size_t, size_t> > SparseMatrix<T>::GetNonzeroElements() {
	std::vector<std::pair<size_t, size_t> > retval;
	for (typename std::map<std::pair<size_t, size_t>, T>::const_iterator it = mat_.begin(); it != mat_.end(); ++it) {
		retval.push_back(it->first);
	}
	return retval;
}

template<class T> void SparseMatrix<T>::Clear() {
	mat_ = std::map<std::pair<size_t, size_t>, T>();
	size_ = std::make_pair<size_t, size_t>(0, 0);
}

template<class T> void SparseMatrix<T>::Clamp(const T &epsilon) {
	std::map<std::pair<size_t, size_t>, T> clamped_mat;
	for (typename std::map<std::pair<size_t, size_t>, T>::const_iterator it = mat_.begin(); it != mat_.end(); ++it) {
		if (it->second > epsilon) {
			clamped_mat[it->first] = it->second;
		}
	}
	mat_ = clamped_mat;
}

template<class T> T SparseMatrix<T>::Get(const std::pair<size_t, size_t> &coordinate) const {
	return mat_[coordinate];
}

template<class T> T SparseMatrix<T>::Get(const size_t &row, const size_t &col) const {
	return mat_[std::make_pair(row, col)];
}

template<class T> T SparseMatrix<T>::GetSafe(const std::pair<size_t, size_t> &coordinate) const {
	if (coordinate.first >= 0 && coordinate.first < size_.first &&
			coordinate.second >= 0 && coordinate.second < size_.second) {
		return mat_[coordinate];
	}
	return T(0);
}

template<class T> T SparseMatrix<T>::GetSafe(const size_t &row, const size_t &col) const {
	return GetSafe(std::make_pair(row, col));
}

template<class T> std::pair<size_t, size_t> SparseMatrix<T>::GetSize() const {
	return size_;
}

template<class T> bool SparseMatrix<T>::SetSize(const size_t &num_row, const size_t &num_col) {
	return SetSize(std::make_pair(num_row, num_col));
}

template<class T> bool SparseMatrix<T>::SetSize(const std::pair<size_t, size_t> &size) {
	if (size.first >= 0 && size.second >= 0) {
		size_ = size;
		return true;
	}
	return false;
}

template<class T> bool SparseMatrix<T>::Set(const std::pair<size_t, size_t> &coordinate,
		 		 const T &value) {
	mat_[coordinate] = value;
	return true;
}

template<class T> bool SparseMatrix<T>::Set(const size_t &row, const size_t &col, const T &value) {
	mat_[std::make_pair(row, col)] = value;
	return true;
}

template<class T> bool SparseMatrix<T>::SetSafe(const std::pair<size_t, size_t> &coordinate,
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

template<class T> bool SparseMatrix<T>::SetSafe(const size_t &row, const size_t &col, const T &value) {
	return SetSafe(std::make_pair(row, col), value);
}

template<class T> bool SparseMatrix<T>::IncrementSafe(const std::pair<size_t, size_t> &coordinate,
				   				 const T &increment) {
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

template<class T> bool SparseMatrix<T>::IncrementSafe(const size_t &row, const size_t &col,
									 const T &increment) {
	return IncrementSafe(std::make_pair(row, col), increment);
}

}  // end namespace kaldi