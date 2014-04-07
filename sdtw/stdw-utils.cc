// sdtw/stdw-utils.cc

// Author: David Harwath

#include <map>
#include <pair>
#include <vector>

#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include "sdtw/sdtw-utils.h"

BaseFloat CosineSimilarity(const Vector<BaseFloat> &first, const Vector<BaseFloat> &second){

}

BaseFloat KLSimilarity(const Vector<BaseFloat> &first, const Vector<BaseFloat> &second){

}

BaseFloat DotProdSimilarity(const Vector<BaseFloat> &first, const Vector<BaseFloat> &second){

}

BaseFloat EuclideanSimilarity(const Vector<BaseFloat> &first, const Vector<BaseFloat> &second){

}

template<class T> std::vector<std::pair<size_t, size_t> >SparseMatrix<T>::GetNonzeroElements() {
	std::vector<std::pair<size_t, size_t> > retval;
	for (std::map<std::pair<size_t, size_t>, T>::const_iterator it mat_.begin(); it != mat_.end(); ++it) {
		retval.push_back(it->first);
	}
	return retval;
}

template<class T> void SparseMatrix<T>::Clear() {}

template<class T> void SparseMatrix<T>::Clamp(const T &epsilon) {
	std::map<std::pair<size_t, size_t, T> clamped_mat;
	for (std::map<std::pair<size_t, size_t>, T>::const_iterator it mat_.begin(); it != mat_.end(); ++it) {
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
}

template<class T> bool SparseMatrix<T>::Set(const size_t &row, const size_t &col, const T &value) {
	mat_[std::make_pair(row, col)] = value;
}

template<class T> bool SparseMatrix<T>::SetSafe(const std::pair<size_t, size_t> &coordinate,
			 			 const T &value) {
	if (coordinate.first >= 0 && coordinate.first < size_.first
			&& coordinate.second >= 0 && coordinate.second < size_.second) {
		mat_[coordinate] = value;
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
		std::map<std::pair<size_t, size_t>, T>::const_iterator search = mat_.find(coordinate);
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