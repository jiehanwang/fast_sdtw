// sdtw/sdtw-utils.h

// Author: David Harwath

#ifndef KALDI_SDTW_SDTW_UTILS_H
#define KALDI_SDTW_SDTW_UTILS_H

#include <map>
#include <pair>
#include <vector>

#include "base/kaldi-common.h"
#include "util/common-utils.h"

namespace kaldi {

BaseFloat CosineSimilarity(const Vector<BaseFloat> &first, const Vector<BaseFloat> &second);
BaseFloat KLSimilarity(const Vector<BaseFloat> &first, const Vector<BaseFloat> &second);
BaseFloat DotProdSimilarity(const Vector<BaseFloat> &first, const Vector<BaseFloat> &second);
BaseFloat EuclideanSimilarity(const Vector<BaseFloat> &first, const Vector<BaseFloat> &second);

struct Line {
	std::pair<size_t, size_t> start;
	std::pair<size_t, size_t> end;
	Line(size_t start_row, size_t start_col, size_t end_row, size_t end_col) {
		start = std::make_pair<size_t, size_t>(start_row, start_col);
		end = std::make_pair<size_t, size_t>(end_row, end_col);
	};
};

struct Path {
	std::vector< std::pair<size_t, size_t> > path_points;
	vector<BaseFloat> similarities;
};

// Define the [][] operator for this class to make things easier.
// Also include a method to iterate over the nonzero elements in O(k) time
// where k is the number of nonzero entries.
template<class T> class SparseMatrix {
	public:
		SparseMatrix() {};
		~SparseMatrix() {};

		std::vector< std::pair<size_t, size_t> > GetNonzeroElements();

	private:
		std::map< std::pair<size_t, size_t>, T > mat;
};

}  // end namespace kaldi

#endif