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

		void Clear();

		void Clamp(const BaseFloat &epsilon);

		T Get(const std::pair<size_t, size_t> &coordinate) const;

		T Get(const size_t &row, const size_t &col) const;

		T GetSafe(const std::pair<size_t, size_t> &coordinate) const;

		T GetSafe(const size_t &row, const size_t &col) const;

		std::pair<size_t, size_t> GetSize() const;

		bool SetSize(const std::pair<size_t, size_t> &size);

		bool Set(const std::pair<size_t, size_t> &coordinate,
				 		 const T &value);

		bool Set(const size_t &row, const size_t &col, const T &value);

		bool SetSafe(const std::pair<size_t, size_t> &coordinate,
					 			 const T &value);

		bool SetSafe(const size_t &row, const size_t &col, const T &value);

		bool IncrementSafe(const std::pair<size_t, size_t> &coordinate,
						   				 const T &increment);

		bool IncrementSafe(const size_t &row, const size_t &col,
											 const T &increment)

	private:
		std::pair<size_t, size_t> size_;
		std::map< std::pair<size_t, size_t>, T > mat_;
};

}  // end namespace kaldi

#endif