// sdtw/sdtw-utils-test.cc

// Author:	David Harwath

#include "sdtw/sdtw-utils.h"

namespace kaldi {

void SparseMatrixGetNonzerosTest() {
	SparseMatrix<int> matrix;
	KALDI_ASSERT(matrix.GetNonzeroElements().size() == 0);
	matrix.SetSize(5, 5);
	matrix.SetSafe(0, 0, 1);
	matrix.SetSafe(3, 3, 1);
	KALDI_ASSERT(matrix.GetNonzeroElements().size() == 2);
	matrix.SetSafe(5, 5, 1);
	matrix.SetSafe(2, 2, 0);
	KALDI_ASSERT(matrix.GetNonzeroElements().size() == 2);
	KALDI_ASSERT(matrix.GetSafe(0, 0) == 1);
	KALDI_ASSERT(matrix.GetSafe(3, 3) == 1);
	KALDI_ASSERT(matrix.GetSafe(2, 2) == 0);
}

void SparseMatrixClampTest() {
	SparseMatrix<BaseFloat> matrix;
	matrix.SetSize(5, 5);
	matrix.SetSafe(2, 3, 0.1);
	matrix.SetSafe(3, 4, 5.0);
	KALDI_ASSERT(matrix.GetNonzeroElements().size() == 2);
	matrix.Clamp(0.5);
	KALDI_ASSERT(matrix.GetNonzeroElements().size() == 1);
	KALDI_ASSERT(matrix.GetSafe(2, 3) == 0);
	KALDI_ASSERT(matrix.GetSafe(matrix.GetNonzeroElements()[0]) == 5.0);
	KALDI_ASSERT(matrix.GetNonzeroElements()[0].first == 3);
	KALDI_ASSERT(matrix.GetNonzeroElements()[0].second == 4);
}

void SparseMatrixIncrementTest() {
	SparseMatrix<int> matrix;
	matrix.SetSize(3,3);
	matrix.SetSafe(0, 0, 1);
	matrix.IncrementSafe(0, 0, 1);
	matrix.IncrementSafe(1, 2, 3);
	KALDI_ASSERT(matrix.GetNonzeroElements().size() == 2);
	KALDI_ASSERT(matrix.GetSafe(0, 0) == 2);
	KALDI_ASSERT(matrix.GetSafe(1, 2) == 3);
}

}  // end namespace kaldi

int main() {
	using namespace kaldi;
	SparseMatrixGetNonzerosTest();
	SparseMatrixClampTest();
	SparseMatrixIncrementTest();
	std::cout << "sdtw-utils-test OK\n";
}
