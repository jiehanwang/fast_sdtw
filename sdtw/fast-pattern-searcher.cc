// sdtw/fast-pattern-searcher.cc

// Author: David Harwath

#include <algorithm>
#include <pair>
#include <vector>

#include "base/kaldi-common.h"
#include "sdtw/fast-pattern-searcher.h"
#include "sdtw/sdtw-utils.h"
#include "util/common-utils.h"

namespace kaldi {

FastPatternSearcher::FastPatternSearcher(
				const FastPatternSearcherConfig &config): config_(config) {
	config.Check();
	// TODO: Set the similarity measure to use here
}

bool FastPatternSearcher::Search(
				const std::vector< Matrix<BaseFloat> &utt_features,
				const std::vector<std::string> &utt_ids,
				PatternStringWriter *pattern_writer) {

	KALDI_ASSERT(utt_features.size() == utt_ids.size());

	// Algorithm flow:
	// For each pair of matrices in utt_features:
	//		1. Compute similarity matrix
	//		2. Quantize similarity matrix
	//		3. Apply median smoothing filter
	//		4. Apply Gaussian blur
	//		5. Apply 1-D Hough transform
	//		6. Pick peaks in 1-D Hough transform
	//		7. Search for line segments in diagonals specified by peaks
	//		8. Filter out line segments whose enclosing box appears to be
	//				solid in the original similarity matrix
	//		9. Go back and refine the warp path of each line segment with SDTW

	// TODO: Figure out a better way to iterate over each pair of matrices,
	//			i.e. with iterators
	// TODO: I should probably precompute normalized features (e.g. L2
	//       normalize each row of each feature matrix if using cosine
	//       similarity) for the sake of efficiency.
	for (int i = 0; i < utt_features.size() - 1; ++i) {
		for (int j = i + 1; j < utt_features.size(); ++j) {
			const Matrix<BaseFloat> &first_features = utt_features[i];
			const Matrix<BaseFloat> &second_features = utt_features[j];
			const std::string first_utt = utt_ids[i];
			const std::string second_utt = utt_ids[j];
			//TODO: Implement these methods as well as SparseMatrix, Line, Path,
			// and PatternStringWriter
			SparseMatrix<BaseFloat> thresholded_raw_similarity_matrix;
			ComputeThresholdedSimilarityMatrix(first_features, second_features,
																				 &thresholded_raw_similarity_matrix);
			SparseMatrix<int32> quantized_similarity_matrix;
			QuantizeMatrix(thresholded_raw_similarity_matrix,
										 config_.quantize_threshold,
										 &quantized_similarity_matrix);
			SparseMatrix<int32> median_smoothed_matrix;
			ApplyMedianSmootherToMatrix(quantized_similarity_matrix,
																	&median_smoothed_matrix);
			SparseMatrix<BaseFloat> blurred_matrix;
			const size_t kernel_radius = 1;
			ApplyGaussianBlurToMatrix(median_smoothed_matrix, kernel_radius,
																&blurred_matrix);
			std::vector<BaseFloat> hough_transform;
			ComputeDiagonalHoughTransform(blurred_matrix, &hough_transform);
			vector<int32> peak_locations;
			const BaseFloat peak_delta = 0.25;
			PickPeaksInVector(hough_transform, peak_delta, &peak_locations);
			std::vector<Line> line_locations;
			ScanDiagsForLines(blurred_matrix, peak_locations, &line_locations);
			std::vector<Line> filtered_line_locations;
			const BaseFloat block_filter_threshold = 0.75;
			FilterBlockLines(quantized_similarity_matrix, line_locations,
											 block_filter_threshold, &filtered_line_locations);
			std::vector<Path> sdtw_paths;
			WarpLinesToPaths(thresholded_raw_similarity_matrix,
											 filtered_line_locations, &sdtw_paths);
			WritePaths(sdtw_paths);
		}
	}
}

void FastPatternSearcher::ComputeThresholdedSimilarityMatrix(
				const Matrix<BaseFloat> &first_features,
				const Matrix<BaseFloat> &second_features,
				SparseMatrix<BaseFloat> *similarity_matrix) {
	KALDI_ASSERT(similarity_matrix != NULL);
	similarity_matrix->Clear();
	const std::pair<size_t, size_t> size =
		std::make_pair<size_t, size_t>(first_features.NumRows(),
																	 second_features.NumRows());
	similarity_matrix->SetSize(size);
	// TODO: finish this method.
}

void FastPatternSearcher::QuantizeMatrix(
				const SparseMatrix<BaseFloat> &input_matrix,
				const BaseFloat &quantization_threshold,
				SparseMatrix<int32> *quantized_matrix) {
	KALDI_ASSERT(quantized_similarity_matrix != NULL);
	quantized_matrix->Clear();
	quantized_matrix->SetSize(input_matrix.GetSize());
	const std::vector< std::pair<size_t, size_t> > nonzeros =
		input_matrix.GetNonzeroElements();
	for (int i = 0; i < nonzeros.size(); ++i) {
		const std::pair<size_t, size_t> &coordinate = nonzeros[i];
		if (input_matrix.Get(coordinate) >= quantization_threshold) {
			quantized_similarity_matrix->Set(coordinate, 1);
		}
	}
}

void FastPatternSearcher::ApplyMedianSmootherToMatrix(
				const SparseMatrix<int32> &input_matrix,
				const int32 &smoother_length,
				const BaseFloat &smoother_median,
				SparseMatrix<int32> *median_smoothed_matrix) {
	KALDI_ASSERT(median_smoothed_matrix != NULL);
	median_smoothed_matrix->Clear();
	median_smoothed_matrix->SetSize(input_matrix.GetSize());
	// Iterates over the nonzero elements of the input matrix. For each 
	// coordinate that holds a nonzero value, increments the neighboring
	// coordinates (that fall within the radius of the diagonal median smoothing
	// filter) by 1.
	SparseMatrix<int32> median_counts;
	const std::vector< std::pair<size_t, size_t> > nonzeros = 
		input_matrix.GetNonzeroElements();
	for (int i = 0; i < nonzeros.size(); ++i) {
		const std::pair<size_t, size_t> &coordinate = nonzeros[i];
		for (int j = -1 * smoother_length; j <= smoother_length; ++j) {
			const std::pair<size_t, size_t> offset_coordinate = 
				std::make_pair<size_t, size_t>(coordinate.first + j,
																			 coordinate.second + j);
			// increment_safe() will not fail if it is supplied with a coordinate
			// that is out of the matrix's range. It will simply not increment
			// anything.
			median_counts.IncrementSafe(offset_coordinate, 1);
		}
	}
	const std::vector< std::pair<size_t, size_t> > nonzero_counts = 
		median_counts.GetNonzeroElements();
	// Precomputes the count threshold for median smoothing. These casts
	// *should* work, but definitely needs to be tested.
	const int32 threshold = static_cast<int32>(
			smoother_median * static_cast<BaseFloat>(2 * smoother_length + 1));
	for (int i = 0; i < nonzero_counts.size(); ++i) {
		// If this count is above the median threshold, sets the corresponding
		// element of median_smoothed_matrix to a 1
		const std::pair<size_t, size_t> coordinate = nonzero_counts[i];
		if (median_counts.Get(coordinate) >= threshold) {
			median_smoothed_matrix.Set(coordinate, 1);
		}
	}
}

void FastPatternSearcher::ApplyGaussianBlurToMatrix(
				const SparseMatrix<int32> &input_matrix,
				const size_t &kernel_radius,
				SparseMatrix<BaseFloat> *blurred_matrix) {
	KALDI_ASSERT(blurred_matrix != NULL);
	KALDI_ASSERT(kernel_radius > 0);
	blurred_matrix->Clear();
	blurred_matrix->SetSize(input_matrix.GetSize());
	// Creates a 2D array to hold our little Gaussian image kernel
	// Kernel radius specifies how many pixels the Gaussian spans away from the
	// center. So a radius of 1 means that the total size is a 3x3 square.
	const size_t kernel_width = 2 * kernel_radius + 1;	
	BaseFloat *kernel = new BaseFloat[kernel_width * kernel_width];
	BaseFloat squareradius = kernel_radius * kernel_radius;
	for (int row = 0; row < kernel_width; ++row) {
		for (int col = 0; col < kernel_width; ++col) {
			const BaseFloat squaredist = sqrt(pow(row - kernel_radius ,2) + 
																				pow(col - kernel_radius ,2));
			kernel[row * kernel_width + col] = 
				exp(-1 * squaredist / (2 * squareradius));
		}
	}
	// Convolves the input matrix with the Gaussian image kernel.
	// Iterates through the nonzero elements of the input matrix, and for each
	// nonzero coordinate plops down a Gaussian (weighted by the value of the
	// nonzero element) at the corresponding location in blurred_matrix
	const std::vector< std::pair<size_t, size_t> > nonzeros = 
		input_matrix.GetNonzeroElements();
	for (int i = 0; i < nonzeros.size(); ++i) { 
		const std::pair<size_t, size_t> &coordinate = nonzeros[i];
		for (kernel_row = 0; kernel_row < kernel_width; ++kernel_row) {
			for (kernel_col = 0; kernel_col < kernel_width; ++ kernel_col) {
				const BaseFloat &kernel_value =
					kernel[kernel_row * kernel_width + kernel_col];
				const BaseFloat &matrix_value = input_matrix.Get(coordinate);
				const size_t increment_row =
					coordinate.first + kernel_row - kernel_radius;
				const size_t increment_col =
					coordinate.second + kernel_col - kernel_radius;
				const std::pair<size_t, size_t> increment_coordinate =
					std::make_pair<size_t, size_t>(increment_row, increment_col);
				blurred_matrix->IncrementSafe(increment_coordinate,
																			matrix_value * kernel_value);
			}
		}
	}
	delete [] kernel;
}

void FastPatternSearcher::ComputeDiagonalHoughTransform(
				const SparseMatrix<BaseFloat> input_matrix,
				std::vector<BaseFloat> *hough_transform) {
	KALDI_ASSERT(hough_transform != NULL);
	hough_transform->clear();
		// For an M by N matrix, there will be (M + N - 1) total diagonals.
		// The lower left corner of the matrix corresponds to diagonal index 0,
		// while the upper right corner of the matrix corresponds to diagonal
		// index (M + N - 2)
		const size_t M = input_matrix.NumRows();
		const size_t N = input_matrix.NumCols();
		hough_transform->resize(M + N - 1);
		// Iterate over the nonzero elements of the matrix, figure out which
		// diagonal the coordinate resides in, and then increment the
		// corresponding index of the hough_transform vector by the value of the
		// matrix at said coordinate.
		// Based on the way we have defined the 0th diagonal (bottom left corner
		// of the input_matrix) and the (M + N - 2)th diagonal (upper right corner
		// of the input matrix), the index of the diagonal at coordinate
		// (row, col) is given by (col - row + M - 1).
		const std::vector< std::pair<size_t, size_t> > nonzeros = 
			input_matrix.GetNonzeroElements();
		for (int i = 0; i < nonzeros.size(); ++i) {
			const std::pair<size_t, size_t> &coordinate = nonzeros[i];
			const int diag = coordinate.second - coordinate.first + M - 1;
			KALDI_ASSERT(diag >= 0 && diag <= (M + N - 1));
			hough_transform[diag] += input_matrix.Get(coordinate);
		}
}

// This function uses the peakdet algorithm by Eli Billauer (public domain).
void FastPatternSearcher::PickPeaksInVector(
				const vector<BaseFloat> &input_vector,
				const BaseFloat &peak_delta
				std::vector<int32> *peak_locations) {
	KALDI_ASSERT(peak_locations != NULL);
	KALDI_ASSERT(input_vector.size() > 0);
	peak_locations->clear();
	BaseFloat minvalue = input_vector[0];
	BaseFloat maxvalue = input_vector[0];
	for (int i = 0; i < input_vector.size(); ++i) {
		if (input_vector[i] < minvalue) {
			minvalue = input_vector[i];
		}
		if (input_vector[i] > maxvalue) {
			maxvalue = input_vector[i];
		}
	}
	const BaseFloat delta = peak_delta * (maxvalue - minvalue);
	BaseFloat mx = minvalue - 1;
	BaseFloat mn = maxvalue + 1;
	int mxidx = -1;
	int mnidx = -1;
	bool findMax = true;
	for (int i = 0; i < input_vector.size(); ++i) {
		const BaseFloat &val = input_vector[i];
		if (val > mx) {
			mx = val;
			mxidx = i;
		} 
		if (val < mn) {
			mn = val;
			mnidx = i;
		}
		if (findMax && val < (mx - delta)) {
				peak_locations.push_back(mxidx);
				mn = val;
				mnidx = i;
				findMax = false;
		} else if (!findMax && val > (mn + delta)) {
				mx = val;
				mxidx = i;
				findMax = true;
		}
	}
}

void FastPatternSearcher::ScanDiagsForLines(
				const SparseMatrix<BaseFloat> &input_matrix,
				const std::vector<int32> &diags_to_scan,
				std::vector<Line> *line_locations) {
	KALDI_ASSERT(line_locations != NULL);
	line_locations->clear();
	const size_t M = input_matrix.NumRows();
	const size_t N = input_matrix.NumCols();
	// The idea here is simple; for each diagonal we want to scan, we simply
	// iterate over the elements in that diagonal (in order), looking for 
	// continuous nonzero regions.
	for (int i = 0; i < diags_to_scan.size(); ++i) {
		size_t row = -1;
		size_t col = -1;
		// the diagonal index is equal to (col - row + M - 1), so we also have
		// that row - col = M - 1 - index. Assuming that we wish to start
		// scanning along either the top or left edge of the matrix, one of
		// row or col must be zero, and neither can be negative. So, applying
		// these constraints to the equation above allows us to uniquely solve
		// for the starting point for our scan.
		int diff = M - 1 - diags_to_scan[i];
		if (diff > 0) {
			row = diff;
			col = 0;
		} else if (diff < 0) {
			row = 0;
			col = -1 * diff;
		} else if (diff == 0) {
			row = 0;
			col = 0;
		}
		for(; (row < M && col < N); ++row, ++col;) {
			// TODO: finish this method.
		}
	}

}

void FastPatternSearcher::FilterBlockLines(
				const SparseMatrix<BaseFloat> &similarity_matrix,
				const std::vector<Line> &line_locations,
				const BaseFloat &block_filter_threshold,
				std::vector<Line> *filtered_line_locations) {
	KALDI_ASSERT(filtered_line_locations != NULL);
	filtered_line_locations->clear();
	// The idea here is pretty simple. Given a line from (start_row, start_col)
	// to (end_row, end_col), compute the average similarity of the rectangle
	// enclosing the line. If the average similarity exceeds the specified
	// threshold, then do not include the line in the filtered list.
	// TODO: finish this method.
	for (int i = 0; i < line_locations.size(); ++i) {
		const Line &line = line_locations[i];
		const std::pair<size_t, size_t> start = line.start;
		const std::pair<size_t, size_t> end = line.end;
		// I don't assume that start is the upper leftmost point of the line
		const size_t row_min = std::min(start.first, end.first);
		const size_t row_max = std::max(start.first, end.first);
		const size_t col_min = std::min(start.second, end.second);
		const size_t col_max = std::max(start.second, end.second);
		BaseFloat blocksum = 0.0;
		for (size_t row = row_min; row <= row_max; ++row) {
			for (size_t col = col_min; col <= col_max; ++col) {
				blocksum += similarity_matrix.GetSafe(
					std::make_pair<size_t, size_t>(row, col));
			}
		}
		int32 num_pixels = (row_max - row_min) * (col_max - col_min);
		KALDI_ASSERT(num_pixels > 0);
		if ((blocksum / static_cast<BaseFloat>(num_pixels)) < 
			block_filter_threshold) {
			filtered_line_locations->push_back(line);
		}
	}
}

void FastPatternSearcher::WarpLinesToPaths(
				const SparseMatrix<BaseFloat> &similarity_matrix,
				const std::vector<Line> &line_locations,
				std::vector<Path> *sdtw_paths) {
	KALDI_ASSERT(sdtw_paths != NULL);
	sdtw_paths->clear();
// TODO: finish this method.
}

void FastPatternSearcher::WritePaths(const std::vector<Path> &sdtw_paths,
																		 PatternStringWriter *writer) {
	KALDI_ASSERT(writer != NULL);
// TODO: finish this method.
}

}  // end namespace kaldi