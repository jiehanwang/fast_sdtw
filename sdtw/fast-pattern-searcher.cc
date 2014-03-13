// sdtw/fast-pattern-searcher.cc

// Author: David Harwath

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
			QuantizeSimilarityMatrix(thresholded_raw_similarity_matrix,
															 &quantized_similarity_matrix);
			SparseMatrix<int32> median_smoothed_matrix;
			ApplyMedianSmootherToMatrix(quantized_similarity_matrix,
																	&median_smoothed_matrix);
			SparseMatrix<BaseFloat> blurred_matrix;
			ApplyGaussianBlurToMatrix(median_smoothed_matrix, &blurred_matrix);
			std::vector<BaseFloat> hough_transform;
			ComputeDiagonalHoughTransform(blurred_matrix, &hough_transform);
			vector<int32> peak_locations;
			BaseFloat peak_delta = 0.25;
			PickPeaksInVector(hough_transform, peak_delta, &peak_locations);
			std::vector<Line> line_locations;
			ScanDiagsForLines(blurred_matrix, peak_locations, &line_locations);
			std::vector<Line> filtered_line_locations;
			BaseFloat block_filter_threshold = 0.75;
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

}

void FastPatternSearcher::QuantizeSimilarityMatrix(
	const SparseMatrix<BaseFloat> &similarity_matrix,
	SparseMatrix<int32> *quantized_similarity_matrix) {
	KALDI_ASSERT(quantized_similarity_matrix != NULL);

}

void FastPatternSearcher::ApplyMedianSmootherToMatrix(
	const SparseMatrix<int32> &input_matrix,
	SparseMatrix<int32> *median_smoothed_matrix) {
	SparseMatrix<int32> median_counts;
	const std::vector< std::pair<size_t, size_t> > nonzeros = 
		input_matrix.GetNonzeroElements();
	for (int i = 0; i < nonzeros.size(); ++i) {
		size_t row = nonzeros[i].first;
		size_t col = nonzeros[i].second;
		// Increment each element within a diagonal L radius from (row,col) in
		// median_counts by 1
	}
	const std::vector< std::pair<size_t, size_t> > nonzero_counts = 
		median_counts.GetNonzeroElements();
	for (int i = 0; i < nonzero_counts.size(); ++i) {
		// If this count is above the median threshold, set the corresponding
		// element of median_smoothed_matrix to a 1
	}
}

void FastPatternSearcher::ApplyGaussianBlurToMatrix(
	const SparseMatrix<int32> &input_matrix,
	SparseMatrix<BaseFloat> *blurred_matrix) {

}

void FastPatternSearcher::ComputeDiagonalHoughTransform(
	const SparseMatrix<BaseFloat> input_matrix,
	std::vector<BaseFloat> *hough_transform) {

}

void FastPatternSearcher::PickPeaksInVector(
	const vector<BaseFloat> &input_vector,
	const BaseFloat &peak_delta
	std::vector<int32> *peak_locations) {

}

void FastPatternSearcher::ScanDiagsForLines(
	const SparseMatrix<BaseFloat> &input_matrix,
	const std::vector<int32> &diags_to_scan,
	std::vector<Line> *line_locations) {

}


void FastPatternSearcher::FilterBlockLines(
	const SparseMatrix<BaseFloat> &similarity_matrix,
	const std::vector<Line> &line_locations,
	const BaseFloat &block_filter_threshold,
	std::vector<Line> *filtered_line_locations) {

}

void FastPatternSearcher::WarpLinesToPaths(
	const SparseMatrix<BaseFloat> &similarity_matrix,
	const std::vector<Line> &line_locations,
	std::vector<Path> *sdtw_paths) {

}

void FastPatternSearcher::WritePaths(const std::vector<Path> &sdtw_paths,
																		 PatternStringWriter *writer) {

}

}  // end namespace kaldi