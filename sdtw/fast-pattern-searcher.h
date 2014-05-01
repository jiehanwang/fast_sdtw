// sdtw/fast-pattern-searcher.h

// Author: David Harwath

#ifndef KALDI_SDTW_FAST_PATTERN_SEARCHER_H
#define KALDI_SDTW_FAST_PATTERN_SEARCHER_H

#include <vector>

#include "base/kaldi-common.h"
#include "sdtw/sdtw-utils.h"
#include "util/common-utils.h"

namespace kaldi {

struct FastPatternSearcherConfig {
	BaseFloat quantize_threshold;
	int32 smoother_length;
	BaseFloat smoother_median;
	BaseFloat peak_delta;
	BaseFloat block_threshold;
	int32 kernel_radius;
	int32 sdtw_width;
	BaseFloat sdtw_budget;
	BaseFloat sdtw_trim;
	int32 min_length;

	FastPatternSearcherConfig(): quantize_threshold(0.75), smoother_length(20), 
		smoother_median(0.45), peak_delta(0.25), block_threshold(0.7), kernel_radius(1), 
		sdtw_width(7), sdtw_budget(10.0), sdtw_trim(0.65), min_length(30) {}

	void Register(OptionsItf *po) {
		po->Register("quantize-threshold", &quantize_threshold,
				"Similarity matrix quantization threshold");
		po->Register("smoother-length", &smoother_length,
				"Context radius of the diagonal median smoothing filter. Total "
				"filter length is twice this value plus one");
		po->Register("smoother-median", &smoother_median,
				"Mu parameter for the median smoothing filter");
		po->Register("peak-delta", &peak_delta,
			"delta parameter for the peakdet peak picker");
		po->Register("block-threshold", &block_threshold,
				"Filter out lines whose enclosing block has an average similarity threshold"
				" greater than this value");
		po->Register("kernel-radius", &kernel_radius,
			"radius of the Gaussian blurring kernel");
		po->Register("sdtw-width", &sdtw_width,
				"S-DTW bandwidth parameter");
		po->Register("sdtw-budget", &sdtw_budget,
				"S-DTW distortion budget for each direction "
				"(forwards and backwards)");
		po->Register("sdtw-trim", &sdtw_trim,
				"Trim frames from the ends of each warp path whose similarity "
				"falls below this threshold");
		po->Register("min-length", &min_length,
				"Throw away S-DTW paths shorter than this length");
	}
	void Check() const {
		KALDI_ASSERT(quantize_threshold >= 0 && smoother_length > 0 && min_length > 0
								 && smoother_median >= 0 && smoother_median <= 1 && kernel_radius >= 0
								 && sdtw_width > 1 && sdtw_budget > 0 && sdtw_trim >= 0
								 && peak_delta > 0);
	}
};

class FastPatternSearcher {
public:
	FastPatternSearcher(const FastPatternSearcherConfig &config);

	void SetOptions(const FastPatternSearcherConfig &config) {
		config_ = config;
	}

	FastPatternSearcherConfig GetOptions() const {
		return config_;
	}

	~FastPatternSearcher() {
	}

	bool Search(
	const std::vector<std::pair<std::string, Matrix<BaseFloat> > > feats,
	PathWriter *pattern_writer) const;

	bool Search(
	const std::vector<std::pair<std::string, Matrix<BaseFloat> > > feats_a,
	const std::vector<std::pair<std::string, Matrix<BaseFloat> > > feats_b,
	PathWriter *pattern_writer) const;

	bool SearchOnePair(
				const Matrix<BaseFloat> &first_features,
				const Matrix<BaseFloat> &second_features,
				const std::string &first_id,
				const std::string &second_id,
				PathWriter *pattern_writer) const;

	Matrix<BaseFloat> L2NormalizeFeatures(const Matrix<BaseFloat> &features) const;

	// Compute a cosine similarity matrix as well as a sparse matrix
	// representing the binary quantized similarity matrix. This
	// could possibly be refactored into two separate methods.
	void ComputeThresholdedSimilarityMatrix(
				const Matrix<BaseFloat> &first_features,
				const Matrix<BaseFloat> &second_features,
				Matrix<BaseFloat> *similarity_matrix,
				SparseMatrix<int32> *quantized_matrix) const;

	// Applies a fixed quantization threshold to the values of the input
	// matrix, writing the quantized (1 or 0 valued) matrix to the provided
	// output matrix. Does not check that the matrix pointed to by
	// quantized_matrix is initially empty, nor does it bother to clear it. For
	// predictable behavior, supply an empty matrix.
	// I don't use this method anymore; the quantization is taken care of
	// by ComputeThresholdedSimilarityMatrix
	void QuantizeMatrix(
				const SparseMatrix<BaseFloat> &input_matrix,
				SparseMatrix<int32> *quantized_matrix) const;

	// Applies a diagonal median smoother to an (assumed) binary input matrix
	// and writes the output to the provided output matrix. Does not check
	// that the provided output matrix is initially all zeros, and does not
	// clear its initial values. For predictable behavior, make sure that
	// the matrix pointed to by median_smoothed_matrix is empty!
	void ApplyMedianSmootherToMatrix(
				const SparseMatrix<int32> &input_matrix,
				SparseMatrix<int32> *median_smoothed_matrix) const;

	void ApplyGaussianBlurToMatrix(
				const SparseMatrix<int32> &input_matrix,
				const size_t &kernel_radius,
				SparseMatrix<BaseFloat> *blurred_matrix) const;

	void ComputeDiagonalHoughTransform(
				const SparseMatrix<BaseFloat> input_matrix,
				std::vector<BaseFloat> *hough_transform) const;

	void PickPeaksInVector(
				const std::vector<BaseFloat> &input_vector,
				const BaseFloat &peak_delta,
				std::vector<int32> *peak_locations) const;

	void ScanDiagsForLines(
				const SparseMatrix<BaseFloat> &input_matrix,
				const std::vector<int32> &diags_to_scan,
				std::vector<Line> *line_locations) const;

	void FilterBlockLines(
				const Matrix<BaseFloat> &similarity_matrix,
				const std::vector<Line> &line_locations,
				const BaseFloat &block_filter_threshold,
				std::vector<Line> *filtered_line_locations) const;

	void SDTWWarp(const Matrix<BaseFloat> &first_features,
								const Matrix<BaseFloat> &second_features,
								const std::pair<size_t, size_t> &start_point,
				  			const std::pair<size_t, size_t> &end_point, Path *path) const;

	void WarpForward(const Matrix<BaseFloat> &similarity_matrix,
									 const std::pair<size_t, size_t> &start_point,
									 Path *path) const;

	void WarpBackward(const Matrix<BaseFloat> &similarity_matrix,
										const std::pair<size_t, size_t> &start_point,
										Path *path) const;

	void MergeAndTrimPaths(const Path &first_half, const Path &second_half,
						   					 Path *result) const;

	void WarpLinesToPaths(
				const Matrix<BaseFloat> &similarity_matrix,
				const std::vector<Line> &line_locations,
				std::vector<Path> *sdtw_paths) const;

	void WritePaths(const std::vector<Path> &sdtw_paths,
									PathWriter *writer) const;

	void WriteOverlaidMatrix(
	const SparseMatrix<BaseFloat> &similarity_matrix,
	const std::vector<Path> sdtw_paths, const std::string key,
	SparseFloatMatrixWriter *matrix_writer) const;

private:
	FastPatternSearcherConfig config_;
};

}  // end namespace kaldi

#endif