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
	bool use_cosine;
	bool use_dotprod;
	bool use_kl;
	BaseFloat quantize_threshold;
	int32 smoother_length;
	BaseFloat smoother_median;
	int32 sdtw_width;
	BaseFloat sdtw_budget;
	BaseFloat sdtw_trim;

	FastPatternSearcherConfig(): use_cosine(true), use_dotprod(false),
		use_kl(false), quantize_threshold(0.5),
		smoother_length(7), smoother_median(0.5), sdtw_width(10),
		sdtw_budget(7.0), sdtw_trim(0.1) {}

	void Register(OptionsItf *po) {
		po->Register("use-cosine", &use_cosine,
				"Use cosine similarity between frames. Behavior may be "
				"unpredictable when multiple similarity measures are specified.");
		po->Register("use-dotprod", &use_dotprod,
				"Use dot product similarity between frames");
		po->Register("use-kl", &use_kl,
				"Use KL similarity between frames");
		po->Register("quantize-threshold", &quantize_threshold,
				"Frame similarities below this value are set to 0");
		po->Register("smoother-length", &smoother_length,
				"Context radius of the diagonal median smoothing filter. Total "
				"filter length is twice this value plus one");
		po->Register("smoother-median", &smoother_median,
				"Mu parameter for the median smoothing filter");
		po->Register("sdtw-width", &sdtw_width,
				"S-DTW bandwidth parameter");
		po->Register("sdtw-budget", &sdtw_budget,
				"S-DTW distortion budget for each direction "
				"(forwards and backwards)");
		po->Register("sdtw-trim", &sdtw_trim,
				"Trim frames from the ends of each warp path whose similarity "
				"falls below this threshold");
	}
	void Check() const {
		KALDI_ASSERT(quantize_threshold >= 0 && smoother_length > 0
								 && smoother_median >= 0 && smoother_median <= 1
								 && sdtw_width > 1 && sdtw_budget > 0 && sdtw_trim >= 0
								 && (use_cosine || use_dotprod || use_kl));
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

	// TODO: Check that I am passing in the pattern_writer properly (pointer)
	bool Search(const std::vector<Matrix<BaseFloat> > &utt_features,
							const std::vector<std::string> &utt_ids,
							PatternStringWriter *pattern_writer) const;

	void ComputeThresholdedSimilarityMatrix(
				const Matrix<BaseFloat> &first_features,
				const Matrix<BaseFloat> &second_features,
				SparseMatrix<BaseFloat> *similarity_matrix) const;

	// Applies a fixed quantization threshold to the values of the input
	// matrix, writing the quantized (1 or 0 valued) matrix to the provided
	// output matrix. Does not check that the matrix pointed to by
	// quantized_matrix is initially empty, nor does it bother to clear it. For
	// predictable behavior, supply an empty matrix.
	void QuantizeMatrix(
				const SparseMatrix<BaseFloat> &input_matrix,
				const BaseFloat &quantization_threshold,
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
				const SparseMatrix<int32> &similarity_matrix,
				const std::vector<Line> &line_locations,
				const BaseFloat &block_filter_threshold,
				std::vector<Line> *filtered_line_locations) const;

	void SDTWWarp(const SparseMatrix<BaseFloat> &similarity_matrix,
								const std::pair<size_t, size_t> &start_point,
				  			const std::pair<size_t, size_t> &end_point, Path *path) const;

	void MergeAndTrimPaths(const Path &first_half, const Path &second_half,
						   Path *result) const;

	void WarpLinesToPaths(
				const SparseMatrix<BaseFloat> &similarity_matrix,
				const std::vector<Line> &line_locations,
				std::vector<Path> *sdtw_paths) const;

	void WritePaths(std::string first_id, std::string second_id,
									const std::vector<Path> &sdtw_paths,
									PatternStringWriter *writer) const;

private:
	FastPatternSearcherConfig config_;
};

}  // end namespace kaldi

#endif