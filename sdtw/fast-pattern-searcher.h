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
	bool use_euclidean;
	bool use_kl;
	BaseFloat quantize_threshold;
	int32 smoother_length;
	BaseFloat smoother_median;
	int32 sdtw_width;
	BaseFloat sdtw_budget;
	BaseFloat sdtw_trim;

	FastPatternSearcherConfig(): use_cosine(true), use_dotprod(false),
		use_euclidean(false), use_kl(false), quantize_threshold(0.5),
		smoother_length(7), smoother_median(0.5), sdtw_width(10),
		sdtw_budget(7.0), sdtw_trim(0.1) {}

	void Register(OptionsItf *po) {
		po.Register("use-cosine", &use_cosine,
				"Use cosine similarity between frames. Behavior may be "
				"unpredictable when multiple similarity measures are specified.");
		po.Register("use-dotprod", &use_dotprod,
				"Use dot product similarity between frames");
		po.Register("use-euclidean", &use_euclidean,
				"Use Euclidean similarity between frames");
		po.Register("use-kl", &use_kl,
				"Use KL similarity between frames");
		po.Register("quantize-threshold", &quantize_threshold,
				"Frame similarities below this value are set to 0");
		po.Register("smoother-length", &smoother_length,
				"Context radius of the diagonal median smoothing filter. Total "
				"filter length is twice this value plus one");
		po.Register("smoother-median", &smoother_median,
				"Mu parameter for the median smoothing filter");
		po.Register("sdtw-width", &sdtw_width,
				"S-DTW bandwidth parameter");
		po.Register("sdtw-budget", &sdtw_budget,
				"S-DTW distortion budget for each direction "
				"(forwards and backwards)");
		po.Register("sdtw-trim", &sdtw_trim,
				"Trim frames from the ends of each warp path whose similarity "
				"falls below this threshold");
	}
	void Check() const {
		KALDI_ASSERT(quantize_threshold >= 0 && smoother_length > 0
								 && smoother_median >= 0 && smoother_median <= 1
								 && sdtw_width > 1 && sdtw_budget > 0 && sdtw_trim >= 0
								 && (use_cosine || use_dotprod || use_euclidean || use_kl));
	}
};

class FastPatternSearcher {
public:
	FastPatternSearcher(const FastPatternSearcherConfig &config);

	void SetOptions(const FastPatternSearcherConfig &config) {
		config_ = config;
	}

	FastPatternSearcherConfig GetOptions() {
		return config_;
	}

	~FastPatternSearcher() {
	}

	// TODO: Check that I am passing in the pattern_writer properly (pointer)
	bool Search(const std::vector< Matrix<BaseFloat> &utt_features,
							const std::vector<std::string> &utt_ids,
							PatternStringWriter *pattern_writer);

private:
	void ComputeThresholdedSimilarityMatrix(
		const Matrix<BaseFloat> &first_features,
		const Matrix<BaseFloat> &second_features,
		SparseMatrix<BaseFloat> *similarity_matrix);

	void QuantizeSimilarityMatrix(
		const SparseMatrix<BaseFloat> &similarity_matrix,
		SparseMatrix<int32> *quantized_similarity_matrix);

	void ApplyMedianSmootherToMatrix(
		const SparseMatrix<int32> &input_matrix,
		SparseMatrix<int32> *median_smoothed_matrix);

	void ApplyGaussianBlurToMatrix(
		const SparseMatrix<int32> &input_matrix,
		SparseMatrix<BaseFloat> *blurred_matrix);

	void ComputeDiagonalHoughTransform(
		const SparseMatrix<BaseFloat> input_matrix,
		std::vector<BaseFloat> *hough_transform);

	void PickPeaksInVector(
		const std::vector<BaseFloat> &input_vector,
		const BaseFloat &peak_delta
		std::vector<size_t> *peak_locations);

	void ScanDiagsForLines(
		const SparseMatrix<BaseFloat> &input_matrix,
		const std::vector<int32> &diags_to_scan,
		std::vector<Line> *line_locations);

	void FilterBlockLines(
		const SparseMatrix<BaseFloat> &similarity_matrix,
		const std::vector<Line> &line_locations,
		const BaseFloat &block_filter_threshold,
		std::vector<Line> *filtered_line_locations);

	void WarpLinesToPaths(
		const SparseMatrix<BaseFloat> &similarity_matrix,
		const &line_locations,
		std::vector<Path> *sdtw_paths);

	void WritePaths(const vector<Path> &sdtw_paths,
									PatternStringWriter *writer);

	FastPatternSearcherConfig config_;
};

}  // end namespace kaldi

#endif