// sdtw/segment-pattern-searcher.h

// Author: David Harwath

#ifndef KALDI_SDTW_SEGMENT_PATTERN_SEARCHER_H
#define KALDI_SDTW_SEGMENT_PATTERN_SEARCHER_H

#include <vector>

#include "base/kaldi-common.h"
#include "landmarks/landmark-utils.h"
#include "sdtw/sdtw-utils.h"
#include "util/common-utils.h"

namespace kaldi {

struct SegmentPatternSearcherConfig {
	int32 width;
	int32 min_length;
	BaseFloat extend;
	BaseFloat block_threshold;
	BaseFloat min_similarity;

	SegmentPatternSearcherConfig(): width(3), extend(0.75), min_length(30),
		block_threshold(0.6), min_similarity(0.6) {}

	void Register(OptionsItf *po) {
		po->Register("block-threshold", &block_threshold,
				"Filter out paths whose enclosing block has an average similarity"
				" greater than this value");
		po->Register("width", &width,
				"S-DTW bandwidth parameter");
		po->Register("extend", &extend,
				"Extend each warp path fragment to adjacent segments with similarity greater"
				" than this threshold");
		po->Register("min-length", &min_length,
				"Length-constrained minimum average parameter");
		po->Register("min-similarity", &min_similarity,
				"Discard warp paths whose similarity per frame (not segment) falls below this threshold")
	}
	void Check() const {
		KALDI_ASSERT(width > 1 && extend > 0 && min_length > 1 &&
			block_threshold >= 0 && min_similarity >= 0);
	}
};

class SegmentPatternSearcher {
public:
	SegmentPatternSearcher(const SegmentPatternSearcherConfig &config): config_(config) {
		config.Check();
	}

	void SetOptions(const SegmentPatternSearcherConfig &config) {
		config_ = config;
	}

	SegmentPatternSearcherConfig GetOptions() const {
		return config_;
	}

	~SegmentPatternSearcher() {
	}

	bool Search(const std::vector<Matrix<BaseFloat> > feats,
							const std::vector<std::vector<int32> > lengths,
							const std::vector<std::string> ids,
							PathWriter *pattern_writer) const;

	bool Search(const std::vector<Matrix<BaseFloat> > feats_a,
							const std::vector<std::vector<int32> > lengths_a,
							const std::vector<std::string> ids_a,
							const std::vector<Matrix<BaseFloat> > feats_b,
							const std::vector<std::vector<int32> > lengths_b,
							const std::vector<std::string> ids_b,
							PathWriter *pattern_writer) const;

	bool SearchOnePair(const Matrix<BaseFloat> &feats_a,
										 const std::vector<int32> &lengths_a,
										 const std::string &id_a,
										 const Matrix<BaseFloat> &feats_b,
										 const std::vector<int32> &lengths_b,
										 const std::string &id_b,
										 PathWriter *pattern_writer) const;

	Matrix<BaseFloat> L2NormalizeFeatures(const Matrix<BaseFloat> &features) const;

	void ComputeSimilarityMatrix(const Matrix<BaseFloat &first_features,
													 		 const Matrix<BaseFloat> &second_features,
													 		 Matrix<BaseFloat> *similarity_matrix) const;

	void SDTWWarp(const Matrix<BaseFloat> &similarity_matrix,
								const std::vector<int32> &row_segment_lengths,
								const std::vector<int32> &col_segment_lengths,
								const std::pair<size_t, size_t> &start_point,
								Path *path) const;

	void RefinePath(const Path &original_path, const std::vector<int32> &row_segment_lengths,
									const std::vector<int32> &col_segment_lengths, Path *result) const;

	void FilterBlockPaths(const Matrix<BaseFloat> &similarity_matrix,
												const std::vector<int32> &row_segment_lengths,
												const std::vector<int32> &col_segment_lengths,
												const std::vector<Path> &paths,
												std::vector<Path> *filtered_paths) const;

	void WritePaths(const std::vector<Path> &sdtw_paths,
									PathWriter *writer) const;

	void WriteOverlaidMatrix(const SparseMatrix<BaseFloat> &similarity_matrix,
													 const std::vector<Path> sdtw_paths, const std::string key,
													 SparseFloatMatrixWriter *matrix_writer) const;

private:
	SegmentPatternSearcherConfig config_;
};

}  // end namespace kaldi

#endif