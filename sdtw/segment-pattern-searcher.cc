// sdtw/segment-pattern-searcher.cc

// Author: David Harwath

#include <algorithm>
#include <utility>
#include <vector>

#include "base/kaldi-common.h"
#include "matrix/kaldi-matrix.h"
#include "landmarks/landmark-utils.h"
#include "sdtw/segment-pattern-searcher.h"
#include "sdtw/sdtw-utils.h"
#include "util/common-utils.h"

namespace kaldi {

bool Search(const std::vector<Matrix<BaseFloat> > feats,
							const std::vector<std::vector<int32> > lengths,
							const std::vector<std::string> ids,
							PathWriter *pattern_writer) const {
	KALDI_ASSERT(pattern_writer != NULL);
	KALDI_ASSERT(feats.size() == lengths.size() && lengths.size() == ids.size());
	// Precompute L-2 normalized features
	std::vector<Matrix<BaseFloat> > normalized_feats;
	for (int32 i = 0; i < feats.size(); ++i) {
		const Matrix<BaseFloat> &f = feats[i];
		normalized_feats.push_back(L2NormalizeFeatures(f));
	}
	// Search between each unique pair of utterances
	for (int i = 0; i < normalized_feats.size() - 1; ++i) {
		for (int j = i + 1; j < normalized_feats.size(); ++j) {
			SearchOnePair(normalized_feats[i], lengths[i], ids[i], normalized_feats[j],
										lengths[j], ids[j], pattern_writer);
		}
	}
	return true;
}

bool Search(const std::vector<Matrix<BaseFloat> > feats_a,
						const std::vector<std::vector<int32> > lengths_a,
						const std::vector<std::string> ids_a,
						const std::vector<Matrix<BaseFloat> > feats_b,
						const std::vector<std::vector<int32> > lengths_b,
						const std::vector<std::string> ids_b,
						PathWriter *pattern_writer) const {
	KALDI_ASSERT(pattern_writer != NULL);
	KALDI_ASSERT(feats_a.size() == lengths_a.size() && lengths_a.size() == ids_a.size() &&
							 feats_b.size() == lengths_b.size() && lengths_b.size() == ids_b.size()));
	// Precompute L-2 normalized features
	std::vector<Matrix<BaseFloat> > normalized_feats_a;
	for (int32 i = 0; i < feats_a.size(); ++i) {
		const Matrix<BaseFloat> &f = feats_a[i];
		normalized_feats_a.push_back(L2NormalizeFeatures(f));
	}
		std::vector<Matrix<BaseFloat> > normalized_feats_b;
	for (int32 i = 0; i < feats_b.size(); ++i) {
		const Matrix<BaseFloat> &f = feats_b[i];
		normalized_feats_b.push_back(L2NormalizeFeatures(f));
	}
	// Search between each unique pair of <utt_a, utt_b> for utt_a in feats_a,
	// utt_b in feats_b
	for (int32 i = 0; i < normalized_feats_a.size(); ++i) {
		for (int32 j = 0; j < normalized_feats_b.size(); ++j) {
			SearchOnePair(normalized_feats_a[i], lengths_a[i], ids_a[i], normalized_feats_b[j],
										lengths_b[j], ids_b[j], pattern_writer);
		}
	}
	return true;
}

bool SearchOnePair(const Matrix<BaseFloat> &feats_a,
									 const std::vector<int32> &lengths_a,
									 const std::string &id_a,
									 const Matrix<BaseFloat> &feats_b,
									 const std::vector<int32> &lengths_b,
									 const std::string &id_b,
									 PathWriter *pattern_writer) const {
	Matrix<BaseFloat> similarity_matrix;
	std::vector<Path> paths;
	std::vector<std::pair<size_t, size_t> > start_points;
	for ( /* add row=0 start points */ ) {}
	for ( /* add col=0 start points */ ) {}
	ComputeSimilarityMatrix(feats_a, feats_b, &similarity_matrix);
	for (int32 i = 0; i < start_points.size(); ++i) {
		Path path;
		const std::pair<size_t, size_t> &start_point = start_points[i];
		SDTWWarp(similarity_matrix, lengths_a, lengths_b, start_point, &path);
		Path refined_path;
		RefinePath(path, lengths_a, lengths_b, &refined_path);
		BaseFloat avg_sim = 0.0;
		for (int32 i = 0; i < path.similarities.size(); ++i) {
			avg_sim += path.similarities[i];
		}
		avg_sim /= path.similarities.size();
		if (refined_path.avg_sim >= config_.min_similarity) {
			paths.push_back(path);
		}
	}
	std::vector<Path> filtered_paths;
	FilterBlockPaths(similarity_matrix, lengths_a, lengths_b, &paths, &filtered_paths);
	WritePaths(filtered_paths, pattern_writer);
}

Matrix<BaseFloat> L2NormalizeFeatures(const Matrix<BaseFloat> &features) const {
	const BaseFloat epsilon = 0.000000001;
	Matrix<BaseFloat> normalized_features(features);
	// Compute row magnitudes
	Vector<BaseFloat> row_mags(features.NumRows());
	for (MatrixIndexT row = 0; row < normalized_features.NumRows(); ++row) {
		BaseFloat mag = 0.0;
		for (MatrixIndexT col = 0; col < normalized_features.NumCols(); ++col) {
			mag += std::pow(normalized_features(row, col), 2);
		}
		row_mags(row) = 1.0 / (epsilon + std::sqrt(mag));
	}
	// Scale each row by its inverse magnitude
	normalized_features.MulRowsVec(row_mags);
	return normalized_features;
}

void ComputeSimilarityMatrix(const Matrix<BaseFloat &first_features,
												 		 const Matrix<BaseFloat> &second_features,
												 		 Matrix<BaseFloat> *similarity_matrix) const {
	KALDI_ASSERT(similarity_matrix != NULL);
	const std::pair<int32, int32> size =
		std::make_pair<int32, int32>(first_features.NumRows(), second_features.NumRows());
	similarity_matrix->Resize(size.first, size.second, kUndefined);
	similarity_matrix->AddMatMat(0.5, first_features, kNoTrans, second_features, kTrans, 0.0);
	similarity_matrix->Add(0.5);
}

void SDTWWarp(const Matrix<BaseFloat> &similarity_matrix,
							const std::vector<int32> &row_segment_lengths,
							const std::vector<int32> &col_segment_lengths,
							const std::pair<size_t, size_t> &start_point,
							Path *path) const {
	// Adapt this from fast-pattern-searcher.cc::WarpForward()
}

void RefinePath(const Path &original_path, const std::vector<int32> &row_segment_lengths,
								const std::vector<int32> &col_segment_lengths, Path *result) const {
	// LCMA followed by path extension
}

void FilterBlockPaths(const Matrix<BaseFloat> &similarity_matrix,
											const std::vector<int32> &row_segment_lengths,
											const std::vector<int32> &col_segment_lengths,
											const std::vector<Path> &paths,
											std::vector<Path> *filtered_paths) const {
	KALDI_ASSERT(filtered_paths != NULL);
	KALDI_ASSERT(similarity_matrix.NumRows() == row_segment_lengths.size());
	KALDI_ASSERT(similarity_matrix.NumCols() == col_segment_lengths.size());
	filtered_paths->clear();
	// The idea here is pretty simple. Given a path from (start_row, start_col)
	// to (end_row, end_col), compute the average similarity of the rectangle
	// enclosing the path. If the average similarity exceeds the specified
	// threshold, then do not include the line in the filtered list.
	for (int i = 0; i < paths.size(); ++i) {
		const Path &path = paths[i];
		if (path.path_points.size() > 0) {
			const std::pair<size_t, size_t> start = path.path_points[0];
			const std::pair<size_t, size_t> end = path.path_points.back();
			const size_t row_min = std::min(start.first, end.first);
			const size_t row_max = std::max(start.first, end.first);
			const size_t col_min = std::min(start.second, end.second);
			const size_t col_max = std::max(start.second, end.second);
			BaseFloat blocksum = 0.0;
			int32 total_blocksize = 0;
			for (size_t row = row_min; row <= row_max; ++row) {
				for (size_t col = col_min; col <= col_max; ++col) {
					const int32 blocksize = row_segment_lengths[row] * col_segment_lengths[col];
					blocksum += similarity_matrix(row, col) * blocksize;
					total_blocksize += blocksize;
				}
			}
			if ((blocksum / total_blocksize) < config_.block_threshold) {
				filtered_paths->push_back(path);
			}
		}
	}
}

void WritePaths(const std::vector<Path> &sdtw_paths,
								PathWriter *writer) const {
	KALDI_ASSERT(writer != NULL);
	KALDI_ASSERT(writer->IsOpen());
	for (int32 i = 0; i < sdtw_paths.size(); ++i) {
		const Path &path = sdtw_paths[i];
		std::stringstream sstm;
		sstm << "<" << path.first_id << "><" << path.second_id << ">-" << i;
		const std::string key = sstm.str();
		writer->Write(key, path);
	}
}

void WriteOverlaidMatrix(const SparseMatrix<BaseFloat> &similarity_matrix,
												 const std::vector<Path> sdtw_paths, const std::string key,
												 SparseFloatMatrixWriter *matrix_writer) const {}
