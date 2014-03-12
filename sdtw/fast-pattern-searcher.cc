// sdtw/fast-pattern-searcher.cc

// Author: David Harwath

#include "sdtw/fast-pattern-searcher.h"

namespace kaldi {

FastPatternSearcher::FastPatternSearcher(
				const FastPatternSearcherConfig &config): config_(config) {
	config.Check();
	// TODO: Set the similarity measure to use here
}

bool FastPatternSearcher::Search(
				const vector< Matrix<BaseFloat> &utt_features,
				const vector<std::string> &utt_ids,
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
			
		}
	}
}

}  // end namespace kaldi