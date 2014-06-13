// sdtwbin/segment-sdtw.cc

// Author: David Harwath

#include <vector>

#include "base/kaldi-common.h"
#include "feat/feature-functions.h" // do I need this?
#include "matrix/kaldi-matrix.h"
#include "landmarks/landmark-utils.h"
#include "sdtw/fast-pattern-searcher.h"
#include "sdtw/sdtw-utils.h"
#include "util/common-utils.h"
#include "util/timer.h"

namespace kaldi {
bool SegmentSeqToMatrix(const SegmentSeq &segs, Matrix<BaseFloat> *mat) {
	KALDI_ASSERT(mat != NULL);
	KALDI_ASSERT(segs.segs.size() > 0);
	const int32 feat_dim = segs.segs[0].feats.size();
	const int32 num_segs = segs.segs.size();
	mat->Resize(num_segs, feat_dim);
	for (int32 row = 0; row < num_segs; ++row) {
		const Segment &seg = segs.segs[row];
		KALDI_ASSERT(seg.feats.size() == feat_dim);
		BaseFloat *row_data = mat->RowData(row);
		for (int32 col = 0; col < feat_dim; ++col) {
			row_data[col] = seg.feats[col];
		}
	}
	return true;
}

bool SegmentSeqToVectorOfLengths(const SegmentSeq &segs, std::vector<int32> *lengths) {
	KALDI_ASSERT(lengths != NULL);
	KALDI_ASSERT(segs.segs.size() > 0);
	lengths->clear();
	for (int32 i = 0; i < segs.segs.size(); ++i) {
		lengths->push_back(segs.segs[i].num_frames);
	}
	return true;
}
}  // end namespace kaldi

int main(int argc, char *argv[]) {
	try {
		using namespace kaldi;
		const char *usage =
				"Perform segment-based pattern discovery.\n"
				"Usage: segment-sdtw [options] segmentseqs-a-rspecifier [segmentseqs-b-rspecifier] patterns-wspecifier\n"
				"If only segmentseqs-a-rspecifier is given, search between each utterance pair in segmentseqs-a-rspecifier.\n"
				"If both segmentseqs-a-rspecifier and segmentseqs-b-rspecifier are given, search between each pair (a, b)";
		ParseOptions po(usage);
		Timer timer;
		SegmentPatternSearcherConfig config;
		config.Register(&po);
		po.Read(argc, argv);
		if (po.NumArgs() < 2 || po.NumArgs() > 3) {
			po.PrintUsage();
			exit(1);
		}
		SegmentPatternSearcher searcher(config);
		std::string segmentseqs_a_rspecifier, segmentseqs_b_rspecifier, patterns_wspecifier;
		// Decide if there is one or two input feature sets to read
		if (po.NumArgs() == 2) {
			segmentseqs_a_rspecifier = po.GetArg(1);
			patterns_wspecifier = po.GetArg(2);
		} else {
			segmentseqs_a_rspecifier = po.GetArg(1);
			segmentseqs_b_rspecifier = po.GetArg(2);
			patterns_wspecifier = po.GetArg(3);
		}
		PathWriter pattern_writer(patterns_wspecifier);
		// Read the first features
		std::vector<Matrix<BaseFloat> > feats_a;
		std::vector<std::string> ids_a;
		std::vector<std::vector<int32> > lengths_a;
		SequentialSegmentSeqReader segs_a_reader(segmentseqs_a_rspecifier);
		for (; !segs_a_reader.Done(); segs_a_reader.Next()) {
			std::string utt_id = segs_a_reader.Key();
			SegmentSeq segs(segs_a_reader.Value());
			segs_a_reader.FreeCurrent();
			if (segs.segs.size() == 0) {
				KALDI_WARN << "Zero-length utterance: " << utt;
				continue;
			}
			Matrix<BaseFloat> mat;
			SegmentSeqToMatrix(segs, &mat);
			std::vector<int32> lengths;
			SegmentSeqToVectorOfLengths(segs, &lengths);
			feats_a.push_back(mat);
			lengths_a.push_back(lengths);
			ids_a.push_back(utt_id);
		}
		// If there is a second set of features, read those then search. Otherwise, just search
		// using the first set.
		if (po.NumArgs() == 3) {
			std::vector<Matrix<BaseFloat> > feats_b;
			std::vector<std::string> ids_b;
			std::vector<std::vector<int32> > lengths_b;
			SequentialSegmentSeqReader segs_b_reader(segmentseqs_b_rspecifier);
			for (; !segs_b_reader.Done(); segs_b_reader.Next()) {
				std::string utt_id = segs_b_reader.Key();
				SegmentSeq segs(segs_b_reader.Value());
				segs_b_reader.FreeCurrent();
				if (segs.segs.size() == 0) {
					KALDI_WARN << "Zero-length utterance: " << utt;
					continue;
				}
				Matrix<BaseFloat> mat;
				SegmentSeqToMatrix(segs, &mat);
				std::vector<int32> lengths;
				SegmentSeqToVectorOfLengths(segs, &lengths);
				feats_b.push_back(mat);
				lengths_b.push_back(lengths);
				ids_b.push_back(utt_id);
			}
			searcher.Search(feats_a, lengths_a, ids_a, feats_b, lengths_b, ids_b, &pattern_writer);
		} else {
			searcher.Search(feats_a, lengths_a, ids_a, &pattern_writer);
		}
		double elapsed = timer.Elapsed();
		KALDI_LOG << "Time taken: " << elapsed << " seconds.";
	} catch(const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}