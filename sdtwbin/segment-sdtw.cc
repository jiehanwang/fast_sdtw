// sdtwbin/segment-sdtw.cc

// Author: David Harwath

#include <vector>

#include "base/kaldi-common.h"
#include "feat/feature-functions.h" // do I need this?
#include "matrix/kaldi-matrix.h" // ditto
#include "sdtw/fast-pattern-searcher.h"
#include "sdtw/sdtw-utils.h"
#include "util/common-utils.h"
#include "util/timer.h"

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
		FastPatternSearcherConfig config;
		config.Register(&po);
		po.Read(argc, argv);
		if (po.NumArgs() < 2 || po.NumArgs() > 3) {
			po.PrintUsage();
			exit(1);
		}
		FastPatternSearcher searcher(config);
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
		std::vector<std::pair<std::string, Matrix<BaseFloat> > > feats_a;
		SequentialSegmentSeqReader segs_a_reader(segmentseqs_a_rspecifier);
		for (; !segs_a_reader.Done(); segs_a_reader.Next()) {
			std::string utt = segs_a_reader.Key();
			SegmentSeq segs(segs_a_reader.Value());
			segs_a_reader.FreeCurrent();
			if (segs.segs.size() == 0) {
				KALDI_WARN << "Zero-length utterance: " << utt;
				continue;
			}
			Matrix<BaseFloat> mat;
			SegmentSeqToMatrix(segs, &mat);
			feats_a.push_back(std::make_pair(utt, mat));
		}
		// If there is a second set of features, read those then search. Otherwise, just search
		// using the first set.
		if (po.NumArgs() == 3) {
			std::vector<std::pair<std::string, Matrix<BaseFloat> > > feats_b;
			SequentialSegmentSeqReader segs_b_reader(segmentseqs_b_rspecifier);
			for (; !segs_b_reader.Done(); segs_b_reader.Next()) {
				std::string utt = segs_b_reader.Key();
				SegmentSeq segs(segs_b_reader.Value());
				segs_b_reader.FreeCurrent();
				if (segs.segs.size() == 0) {
					KALDI_WARN << "Zero-length utterance: " << utt;
					continue;
				}
				Matrix<BaseFloat> mat;
				SegmentSeqToMatrix(segs, &mat);
				feats_b.push_back(std::make_pair(utt, mat));
			}
			searcher.Search(feats_a, feats_b, &pattern_writer);
		} else {
			searcher.Search(feats_a, &pattern_writer);
		}
		double elapsed = timer.Elapsed();
		KALDI_LOG << "Time taken: " << elapsed << " seconds.";
	} catch(const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}