// sdtwbin/fast-sdtw.cc

// Author: David Harwath

#include <vector>

#include "base/kaldi-common.h"
#include "feat/feature-functions.h" // do I need this?
#include "matrix/kaldi-matrix.h" // ditto
#include "sdtw/fast-pattern-searcher.h"
#include "sdtw/sdtw-utils.h"
#include "util/common-utils.h"
#include "util/timer.h"

int main(int argc, char *argv[]) {
	try {
		using namespace kaldi;
		typedef kaldi::int32 int32;

		const char *usage =
				"Perform fast pattern discovery.\n"
				"Usage: fast-sdtw [options] features-a-rspecifier [features-b-rspecifier] patterns-wspecifier\n"
				"If only features-a-rspecifier is given, search between each utterance pair in features-a-rspecifier."
				"If both features-a-rspecifier and features-b-rspecifier are given, search between each pair (a, b)";
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
		std::string features_a_rspecifier, features_b_rspecifier, patterns_wspecifier;
		// Decide if there is one or two input feature sets to read
		if (po.NumArgs() == 2) {
			features_a_rspecifier = po.GetArg(1);
			patterns_wspecifier = po.GetArg(2);
		} else {
			features_a_rspecifier = po.GetArg(1);
			features_b_rspecifier = po.GetArg(2);
			patterns_wspecifier = po.GetArg(3);
		}
		PathWriter pattern_writer(patterns_wspecifier);
		// Read the first features
		std::vector<std::pair<std::string, Matrix<BaseFloat> > > feats_a;
		SequentialBaseFloatMatrixReader feats_a_reader(features_a_rspecifier);
		for (; !feats_a_reader.Done(); feats_a_reader.Next()) {
			std::string utt = feats_a_reader.Key();
			Matrix<BaseFloat> features(feats_a_reader.Value());
			feats_a_reader.FreeCurrent();
			if (features.NumRows() == 0) {
				KALDI_WARN << "Zero-length utterance: " << utt;
				continue;
			}
			feats_a.push_back(std::make_pair(utt, features));
		}
		// If there is a second set of features, read those then search. Otherwise, just search
		// using the first set.
		if (po.NumArgs() == 3) {
			std::vector<std::pair<std::string, Matrix<BaseFloat> > > feats_b;
			SequentialBaseFloatMatrixReader feats_b_reader(features_b_rspecifier);
			for (; !feats_b_reader.Done(); feats_b_reader.Next()) {
				std::string utt = feats_b_reader.Key();
				Matrix<BaseFloat> features(feats_b_reader.Value());
				feats_b_reader.FreeCurrent();
				if (features.NumRows() == 0) {
					KALDI_WARN << "Zero-length utterance: " << utt;
					continue;
				}
				feats_b.push_back(std::make_pair(utt, features));
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