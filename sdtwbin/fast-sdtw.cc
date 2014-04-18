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
				"Usage: fast-sdtw [options] features-rspecifier patterns-wspecifier";
		ParseOptions po(usage);
		Timer timer;
		FastPatternSearcherConfig config;
		config.Register(&po);
		po.Read(argc, argv);
		if (po.NumArgs() != 2) {
			po.PrintUsage();
			exit(1);
		}
		std::string features_rspecifier = po.GetArg(1),
								patterns_wspecifier = po.GetArg(2);
		int32 num_err = 0;
		std::vector<std::string> utt_ids;
		std::vector< Matrix<BaseFloat> > utt_features;
		SequentialBaseFloatMatrixReader feature_reader(features_rspecifier);
		PathWriter pattern_writer(patterns_wspecifier);
		for (; !feature_reader.Done(); feature_reader.Next()) {
			std::string utt = feature_reader.Key();
			Matrix<BaseFloat> features(feature_reader.Value());
			feature_reader.FreeCurrent(); // Do I need this? What does this do?
			if (features.NumRows() == 0) {
				KALDI_WARN << "Zero-length utterance: " << utt;
				num_err++;
				continue;
			}
			utt_ids.push_back(utt);
			utt_features.push_back(features);
		}

		FastPatternSearcher searcher(config);
		searcher.Search(utt_features, utt_ids, &pattern_writer);

		double elapsed = timer.Elapsed();
		KALDI_LOG << "Time taken: " << elapsed << " seconds.";

	} catch(const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}