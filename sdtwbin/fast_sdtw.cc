// sdtwbin/fast-sdtw.cc

// Author: David Harwath

#include "base/kaldi-common.h"
#include "feat/feature-functions.h" // do I need this?
#include "sdtw/fast-pattern-searcher.h"
#include "util/common-utils"
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
		std::string distance_measure = "cosine";
		BaseFloat quantize_threshold = 0.5;
		int32 smoother_length = 7;
		BaseFloat smoother_median = 0.5;
		int32 sdtw_width = 10;
		BaseFloat sdtw_budget = 15.0;
		BaseFloat sdtw_trim = 0.2;
		FastPatternSearcherConfig config;

		config.Register(&po);
		po.Register("distance-measure", &distance-measure,
				"Function to compute frame-wise distances. Must be one of: cosine, "
				"euclidean, kl, dotproduct");
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
		po.Read(argc, argv);

		if (po.NumArgs() != 2) {
			po.PrintUsage();
			exit(1);
		}

		std::string features_rspecifier = po.GetArg(1),
								patterns_wspecifier = po.GetArg(2);

		int32 num_err = 0;
		vector<std::string> utt_ids;
		vector< Matrix<BaseFloat> > utt_features;

		SequentialBaseFloatMatrixReader feature_reader(features_rspecifier);
		for (; !feature_reader.Done(); feature_reader.Next()) {
			std::string utt = feature_reader.Key();
			Matrix<BaseFloat> features (feature_reader.Value());
			feature_reader.FreeCurrent(); // Do I need this? What does this do?
			if (feature.NumRows() == 0) {
				KALDI_WARN << "Zero-length utterance: " << utt;
				num_err++;
				continue;
			}
			utt_ids.push_back(utt);
			utt_features.push_back(features);
		}

		FastPatternSearcher searcher(config);
		searcher.search(utt_features, utt_ids, patterns_wspecifier);

		double elapsed = timer.Elapsed();
		KALDI_LOG << "Time taken: " << elapsed " seconds.";

	} catch(const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}