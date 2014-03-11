#ifndef
#define FAST_SDTW_PATTERN_SEARCHER_H

class PatternSearcher {
private:
	int L;  // The median smoother radius
	int W;  // S-DTW band parameter
	int smear_size;  // Gaussian smearing kernel size
	float delta;  // Peak picking parameter
	float distortion_budget;  // S-DTW distortion budget
	float trim_thresh;  // S-DTW path trim threshold  
	float quant_thresh;  // Similarity matrix quantization threshold
	float mu;  // Median smoother parameter
	vector<string> input_files;  // The files to search over

public:
	
};

#endif