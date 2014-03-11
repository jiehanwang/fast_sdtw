#include <boost/program_options.hpp>
using namespace boost;
namespace po = boost::program_options;

#include <kaldi-matrix.h>

#include <iostream>
#include <algorithm>
#include <iterator>
using namespace std;

template<class T>
ostream& operator<<(ostream& os, const vector<T>& v) {
	copy(v.begin(), v.end(), ostream_iterator<T>(cout, " "));
	return os;
}

int main(int argc, char* argv[]) {
	try {
		int smoother_length, smoother_median, sdtw_width;
		float quantize_threshold, smoother_median, sdtw_budget, sdtw_trim;
		string measure;
		
		po::options_description desc("Allowed options");
		desc.add_options()
		("help", "produce help message")
		("measure", po::value<string>(&measure)->default_value("cosine"),
		 "similarity measure, can be: euclidean, cosine, kl, dot")
		("quantize", po::value<float>(&quantize_threshold)->default_value(0.5),
		 "similarity quantization threshold")
		("smoother-length", po::value<int>(&smoother_length)->default_value(7),
		 "median smoother radius")
		("smoother-median", po::value<float>(&smoother_median)->default_value(0.5),
		 "median smoother mu parameter")
		("SDTW-width", po::value<int>(&sdtw_width)->default_value(10),
		 "Segmental DTW Bandwidth parameter")
		("SDTW-budget", po::value<float>(&sdtw_budget)->default_value(10.0),
		 "Segmental DTW distortion budget")
		("SDTW-trim", po::value<float>(&sdtw_trim)->default_value(0.9),
		 "Segmental DTW edge trim threshold")
		("verbose", po::value<int>()->implicit_value(1),
		 "enable verbosity (optionally specify level)")
		("input-file", po::value< vector<string> >(),
			"input file, assumed to be in kaldi binary matrix format");
		
		po::positional_options_description p;
		p.add("input-file", -1);

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).
		options(desc).positional(p).run(), vm);
		po::notify(vm);

		if (vm.count("help")) {
			cout << "Usage: options_description [options]\n";
			cout << desc;
			return 0;
		}
		if (vm.count("input-file"))
		{
			cout << "Input files are: "
			<< vm["input-file"].as< vector<string> >() << "\n";
		}
		if (vm.count("verbose")) {
			cout << "Verbosity enabled. Level is " << vm["verbose"].as<int>()
			<< "\n";
		}
		cout << "Total median smoother length is " << (2 * L + 1) << "\n";
	}
	catch(std::exception& e) {
		cout << e.what() << "\n";
		return 1;
	}

	return 0;

}