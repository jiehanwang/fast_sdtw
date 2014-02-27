/* -*- C++ -*-
 *
 * Author: David Harwath
 *
 */

#include <boost/program_options.hpp>

using namespace boost;
namespace po = boost::program_options;

#include <iostream>
#include <algorithm>
#include <iterator>

using namespace std;


 int main(int argc, char **argv) {
 	po::options_description desc("Allowed options");
 	desc.add_options()
 		("help", "Print help message")
 		("input-file", po::value< vector<string> >(), "input file")
 	;

 	po::positional_options_description p;
 	p.add("input-file", -1);

 	po::variables_map vm;
 	po::store(po::command_line_parser(argc, argv).
 			  options(desc).positional(p).run(), vm);
 	po::notify(vm);

 	if (vm.count("help")) {
 		cout << desc << endl;
 		return 1;
 	}

	if (vm.count("input-file")) {
		//const vector<string> &input_files = vm["intput-file"].as< vector<string> >();
    	cout << "Input files are: ";
    	for (vector<string>::const_iterator it = vm["intput-file"].as< vector<string> >().begin();
    		 it != vm["intput-file"].as< vector<string> >().end(); ++it) {
    		cout << *it << ' ';
    	}
    	cout << endl;
	}

 	return 0;
 }