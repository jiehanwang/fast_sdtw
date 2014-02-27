/* -*- C++ -*-
 *
 * Author: David Harwath
 *
 */

#include <boost/program_options.hpp>

using namespace boost;
namespace po = boost::program_options;

#include <iostream>

using namespace std;


 int main(int argc, char **argv) {
 	po::options_description desc("Allowed options");
 	desc.add_options()
 		("help", "Print help message")
 		("input-file", po::value< vector<string> >(), "input file")
 	;

 	po::positional_options_descirption p;
 	p.add("input-file", -1);

 	po::variables_map vm;
 	po::store(po::command_line_parser(argc, argv).
 			  options(desc).positional(p).run(), vm);
 	po::notify(vm);

 	if (vm.count("help")) {
 		cout << desc << "\n";
 		return 1;
 	}

	if (vm.count("input-file"))
	{
    	cout << "Input files are: " 
       	<< vm["input-file"].as< vector<string> >() << "\n";
	}

 	return 0;
 }