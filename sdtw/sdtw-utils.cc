// sdtw/stdw-utils.cc

// Author: David Harwath

#include <iostream>
#include <map>
#include <utility>
#include <vector>

#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include "sdtw/sdtw-utils.h"

namespace kaldi {

BaseFloat CosineSimilarity(const Vector<BaseFloat> &first,
													 const Vector<BaseFloat> &second){
	Vector<BaseFloat> f = first;
	Vector<BaseFloat> s = second;
	f.Scale(1.0 / f.Norm(2));
	s.Scale(1.0 / f.Norm(2));
	f.MulElements(s);
	return f.Sum();
}

BaseFloat KLSimilarity(const Vector<BaseFloat> &first,
											 const Vector<BaseFloat> &second){
	// TODO: Implement this.
	return 0.0;
}

BaseFloat DotProdSimilarity(const Vector<BaseFloat> &first,
														const Vector<BaseFloat> &second){
	Vector<BaseFloat> f = first;
	f.MulElements(second);
	return f.Sum();
}

void Path::Read(std::istream &in_stream, bool binary) {
  int32 path_length;
  std::string id1, id2;
  ExpectToken(in_stream, binary, "<ID1>");
  ReadToken(in_stream, binary, &id1);
  ExpectToken(in_stream, binary, "<ID2>");
  ReadToken(in_stream, binary, &id2);
  ExpectToken(in_stream, binary, "<LENGTH>");
  ReadBasicType(in_stream, binary, &path_length);
  similarities.clear();
  path_points.clear();
  similarities.reserve(path_length);
  path_points.reserve(path_length);
  for (int32 i = 0; i < path_length; ++i) {
  	size_t first, second;
  	BaseFloat similarity;
  	ReadBasicType(in_stream, binary, &first);
  	ReadBasicType(in_stream, binary, &second);
  	ReadBasicType(in_stream, binary, &similarity);
    similarities.push_back(similarity);
    path_points.push_back(std::make_pair(first, second));
  }
}

void Path::Write(std::ostream &out_stream, bool binary) const {
  int32 path_length = this->path_points.size();
  KALDI_ASSERT(path_length == this->similarities.size());
  if (path_length == 0) {
    KALDI_WARN << "Trying to write empty Path object.";
  }
  WriteToken(out_stream, binary, "<ID1>");
  WriteToken(out_stream, binary, first_id);
  WriteToken(out_stream, binary, "<ID2>");
  WriteToken(out_stream, binary, second_id);
  WriteToken(out_stream, binary, "<LENGTH>");
 	WriteBasicType(out_stream, binary, path_length);
  for (int32 i = 0; i < path_length; ++i) {
    size_t first = path_points[i].first;
    size_t second = path_points[i].second;
    BaseFloat similarity = similarities[i];
    WriteBasicType(out_stream, binary, first);
    WriteBasicType(out_stream, binary, second);
    WriteBasicType(out_stream, binary, similarity);
  }
}

}  // end namespace kaldi