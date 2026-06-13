#include "data.h"
#include <cmath>
#include <algorithm>

point::point(const std::vector<double>& f):feature(f){}

dataset::dataset(){}

dataset::dataset(const std::vector<point>& d,const std::vector<std::string>& l):
    data(d),label(l)
{}

std::size_t dataset::size(){
    return data.size();
}