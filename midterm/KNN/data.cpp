#include "data.h"
#include <cmath>
#include <algorithm>

point::point(const std::vector<double>& c):characteristic(c){}

dataset::dataset(){}

dataset::dataset(const std::vector<point>& p,const std::vector<int>& q):
    data(p),quality(q)
{}

double euclidean_distance(const point& a, const point& b)
{
    double sum = 0;
    double diff = 0;
    for(size_t i = 0;i < a.characteristic.size();++i)
    {
        diff = a.characteristic[i] - b.characteristic[i];
        sum += diff * diff;
    }
    return sum;
}
