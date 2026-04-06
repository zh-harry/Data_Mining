#ifndef DATA_H
#define DATA_H
#include <vector>

class point
{
public:
    point(const std::vector<double>& c);

    std::vector<double> characteristic;
};

class dataset
{
public:
    dataset();
    dataset(const std::vector<point>& p,const std::vector<int>& q);
    
    std::vector<point> data;
    std::vector<int> quality;
};

double euclidean_distance(const point& a, const point& b);

#endif