#ifndef DATA_H
#define DATA_H
#include <vector>
#include <string>

class point
{
public:
    point() = default;
    point(const std::vector<double>& f);

    std::vector<double> feature;
};

class dataset
{
public:
    dataset();
    dataset(const std::vector<point>& d,const std::vector<std::string>& l);
    std::size_t size();

    std::vector<point> data;
    std::vector<std::string> label;
};

#endif