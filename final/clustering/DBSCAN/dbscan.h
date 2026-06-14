#ifndef DBSCAN_H
#define DBSCAN_H
#include "data.h"

class DBSCAN
{
public:
    DBSCAN() = default;
    DBSCAN(double eps, std::size_t min_samples);

    std::vector<std::size_t> fit(const std::vector<point>& train_data);
private:
    double eps;
    std::size_t min_samples;
    double euclidean_distance(const point& a, const point& b);   
};

#endif