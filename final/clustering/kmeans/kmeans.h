#ifndef KMEANS_H
#define KMEANS_H
#include "data.h"
#include <vector>
#include <string>

class KMeans
{
public:
    KMeans() = default;
    KMeans(std::size_t k);

    std::vector<std::size_t> fit(const std::vector<point>& train_data);
private:
    std::size_t k;
    std::vector<point> _centroids;
    double euclidean_distance(const point& a, const point& b);
};

#endif