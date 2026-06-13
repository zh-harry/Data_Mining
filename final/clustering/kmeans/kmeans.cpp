#include "kmeans.h"
#include <cstdlib>
#include <random>
#include <limits>
#include <algorithm>

KMeans::KMeans(std::size_t k):k(k){}

std::vector<std::size_t> KMeans::fit(const std::vector<point>& train_data)
{
    std::size_t data_size = train_data.size();
    std::size_t dimension = train_data[0].feature.size();

    // random choose k centroids
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<std::size_t> idx(data_size);
    for(std::size_t i = 0;i < data_size;++i)
    {
        idx[i] = i;
    }
    std::shuffle(idx.begin(), idx.end(), gen);

    std::vector<point> centroids(this->k);
    for(std::size_t i = 0;i < this->k;++i)
    {
        centroids[i] = train_data[idx[i]];
    }

    // record the cluster assignment => cluster[i] = 0,1
    std::vector<std::size_t> cluster(data_size);

    while(true)
    {
        // cluster assignment
        for(std::size_t i = 0;i < data_size;++i)
        {
            double mn_dist = std::numeric_limits<double>::max();
            std::size_t mn_idx = 0;
            for(std::size_t j = 0;j < this->k;++j)
            {
                double dist = euclidean_distance(train_data[i],centroids[j]);
                if(dist < mn_dist)
                {
                    mn_dist = dist;
                    mn_idx = j;
                }
            }
            cluster[i] = mn_idx;
        }
    
        // re-compute centroids
        std::vector<point> new_centroids(this->k);
        for(std::size_t i = 0;i < this->k;++i)
        {
            new_centroids[i].feature.assign(dimension, 0.0);
        }
        std::vector<int> cnt(this->k, 0);
        for(std::size_t i = 0;i < data_size;++i)
        {
            std::size_t c = cluster[i];
            for(std::size_t j = 0;j < dimension;++j)
            {
                new_centroids[c].feature[j] += train_data[i].feature[j];
            }
            ++cnt[c];
        }
        for(std::size_t i = 0;i < this->k;++i)
        {
            if(cnt[i] > 0){
                for(std::size_t j = 0;j < dimension;++j)
                {
                    new_centroids[i].feature[j] /= cnt[i];
                }
            }
            else
            {
                // deal with empty cluster
                std::size_t mx_cnt = 0;
                std::size_t mx_idx = 0;
                for(std::size_t c = 0;c < this->k;++c)
                {
                    if(cnt[c] > mx_cnt)
                    {
                        mx_cnt = cnt[c];
                        mx_idx = c;
                    }
                }
                for(std::size_t p = 0;p < data_size;++p)
                {
                    if(cluster[p] == mx_idx)
                    {
                        new_centroids[i] = train_data[p];
                        cluster[p] = i; // avoid two empty cluster chooses same point
                        break;
                    }
                }
            }
        }

        // check convergence
        double tolerance = 1e-4;
        bool conv = true;
        for(std::size_t i = 0;i < this->k;++i)
        {
            double dist = euclidean_distance(new_centroids[i],centroids[i]);
            if(dist > tolerance)
            {
                conv = false;
            }
        }
        if(conv)
        {
            this->_centroids = centroids;
            break;
        }
        centroids = new_centroids;
    }
    return cluster;
}

double KMeans::euclidean_distance(const point& a, const point& b)
{
    double sum = 0;
    double diff = 0;
    for(size_t i = 0;i < a.feature.size();++i)
    {
        diff = a.feature[i] - b.feature[i];
        sum += diff * diff;
    }
    return sum;
}