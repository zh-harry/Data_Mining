#include "data.h"
#include "dbscan.h"
#include <queue>

DBSCAN::DBSCAN(double eps, std::size_t min_samples)
{
    this->eps = eps;
    this->min_samples = min_samples;
}

std::vector<std::size_t> DBSCAN::fit(const std::vector<point>& train_data)
{
    std::size_t data_size = train_data.size();

    std::vector<bool> visited(data_size, false);
    std::vector<std::size_t> cluster(data_size, 0);
    std::size_t curr_cluster = 1;

    for(std::size_t i = 0;i < data_size;++i)
    {
        if(visited[i])
        {
            continue;
        }
        visited[i] = true;

        // find neighbors
        std::vector<std::size_t> neighbors;
        for(std::size_t j = 0;j < data_size;++j)
        {
            double dist = euclidean_distance(train_data[i],train_data[j]);
            if(dist < this->eps)
            {
                neighbors.push_back(j);
            }
        }
        std::size_t neighbors_size = neighbors.size();

        // noise point
        if(neighbors_size < this->min_samples)
        {
            cluster[i] = 0;
            continue;
        }

        // create new cluster (BFS)
        cluster[i] = curr_cluster;
        
        std::queue<std::size_t> neighbors_list;
        for(std::size_t j = 0;j < neighbors_size;++j)
        {
            neighbors_list.push(neighbors[j]);
        }

        while(!neighbors_list.empty())
        {
            std::size_t person = neighbors_list.front();
            neighbors_list.pop();
            
            if(!visited[person])
            {
                visited[person] = true;
                std::vector<std::size_t> neighbors_prime;
                for(std::size_t j = 0;j < data_size;++j)
                {
                    double dist = euclidean_distance(train_data[person],train_data[j]);
                    if(dist < this->eps)
                    {
                        neighbors_prime.push_back(j);
                    }
                }
                std::size_t neighbors_prime_size = neighbors_prime.size();
                if(neighbors_prime_size >= this->min_samples)
                {
                    for(std::size_t j = 0;j < neighbors_prime_size;++j)
                    {
                        if(!visited[neighbors_prime[j]])
                        {
                            neighbors_list.push(neighbors_prime[j]);
                        }
                    }
                }
            }
            if(cluster[person] == 0)
            {
                cluster[person] = curr_cluster;
            }
        }
        ++curr_cluster;
    }

    return cluster;
}

double DBSCAN::euclidean_distance(const point& a, const point& b)
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