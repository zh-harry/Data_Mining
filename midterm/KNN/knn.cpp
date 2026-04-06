#include "knn.h"
#include <algorithm>
#include <map>
#include <cmath>

void KNN::train(const dataset& dset)
{
    this->train_data = dset;
}

std::vector<int> KNN::predict(const dataset& test_data, std::size_t k)
{
    std::vector<int> result;

    for(const auto& pnt : test_data.data)
    {
        std::vector<std::pair<double,int>> dist(this->train_data.data.size());
        for(std::size_t i = 0;i < this->train_data.data.size();++i)
        {
            dist[i].first = euclidean_distance(pnt,this->train_data.data[i]);
            dist[i].second = this->train_data.quality[i];
        }

        sort(dist.begin(),dist.end());

        std::map<int, std::size_t> cnt;
        std::pair<std::size_t, int> mx{0,0};
        for(std::size_t i = 0;i < k;++i)
        {
            ++cnt[dist[i].second];
        }
        for(auto& j : cnt)
        {
            std::pair<std::size_t, int> cur{j.second, j.first};
            if(cur > mx)
            {
                mx = cur;
            }   
        }

        result.emplace_back(mx.second);
    }

    return result;
}