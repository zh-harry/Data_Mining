#include "datatransformer.h"
#include <cmath>

void datatransformer::fit(const dataset& train_data)
{
    std::size_t colsize = train_data.data[0].characteristic.size();
    std::size_t rowsize = train_data.data.size();
    std::vector<double> sum(colsize);
    std::vector<double> sum2(colsize);

    col_mu.resize(colsize);
    col_sigma.resize(colsize);

    for(const auto& pnt : train_data.data)
    {
        for(std::size_t i = 0;i < colsize;++i)
        {
            sum[i] += pnt.characteristic[i];
            sum2[i] += pnt.characteristic[i] * pnt.characteristic[i];
        }
    }

    for(std::size_t i = 0;i < colsize;++i)
    {
        col_sigma[i] = sqrt((sum2[i] - sum[i]*sum[i]/rowsize)/rowsize);
        col_mu[i] = sum[i]/rowsize;
    }
}

void datatransformer::standardize(dataset& target_data)
{
    std::size_t colsize = target_data.data[0].characteristic.size();
    for(auto& pnt : target_data.data)
    {
        for(std::size_t i = 0;i < colsize;++i)
        {
            pnt.characteristic[i] = (pnt.characteristic[i] - col_mu[i])/col_sigma[i];
        }
    }
}