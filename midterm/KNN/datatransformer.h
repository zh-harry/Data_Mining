#ifndef DATATRANSFORMER_H
#define DATATRANSFORMER_H
#include "data.h"
#include <vector>

class datatransformer
{
public:
    void fit(const dataset& train_data);
    void standardize(dataset& target_data);
private:
    std::vector<double> col_sigma;
    std::vector<double> col_mu;
};

#endif