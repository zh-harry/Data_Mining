#ifndef KNN_H
#define KNN_H
#include "data.h"

class KNN
{
public:
    void train(const dataset& dset);
    std::vector<int> predict(const dataset& test_data, std::size_t k);
    
    dataset train_data;
};

#endif