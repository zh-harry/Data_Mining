#include <iostream>
#include<string>
#include "data.h"
#include "datareader.h"
#include "datatransformer.h"
#include "knn.h"

int main(int argc, char* argv[])
{
    dataset train_data, test_data;
    datareader reader;
    datatransformer transformer;

    // std::cout << "Loading the training data..." << std::endl;
    reader.read("../data/train.csv",train_data);
    // std::cout << "Loading the training data successfully" << std::endl;
    
    // std::cout << "Loading the testing data..." << std::endl;
    reader.read("../data/test.csv",test_data);
    // std::cout << "Loading the testing data successfully" << std::endl;

    transformer.fit(train_data);
    transformer.standardize(train_data);
    transformer.standardize(test_data);

    std::size_t k;
    k = std::stoi(argv[1]);

    KNN knn;
    std::vector<int> result;
    knn.train(train_data);
    result = knn.predict(test_data,k);

    std::size_t correct = 0;
    for(std::size_t i = 0;i < test_data.quality.size();++i)
    {
        if(result[i] == test_data.quality[i]) ++correct;
    }
    std::cout << "The accuracy at k = " << k << " is : ";
    std::cout << (double)correct/test_data.quality.size()*100 << std::endl;
    return 0;
}