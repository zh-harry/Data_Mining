#include "data.h"
#include "datareader.h"
#include "kmeans.h"
#include <iostream>

int main(int argc, char* argv[]){
    dataset classify_results;
    datareader reader;

    reader.read(argv[1], classify_results);

    // extract unkown classification
    std::vector<point> unknowns;
    std::vector<std::size_t> unknowns_original_indx;
    std::size_t original_size = classify_results.size();
    for(std::size_t i = 0;i < original_size;++i)
    {
        if(classify_results.label[i] == "Unknown")
        {
            unknowns.push_back(classify_results.data[i]);
            unknowns_original_indx.push_back(i);
        }
    }

    // cluster for two extra types
    KMeans kmeans(2);
    std::vector<std::size_t> labels = kmeans.fit(unknowns);

    // output result (labels)
    for(std::size_t i = 0;i < labels.size();++i)
    {
        std::size_t idx = unknowns_original_indx[i];
        classify_results.label[idx] = std::to_string(labels[i]);
    }
    for(std::size_t i = 0;i < original_size;++i)
    {
        std::cout << classify_results.label[i] << std::endl;
    }

    return 0;
}