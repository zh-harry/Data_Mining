#include "data.h"
#include "datareader.h"
#include "dbscan.h"
#include <iostream>
#include <unordered_map>

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

    // cluster for unknown types
    double eps = std::stod(argv[3]);
    std::size_t min_samples = static_cast<std::size_t>(std::stoull(argv[4]));
    DBSCAN dbscan(eps, min_samples);
    std::vector<std::size_t> cluster_idx = dbscan.fit(unknowns);

    // caluculate the total clusters
    std::size_t mx_cluster = 0;
    for(std::size_t i = 0;i < cluster_idx.size();++i)
    {
        if(cluster_idx[i] > mx_cluster)
        {
            mx_cluster = cluster_idx[i];
        }
    }

    // output result (cluster_idx)
    dataset final_result = classify_results;
    for(std::size_t i = 0;i < cluster_idx.size();++i)
    {
        std::size_t idx = unknowns_original_indx[i];
        final_result.label[idx] = std::to_string(cluster_idx[i]);
    }
    std::cout << "result (in cluster index):" << std::endl;
    for(std::size_t i = 0;i < original_size;++i)
    {
        std::cout << final_result.label[i] << std::endl;
    }

    // find corresponding label for each cluster
    dataset test_data;
    reader.read(argv[2], test_data);
    std::vector<std::unordered_map<std::string, int>> label_cnt;
    label_cnt.resize(mx_cluster + 1);
    for(std::size_t i = 0;i < unknowns_original_indx.size();++i)
    {
        std::size_t idx = unknowns_original_indx[i];
        std::size_t cluster_idx_ = static_cast<std::size_t>(std::stoull(final_result.label[idx]));
        ++label_cnt[cluster_idx_][test_data.label[idx]];
    }
    std::vector<int> mx_cnt(label_cnt.size(),0);
    std::vector<std::string> mx_label(label_cnt.size());
    mx_label[0] = "Noise";
    for(std::size_t i = 1;i < mx_cnt.size();++i)
    {
        for(auto& mp : label_cnt[i])
        {
            if(mp.second > mx_cnt[i])
            {
                mx_cnt[i] = mp.second;
                mx_label[i] = mp.first;
            }
        }
    }
    for(std::size_t i = 0;i < unknowns_original_indx.size();++i)
    {
        std::size_t idx = unknowns_original_indx[i];
        std::size_t cluster_idx_ = static_cast<std::size_t>(std::stoull(final_result.label[idx]));
        final_result.label[idx] = mx_label[cluster_idx_];
    }

    // output result 
    std::cout << "result (in cluster index):" << std::endl;
    std::cout << "==============================" << std::endl;
    for(std::size_t i = 0;i < mx_label.size();++i)
    {
        std::cout << "cluster " << i << ": ";
        std::cout << mx_label[i] << std::endl;
    }
    std::cout << "==============================" << std::endl;
    for(std::size_t i = 0;i < original_size;++i)
    {
        std::cout << final_result.label[i] << std::endl;
    }

    // calculate acc
    int acc_cnt = 0;
    for(std::size_t i = 0;i < test_data.size();++i)
    {
        if(final_result.label[i] == test_data.label[i])
        {
            ++acc_cnt;
        }
    }
    std::cout << "Overall Accuracy: " << (double)acc_cnt/test_data.size() << std::endl;
    return 0;
}