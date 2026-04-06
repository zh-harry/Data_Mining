#include "data.h"
#include "datareader.h"
#include <fstream>
#include <iostream>
#include <string>

static std::vector<std::string> split(const std::string& s,const std::string& delim)
{
    std::vector<std::string> result;
    std::string::size_type begin,end;
    
    begin = 0;
    end = s.find(delim);
    while(end != std::string::npos)
    {
        if(begin != end)
        {
            result.emplace_back(s.substr(begin,end-begin));
        }
        begin = end + delim.length();
        end = s.find(delim,begin);
    } 
    if(begin != s.length())
    {
        result.emplace_back(s.substr(begin));
    }

    return result;
}

void datareader::read(const std::string& filename, dataset& dset)
{
    std::ifstream ifs(filename);
    if(!ifs.is_open())
    {
        std::cerr << "File name invalid" << std::endl;
        exit(1);
    }

    std::string s;
    while(ifs >> s)
    {
        std::vector<std::string> datas;
        datas = split(s,",");

        int quality = std::stoi(datas.back());
        datas.pop_back();

        std::vector<double> pnt;
        for(auto& data : datas)
        {
            double value = std::stod(data);
            pnt.emplace_back(value);
        }
        dset.data.emplace_back(pnt);
        dset.quality.emplace_back(quality);
    }

    ifs.close();
}