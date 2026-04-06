#ifndef DATAREADER_H
#define DATAREADER_H
#include "data.h"
#include <string>

class datareader
{
public:
    void read(const std::string& filename, dataset& dset);
};

#endif