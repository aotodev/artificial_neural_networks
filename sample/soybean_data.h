#pragma once

#include "../core.hpp"

#include <string>

#include <vector>

#include <iostream>
#include <fstream>
#include <sstream>

/* just a helper to load our soybean data */
inline vector_t load_soybean_series(const std::string& path, uint32_t obervationCount = 0)
{
    /* RAII, no need to close it */
    std::ifstream csv_stream(path);

    if (csv_stream.fail())
        LOG_ERROR("failed to load file iris data");

    std::string line;

    vector_t outData;
    outData.reserve(obervationCount ? obervationCount : 1024);

    while (std::getline(csv_stream, line))
        outData.emplace_back(std::stof(line));

    /* release any extra memory if needed */
    outData.shrink_to_fit();

    LOG_VERBOSE("loaded data size == %zu", outData.size());

    return outData;
}
