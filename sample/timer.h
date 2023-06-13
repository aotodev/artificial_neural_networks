#pragma once

#include <chrono>

class benchmark_timer
{
public:

    benchmark_timer(const std::string& text)
        : m_text(text)
    {
        m_start_point = std::chrono::high_resolution_clock::now();
    }

    ~benchmark_timer()
    {
        stop();
    }
    
private:
    void stop()
    {
        auto endPoint = std::chrono::high_resolution_clock::now();

        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(m_start_point).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(endPoint).time_since_epoch().count();

        auto duration = end - start;
        double ms = duration * 0.001;

        printf("\033[0;33;44mBENCHMARK[%s]: %.4fms\033[0m\n", m_text.c_str(), ms);

    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start_point;
    std::string m_text;
};
