#define VERBOSITY 2

#include "../neural_network.hpp"

#include "soybean_data.h"
#include "timer.h"

int main()
{
    auto soybeanData = load_soybean_series("data/soybean_prices.csv", 12297);
    ann_data data;

    assert(!soybeanData.empty());

    /* set data */
    {
        /* window sizes */
        constexpr size_t inputWidth = 8;
        constexpr size_t outputWidth = 1;
        constexpr size_t windowWidth = inputWidth + outputWidth;

        data.training_count = soybeanData.size() - windowWidth;
        data.training_data = soybeanData.data();
        data.training_labels = data.training_data + inputWidth;

        data.validation_count = data.training_count / 6; // ~16%  // 2048
        data.validation_data = data.training_data + (data.training_count - data.validation_count);
        data.validation_labels = data.validation_data + inputWidth;
    }

    neural_network<8, 64, 256, 64, 1> model;
    model.initialize(he(), 0.1f);

    /* train mmodel */
    {
        benchmark_timer time("neural_network::fit");
        model.fit(data, relu(), mse(), adam(), false, 5, 1024);
    }
}
