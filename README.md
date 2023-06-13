Artificial Neural Network
=====

Multi-threaded dense neural network from scratch in C++20 using AVX SIMD instructions

***

## Sample result using soybean historical prices
inputing 8 days to predict the 9th <br />

 - About 1.5 seconds to get a a mse cost under 0.1 using a relatively old CPU:

Example code:
```cpp
    neural_network<8, 64, 256, 64, 1> model;
    model.initialize(he(), 0.1f);
    model.fit(data, relu(), mse(), adam(), false, 5, 1024);
```

Console output:
```text
----------------------------------------------------
neuron count            == 385
input/output size       == 8/1
----------------------------------------------------
layer[0] == [8, 64]
layer[1] == [64, 256]
layer[2] == [256, 64]
layer[3] == [64, 1]
----------------------------------------------------
total learnable parameters: 33729
of which 33344 weights and 385 biases
----------------------------------------------------

training count == 12288, minibatchSize == 1024
iterations per epoch == 12

EPOCH 0 | cost(MSE): 2.4070
EPOCH 1 | cost(MSE): 0.1338
EPOCH 2 | cost(MSE): 0.4088
EPOCH 3 | cost(MSE): 0.2019
EPOCH 4 | cost(MSE): 0.0712

BENCHMARK[neural_network::fit]: 1526.8950ms
```

 - tested on AMD FX-8320E(2014), using the base clock only (3.2GHz)

***
