Artificial Neural Network
=====

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://git.stabletec.com/utilities/vksbc/blob/master/LICENSE)
<br/>
Multi-threaded dense neural network from scratch in C++20 using AVX SIMD instructions (see technical overview below)

Sample time-series forcasting
=====
### Soybean futures
inputing the closing price of the past 8 days to predict the next one in the future.
<br/>
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
thead pool size == 8
iterations per epoch == 12

EPOCH 0 | cost(MSE): 5.7678
EPOCH 1 | cost(MSE): 0.8538
EPOCH 2 | cost(MSE): 0.2913
EPOCH 3 | cost(MSE): 0.1588
EPOCH 4 | cost(MSE): 0.0653

BENCHMARK[neural_network::fit]: 1534.7310ms
```
*Tested on AMD FX-8320E(2014), using only the base clock (3.2GHz), Arch Linux*
<br/>

Technical overview
=====
The main focus of this project is performance. Here below are some of the technics used to that end.
## Statically polymorphic functors
 - Initializer, optimizer, activation and loss are all features decided before compilation, making them perfect candidates for static polymorphism.
 - They are passed as functor parameters and the correct method is decided at compile-time through static polymorphism using the Curiously recurring template pattern.
 - Since its type is know at compile-time, it can be trivially inlined allowing further optimizations from the compiler.
 - Functors are stateless and passed by r-value.
 - It is also important to consider the benefits code locality can provide by taking advantage of the CPU's instruction cache.

Example code with disassemble:
```cpp
template<typename T>
struct activation_func
{
    int operator()(int input) { return ((T*)this)->operator()(input); }
};

struct relu : public activation_func<relu>
{
    int operator()(int input) { return input > 0 ? input : 0; }
};

struct sigmoid : public activation_func<sigmoid>
{
    int operator()(int input) { return 1.0f / (1.0f + (std::exp(-input))); }
};

template<typename Activation>
inline int test(activation_func<Activation>&& activation, int input)
{
    return activation(input);
}

int main()
{
    /* use cin to hide the input value from the compiler otherwise it will just calculate the result and move it the eax register! */
    int i;
    std::cin >> i;

    return test(relu(), i);
}
```
GCC 13 will generate the following x86 assembly with -O3:
```assembly
;generated on https://godbolt.org

main:
        sub     rsp, 24
        mov     edi, OFFSET FLAT:_ZSt3cin
        lea     rsi, [rsp+12]
        call    std::basic_istream<char, std::char_traits<char> >::operator>>(int&)

        ; ReLU
        mov     eax, DWORD PTR [rsp+12] ;set eax to the value gotten from cin
        xor     edx, edx ;set edx to zero
        test    eax, eax ;will set SF=1 if less than 0
        cmovs   eax, edx ;if SF==1 move edx (holding 0) to eax (return value)

        add     rsp, 24
        ret
```
As it can be seen, there are no calls besides std::cin; the ReLU instructions were completely inlined and optimized inside the main function's call stack. This would have been impossible with virtual functions or function pointers.
<br/>

## Low-lock
 All multithread applications need some form of synchronization to some extent. However, careful consideration about its implementation is needed when low-latency/high-throughput is a concern; overusing mutexes and atomics can hurt performance considerably and in some cases perform even worse than its single-threaded counter-part.

The *fit* function does not use any mutexes at all, instead:
- It generates one gradient vector for every thread in the pool, divides the data points accordingly and perform the forward pass concurrently without sharing any data.
- There will be one atomic unsigned int to serve as a flag, which each thread will signal with their respective bit when finished.
- When all flags have been set, combine the results into one single gradient before passing it to the optimizer to update the parameters.
- Unlike the rest of the model (see the cache-locality section), these individual gradient vectors will be put into a 2D vector; usually we want the data to be close together in memory, but not in this case, as it could result in false sharing (when different threads access different data that gets mapped to the same cache line).

*Note: On many platforms, atomics of primitive types will be lockless, making the model essentially lock-free. However, there are no guarantees that this will be the case across all platforms.*
<br/>

## SIMD
Single Instruction Multiple Data; the benefits to performance should be obvious.

What is usually not so obvious is that because the *_mm256_fmadd_ps* intrinsic is a fused operation (fused multiply-add), the intermediate result from the multiplication part has infinite precision and only after adding to the third element will it be rounded to nearest float32 values. This means that our models will not only be faster, but also have higher precision!
<br/>

## Data layout and cache-locality
*"Cache-lines are the key! Undoubtedly! If you make even single error in data layout, you will get 100% slower solution! No jokes!" - Dmitry Vyukov*

The layout of the data shapes the way we design our code. It is certainly one of the most important factors when it comes performance, and even more so in the case of Deep learning.

The model tries to keep data always close in memory, and to access it in the most cache-friendly way possible.

 - **The weight matrices are stored in column-major order.**
    - The input row-vector of each layer is multiplied by a weight matrix, meaning that we will calculate the dot product between the input vector with each of the columns of the matrix. If we were to structure this matrix in row-major order not only would it be extremely inefficient but also make it virtually impossible to compute with SIMD instructions.
 - **The weight matrices are stored in a single contiguous vector.**
    - Better avoid 2D arrays (arr[m][n]) as they are nothing more than an array of pointers. Besides the obvious cache-misses, it would also miss on other optimizations such as cache-line prefetches.

All the weights from all layers are stored on a single array, as are all the biases:

```c++
float weights[] = { w0[0,0], w0[1,0], ..., w0[n,0], w0[0,1], w0[1,1], ..., w0[m,n], w1[0,0], ..., w1[m,n], ..., wLast[m,n] };
```
***
