#pragma once

#include "core.hpp"

#include <cmath>
#include <assert.h>

#include <immintrin.h>
#include "vendor/avx_mathfun.h"


template<typename T>
struct activation
{
	/* operates on single value */
	float operator()(float input)
	{
		return ((T*)this)->operator()(input);
	}

	/* operates on single value */
	float derivative(float inValue)
	{
		return ((T*)this)->derivative(inValue);
	}

	/* operates on an entire layer using simd */
	void add_bias_activation(float* outVec, const float* inBiases, size_t n)
	{
		((T*)this)->add_bias_activation(outVec, inBiases, n);
	}

	static activation<T> static_class()
	{
		return activation<T>();
	}
};

struct sigmoid : public activation<sigmoid>
{
	float operator()(float input)
	{
		return 1.0f / (1.0f + (std::exp(-input)));
	}

	float derivative(float inValue)
	{
		return inValue * (1.0f - inValue);
	}

	void add_bias_activation(float* outVec, const float* inBiases, size_t n)
	{
        size_t leftover = n % 8;

        for (size_t i = 0; i < (n - leftover); i+=8)
        {
            __m256 temp = _mm256_add_ps(_mm256_loadu_ps(&outVec[i]), _mm256_loadu_ps(&inBiases[i]));
			_mm256_mul_ps(temp, _mm256_set1_ps(-1.0f));

			temp = exp256_ps(temp);
			temp = _mm256_add_ps(temp, _mm256_set1_ps(1.0f));

            _mm256_storeu_ps(&outVec[i], _mm256_div_ps(_mm256_set1_ps(1.0f), temp));
        }

        if (leftover)
        {
            size_t offset = n - leftover;
            for(size_t i = offset; i < n; i++)
                outVec[i] = 1.0f / (1.0f + (std::exp(-1.0f * (outVec[i] + inBiases[i]))));
        }		
	}
};

// tanh
struct hyperbolic_tan : public activation<hyperbolic_tan>
{
	float operator()(float input)
	{
		return std::tanh(input);
	}

	float derivative(float inValue)
	{
		return 1.0f - std::pow(inValue, 2);
	}

	void add_bias_activation(float* outVec, const float* inBiases, size_t n)
	{
        size_t leftover = n % 8;

		/* there is no simd instruction for tanh (only intel extensions)
		 * so we'll use exp256 to get the value through the definition:
		 * tanh(x) == (exp(2x) - 1) / (exp(2x) + 1) 
		 */

        for (size_t i = 0; i < (n - leftover); i+=8)
        {
			/* add bias */ 
			__m256 output = _mm256_add_ps(_mm256_loadu_ps(&outVec[i]), _mm256_loadu_ps(&inBiases[i]));

			/* multiply by 2 (we need 2x) */
			output = _mm256_mul_ps(output, _mm256_set1_ps(2.0f));

			/* raise e to 2x */
			__m256 exp_ = exp256_ps(output);

			/* (exp(2x) - 1) */
			__m256 dividend = _mm256_sub_ps(exp_, _mm256_set1_ps(1.0f));

			/* (exp(2x) + 1) */
			__m256 divisor = _mm256_add_ps(exp_, _mm256_set1_ps(1.0f));

			/* divide and store the result */ 
            _mm256_storeu_ps(&outVec[i], _mm256_max_ps(dividend, divisor));
        }

        if (leftover)
        {
            size_t offset = n - leftover;
            for(size_t i = offset; i < n; i++)
                outVec[i] = std::tanh(outVec[i] + inBiases[i]);
        }		
	}
};

struct relu : public activation<relu>
{
	float operator()(float input)
	{
		return std::max(0.0f, input);
	}

	float derivative(float inValue)
	{
		return inValue > 0.0f ? 1.0f : 0.0f;
	}

	void add_bias_activation(float* outVec, const float* inBiases, size_t n)
	{
        const __m256 zero_vector(_mm256_setzero_ps());
        size_t leftover = n % 8;

        for (size_t i = 0; i < (n - leftover); i+=8)
        {
            __m256 sum = _mm256_add_ps(_mm256_loadu_ps(&outVec[i]), _mm256_loadu_ps(&inBiases[i]));
            _mm256_storeu_ps(&outVec[i], _mm256_max_ps(sum, zero_vector));
        }

        if (leftover)
        {
            size_t offset = n - leftover;
            for(size_t i = offset; i < n; i++)
                outVec[i] = std::max(outVec[i] + inBiases[i], 0.0f);
        }		
	}
};

struct leaky_relu : public activation<leaky_relu>
{
	leaky_relu(float alpha = 0.01f) : m_alpha(alpha) {}

	float operator()(float input)
	{
		return input > 0 ? input : input * m_alpha;
	}

	float derivative(float inValue)
	{
		return inValue > 0 ? 1.0f : m_alpha;
	}

	void add_bias_activation(float* outVec, const float* inBiases, size_t n)
	{
		/* there is no good way to perform leaky relu with simd instructions
		 * so we'll just add the bias using instrinsics and the leaky relu activation will be serial
		 */

        const __m256 zero_vector(_mm256_setzero_ps());
        size_t leftover = n % 8;

        for (size_t i = 0; i < (n - leftover); i+=8)
        {
            __m256 sum = _mm256_add_ps(_mm256_loadu_ps(&outVec[i]), _mm256_loadu_ps(&inBiases[i]));
            _mm256_storeu_ps(&outVec[i], sum);
        }

        if (leftover)
        {
            size_t offset = n - leftover;
            for(size_t i = offset; i < n; i++)
                outVec[i] = outVec[i] + inBiases[i];
        }

		/* perform leaky relu */
		for (size_t i = 0; i < n; i++)
        {
			float output = outVec[i];
            outVec[i] = output > 0 ? output : output * m_alpha;
        }		
	}

private:
	float m_alpha;
};

