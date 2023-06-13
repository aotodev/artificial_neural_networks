#pragma once

#include "core.hpp"

#include <cmath>
#include <assert.h>

#include <algorithm>
#include <random>

inline constexpr float absolute(float inValue)
{
	return inValue * (-1.0f + 2.0f * (float)(inValue > 0.0f));
}

template<typename T>
struct initializer
{
	template<size_t n>
	void operator()(float* outWeights, const layout_t<n>& layout)
	{
		((T*)this)->operator()(outWeights, layout);
	}

	constexpr const char* tag()
	{
		return ((T*)this)->tag();
	}
};

/* usually for sigmoid & tanh */
struct xavier : public initializer<xavier>
{
	template<size_t n>
	void operator()(float* outWeights, const layout_t<n>& layout)
	{
		/* xavier uses the uniform distribution for its random generator */

		std::random_device randomDevice;
		std::default_random_engine randomEngine(randomDevice());
		std::uniform_real_distribution<float> uniform_distribution(0, 1);
		uint32_t weightsPtr = 0;

		for (auto& [inputSize, outputSize] : layout)
		{
			const float upper = 1.0f / (std::sqrt((float)inputSize));
			const float lower = -1.0f * upper;

			for (uint32_t i = 0; i < (inputSize * outputSize); ++i)
			{
				outWeights[weightsPtr++] = lower + uniform_distribution(randomEngine) * (upper - lower);
			}
		}
	}

	constexpr const char* tag()
	{
		return "Xavier";
	}
};

/* usually for sigmoid & tanh */
struct normalized_xavier : public initializer<normalized_xavier>
{
	template<size_t n>
	inline void operator()(float* outWeights, const layout_t<n>& layout)
	{
		std::random_device randomDevice;
		std::default_random_engine randomEngine(randomDevice());

		uint32_t weightsPtr = 0;

		for (auto& [inputSize, outputSize] : layout)
		{
			const float upper = std::sqrt(6.0f) / std::sqrt((float)(inputSize + outputSize));
			const float lower = -1.0f * upper;
			std::uniform_real_distribution<float> uniform_distribution(lower, upper);

			for (uint32_t i = 0; i < (inputSize * outputSize); ++i)
			{
				outWeights[weightsPtr++] = uniform_distribution(randomEngine);
			}
		}
	}

	constexpr const char* tag()
	{
		return "Normalized Xavier";
	}
};

/*  usually for ReLU & leaky ReLU */
struct he : public initializer<he>
{
	template<size_t n>
	inline void operator()(float* outWeights, const layout_t<n>& layout)
	{
		std::random_device randomDevice;
		std::default_random_engine randomEngine(randomDevice());

		uint32_t weightsPtr = 0;

		for (auto& [inputSize, outputSize] : layout)
		{
			const float standardDeviation = std::sqrt(2.0f / (float)inputSize);
			std::normal_distribution<float> gaussian_distribution(0.0f, standardDeviation);

			for (uint32_t i = 0; i < (inputSize * outputSize); ++i)
			{
				outWeights[weightsPtr++] = gaussian_distribution(randomEngine);
			}
		}
	}

	constexpr const char* tag()
	{
		return "He";
	}
};
