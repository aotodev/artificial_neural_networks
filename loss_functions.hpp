#pragma once

#include "core.hpp"

#include <cmath>
#include <assert.h>

template<typename T>
struct loss
{
	float operator()(const float* prediction, const float* label, uint32_t count)
	{
		return ((T*)this)->operator()(prediction, label, count);
	}

	float operator()(const float prediction, const float label)
	{
		return ((T*)this)->operator()(prediction, label);
	}

	float derivative(const float* prediction, const float* label, uint32_t count)
	{
		return ((T*)this)->derivative(prediction, label, count);
	}

	float derivative(const float prediction, const float label)
	{
		return ((T*)this)->derivative(prediction, label);
	}

	constexpr const char* tag()
	{
		return ((T*)this)->tag();
	}

	static loss<T> static_class()
	{
		return loss<T>();
	}
};

struct mse : public loss<mse>
{
	float operator()(const float* prediction, const float* label, uint32_t count)
	{
		float errorSUM = 0.0f;

		for (uint32_t i = 0; i < count; ++i)
		{
			errorSUM += (std::pow(label[i] - prediction[i], 2));
		}

		return (errorSUM / (float)count);
	}

	float operator()(const float prediction, const float label)
	{
		return std::pow(label - prediction, 2);
	}

	float derivative(const float* prediction, const float* label, uint32_t count)
	{
		float errorSUM = 0.0f;

		for (uint32_t i = 0; i < count; ++i)
		{
			errorSUM += (label[i] - prediction[i]);
		}

		return -1.0f * (errorSUM / (float)count);
	}

	float derivative(const float prediction, const float label)
	{
		return -1.0f * (label - prediction);
	}

	constexpr const char* tag()
	{
		return "MSE";
	}
};

struct rmse : public loss<rmse>
{
	float operator()(const float* prediction, const float* label, uint32_t count)
	{
		float errorSUM = 0.0f;

		for (uint32_t i = 0; i < count; ++i)
		{
			errorSUM += (std::pow(label[i] - prediction[i], 2));
		}

		return std::sqrt(errorSUM / (float)count);
	}

	float operator()(const float prediction, const float label)
	{
		return std::abs(label - prediction);
	}

	float derivative(const float* prediction, const float* label, uint32_t count)
	{
		float errorSUM = 0.0f;
		float sse = 0.0f;

		for (uint32_t i = 0; i < count; ++i)
		{
			errorSUM += (label[i] - prediction[i]);
			sse += (std::pow(label[i] - prediction[i], 2));
		}

		return errorSUM / std::sqrt(float(count) * sse);
	}

	float derivative(const float prediction, const float label)
	{
		return (prediction - label) / std::pow(label - prediction, 2);
	}

	constexpr const char* tag()
	{
		return "RMSE";
	}
};
