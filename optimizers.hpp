#pragma once

#include "core.hpp"
#include "simd.hpp"

#include <cmath>
#include <immintrin.h> //AVX

template<typename T>
struct optimizer
{
	template<size_t n>
	void back_propagate(const layout_t<n>& layout, const vector_t& weightsGradient, const vector_t& biasesGradient, vector_t& outWeights, vector_t& outBiases, float iteration)
	{
		((T*)this)->back_propagate(layout, weightsGradient, biasesGradient, outWeights, outBiases, iteration);
	}

	constexpr const char* tag()
	{
		return ((T*)this)->tag();
	}
};

struct sgd : public optimizer<sgd>
{
	sgd(float learningRate = 0.01f, float momentum = 0.99f)
		: m_learning_rate(learningRate), m_momentum(momentum) {}

	template<size_t n>
	void back_propagate(const layout_t<n>& layout, const vector_t& weightsGradient, const vector_t& biasesGradient, vector_t& outWeights, vector_t& outBiases, float iteration)
	{
		if (m_weights_update_values.empty())
			m_weights_update_values.resize(weightsGradient.size(), 0.0f);

		if (m_biases_update_values.empty())
			m_biases_update_values.resize(biasesGradient.size(), 0.0f);

		__m256 alpha = _mm256_set1_ps(m_learning_rate);
		__m256 momentum = _mm256_set1_ps(m_momentum);

		size_t bIndex = 0;
		size_t wIndex = 0;

		for (auto& [rowCount, columnCount] : layout)
		{
			for (size_t j = 0; j < columnCount; j++, bIndex++)
			{
				for (size_t i = 0; i < rowCount / 8; i++, wIndex += 8)
				{
					_mm256_storeu_ps(&m_weights_update_values[wIndex], _mm256_add_ps(_mm256_mul_ps(alpha, _mm256_loadu_ps(&weightsGradient[wIndex])), _mm256_mul_ps(momentum, _mm256_loadu_ps(&m_weights_update_values[wIndex]))));
					_mm256_storeu_ps(&outWeights[wIndex], _mm256_sub_ps(_mm256_loadu_ps(&outWeights[wIndex]), _mm256_loadu_ps(&m_weights_update_values[wIndex])));
				}

				m_biases_update_values[bIndex] = (biasesGradient[bIndex] * m_learning_rate) + (m_biases_update_values[bIndex] * m_momentum);
				outBiases[bIndex] -= m_biases_update_values[bIndex];
			}
		}
	}

	constexpr const char* tag()
	{
		return "SGD";
	}

private:
	/* to store gradient(t-1) for sgd's momentum */
	vector_t m_weights_update_values, m_biases_update_values;
	float m_learning_rate = 0.01f, m_momentum = 0.0f;
};

struct adam : public optimizer<adam>
{
	adam(float stepsize = 0.001f, float beta_1 = 0.9f, float beta_2 = 0.999f, float epsilon = 1e-08f)
		: m_stepsize(stepsize), m_beta_1(beta_1), m_beta_2(beta_2), m_epsilon(epsilon) {}

	template<size_t n>
	void back_propagate(const layout_t<n>& layout, const vector_t& weightsGradient, const vector_t& biasesGradient, vector_t& outWeights, vector_t& outBiases, float iteration)
	{
		/* gradient decaying average & gradient squared decaying average */
		if (m_weights_moment_vector_1.empty())
		{
			m_weights_moment_vector_1.resize(weightsGradient.size(), 0.0f);
			m_weights_moment_vector_2.resize(weightsGradient.size(), 0.0f);
			m_biases_moment_vector_1.resize(biasesGradient.size(), 0.0f);
			m_biases_moment_vector_2.resize(biasesGradient.size(), 0.0f);
		}

		__m256 it = _mm256_set1_ps(iteration + 1.0f);
		__m256 unit = _mm256_set1_ps(1.0f);
		__m256 step = _mm256_set1_ps(m_stepsize);
		__m256 beta_1 = _mm256_set1_ps(m_beta_1);
		__m256 beta_2 = _mm256_set1_ps(m_beta_2);
		__m256 epsilon = _mm256_set1_ps(m_epsilon);

		size_t wIndex = 0, bIndex = 0;

		for (auto& [rowCount, columnCount] : layout)
		{
			for (size_t j = 0; j < columnCount; ++j)
			{
				for (size_t i = 0; i < rowCount / 8; ++i)
				{
					/* calculate moving averages */
					_mm256_storeu_ps(&m_weights_moment_vector_1[wIndex], _mm256_fmadd_ps(beta_1, _mm256_loadu_ps(&m_weights_moment_vector_1[wIndex]), _mm256_mul_ps(_mm256_sub_ps(unit, beta_1), _mm256_loadu_ps(&weightsGradient[wIndex]))));
					_mm256_storeu_ps(&m_weights_moment_vector_2[wIndex], _mm256_fmadd_ps(beta_2, _mm256_loadu_ps(&m_weights_moment_vector_2[wIndex]), _mm256_mul_ps(_mm256_sub_ps(unit, beta_2), simd::pow2_ps(_mm256_loadu_ps(&weightsGradient[wIndex])))));

					__m256 moment_1 = _mm256_div_ps(_mm256_loadu_ps(&m_weights_moment_vector_1[wIndex]), _mm256_sub_ps(unit, _mm256_pow_ps(beta_1, it)));
					__m256 moment_2 = _mm256_div_ps(_mm256_loadu_ps(&m_weights_moment_vector_2[wIndex]), _mm256_sub_ps(unit, _mm256_pow_ps(beta_2, it)));

					/* update weights */
					_mm256_storeu_ps(&outWeights[wIndex], _mm256_sub_ps(_mm256_loadu_ps(&outWeights[wIndex]), _mm256_div_ps(_mm256_mul_ps(step, moment_1), _mm256_add_ps(_mm256_sqrt_ps(moment_2), epsilon))));

					wIndex += 8;
				}

				m_biases_moment_vector_1[bIndex] = m_beta_1 * m_biases_moment_vector_1[bIndex] + (1 - m_beta_1) * biasesGradient[bIndex];
				m_biases_moment_vector_2[bIndex] = m_beta_2 * m_biases_moment_vector_2[bIndex] + (1 - m_beta_2) * std::pow(biasesGradient[bIndex], 2.0f);

				float biasMoment_1 = m_biases_moment_vector_1[bIndex] / (1.0f - std::pow(m_beta_1, iteration + 1.0f));
				float biasMoment_2 = m_biases_moment_vector_2[bIndex] / (1.0f - std::pow(m_beta_2, iteration + 1.0f));

				outBiases[bIndex] -= (m_stepsize * biasMoment_1) / (std::sqrt(biasMoment_2) + m_epsilon);
				bIndex++;
			}
		}
	}

	constexpr const char* tag()
	{
		return "Adam";
	}

private:
	/* to store the 2 exponentially decaying moving average vectors (t-1) */
	vector_t m_weights_moment_vector_1, m_weights_moment_vector_2;
	vector_t m_biases_moment_vector_1, m_biases_moment_vector_2;

	float m_stepsize, m_beta_1, m_beta_2, m_epsilon;
};
