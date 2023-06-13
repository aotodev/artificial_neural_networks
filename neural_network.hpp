#pragma once

#include "core.hpp"
#include "simd.hpp"

#include "initializer_functions.hpp"
#include "activation_functions.hpp"
#include "loss_functions.hpp"
#include "optimizers.hpp"

#include <cmath>
#include <memory>
#include <ranges>
#include <algorithm>

#include <thread>
#include <atomic>

#include <limits>

struct ann_data
{
	float* training_data = nullptr;
	float* training_labels = nullptr;
	uint32_t training_count = 0;

	float* validation_data = nullptr;
	float* validation_labels = nullptr;
	uint32_t validation_count = 0;

	/* offset between data points */
	uint32_t data_offset = 1, labels_offset = 1;
};

/* u is the layout of the model. each integer indicates the size of each layer:
 * input, hidden_layer_0, ... , hidden_layer_n and output.
 * a valid model needs to have at least 3 layers (input -> hidden -> output)
 */

template<uint32_t... u>
class neural_network
{
public:
	static_assert(sizeof...(u) > 2);
	
	static constexpr bool s_single_value_output = ((std::forward<uint32_t>(u), ...) == 1);

	constexpr neural_network()
	{
		constexpr uint32_t pack[sizeof...(u)] = { u... };

		m_input_count = pack[0];
		m_output_count = pack[sizeof...(u) - 1];

		m_biases_count = 0;
		m_weights_count = 0;

		for(auto i = 0; i < sizeof...(u) - 1; i++)
		{
			m_layout[i] = std::make_pair(pack[i], pack[i + 1]);
			m_biases_count += pack[i + 1];
			m_weights_count += pack[i] * pack[i + 1];
		}

		m_neuron_count = m_biases_count;

		m_biases.resize(m_biases_count);
		m_weights.resize(m_weights_count);
	}

	template<typename T>
	void initialize(initializer<T>&& initializer, float biasInitValue = 0.1f);

	template<typename Activation, typename Loss, typename Optimizer>
	void fit(ann_data data, activation<Activation>&& activation, loss<Loss> loss, optimizer<Optimizer>&& optimizer, bool averageGradient, uint32_t epochs, uint32_t minibatchSize);

private:
	struct batch_data
	{
		float* data = nullptr;
		float* labels = nullptr;
		uint32_t count = 0;
	};

	template<typename Activation, typename Loss>
	void forward_pass(batch_data data, activation<Activation>&& activation, loss<Loss> loss, vector_t& inWeightsGradient, vector_t& inBiasesGradient);

	/* returns the  average cost */
	template<typename Activation, typename Loss>
	float test(batch_data testData, activation<Activation>&& activation, loss<Loss> loss);

private:
	layout_t<sizeof...(u) - 1> m_layout;

	alignas(CACHELINE_SIZE) vector_t m_weights;
	alignas(CACHELINE_SIZE) vector_t m_biases;

	uint32_t m_input_count, m_output_count, m_neuron_count, m_weights_count, m_biases_count;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t... u>
template<typename T>
void neural_network<u...>::initialize(initializer<T>&& initializer, float biasInitValue)
{
	/* init biases */
	simd::set_range_value(m_biases.data(), m_biases.size(), biasInitValue);

	/* init weights */
	initializer(m_weights.data(), m_layout);

	if constexpr (VERBOSITY)
	{
		LOG_INFO("\n----------------------------------------------------");
		LOG_INFO("neuron count\t\t== %u", m_neuron_count);
		LOG_INFO("input/output size\t== %u/%u", m_input_count, m_output_count);

		LOG_INFO("----------------------------------------------------");

		uint32_t i = 0;
		for (auto [inputSize, outputSize] : m_layout)
			LOG_INFO("layer[%u] == [%u, %u]", i++, inputSize, outputSize);

		LOG_INFO("----------------------------------------------------");
		LOG_INFO("total learnable parameters: %u", m_weights_count + m_biases_count);
		LOG_INFO("of which %u weights and %u biases", m_weights_count, m_biases_count);
		LOG_INFO("----------------------------------------------------\n");
	}
}

template<uint32_t... u>
template<typename Activation, typename Loss>
void neural_network<u...>::forward_pass(batch_data data, activation<Activation>&& activation, loss<Loss> loss, vector_t& inWeightsGradient, vector_t& inBiasesGradient)
{
	alignas(CACHELINE_SIZE) vector_t neuronOutputs(m_neuron_count, 0.0f);

	size_t biasesOffset = 0, weightsOffset = 0, outputOffset = 0;

	const float* input = data.data;
	for (auto& [rowCount, columnCount] : m_layout)
	{
		simd::vec_mat_mul(input, &m_weights[weightsOffset], &neuronOutputs[outputOffset], rowCount, columnCount);
		activation.add_bias_activation(&neuronOutputs[outputOffset], &m_biases[biasesOffset], columnCount);

		input = neuronOutputs.data() + outputOffset;

		outputOffset += columnCount;
		biasesOffset += columnCount;
		weightsOffset += columnCount * rowCount;
	}

	// calculate gradient
	/* iterate backwards 1 bias at a time... */
	ptrdiff_t bIndex = inBiasesGradient.size() - 1;
	ptrdiff_t deltaIndex = neuronOutputs.size() - 1;

	/* ...and 8 synapses at a time */
	ptrdiff_t neuronIndex = m_neuron_count - 8;
	ptrdiff_t wIndex = inWeightsGradient.size() - 8;
	ptrdiff_t dataOffset = m_input_count - 8;

	vector_t cumulativeDeltas(m_neuron_count, 0.0f);

	if constexpr (s_single_value_output)
		cumulativeDeltas[deltaIndex] = loss.derivative(neuronOutputs[deltaIndex], *data.labels);
	else
	 	cumulativeDeltas[deltaIndex] = loss.derivative(&neuronOutputs[deltaIndex], data.labels, m_output_count);

	for (auto& [rowCount, columnCount] : m_layout | std::views::reverse)
	{
		neuronIndex -= columnCount;

		for (size_t j = 0; j < columnCount; j++)
		{
			float layerCost = cumulativeDeltas[deltaIndex - j] * activation.derivative(neuronOutputs[deltaIndex - j]);

			/* dell with respect to the bias = 1 * layerCost */
			inBiasesGradient[bIndex--] += layerCost; 

			__m256 mlayerCost(_mm256_set1_ps(layerCost));

			for (size_t i = 0; i < rowCount; i += 8)
			{
				/* only called for hidden (middle) layers */
				if (neuronIndex > 0)
				{
					_mm256_storeu_ps(&cumulativeDeltas[neuronIndex - i], _mm256_fmadd_ps(mlayerCost, _mm256_loadu_ps(&m_weights[wIndex]), _mm256_loadu_ps(&cumulativeDeltas[neuronIndex - i])));
					_mm256_storeu_ps(&inWeightsGradient[wIndex], _mm256_fmadd_ps(mlayerCost, _mm256_loadu_ps(&neuronOutputs[neuronIndex - i]), _mm256_loadu_ps(&inWeightsGradient[wIndex])));

					wIndex -= 8;
				}
				/* only called for input layer */
				else 
				{
					_mm256_storeu_ps(&inWeightsGradient[wIndex], _mm256_fmadd_ps(mlayerCost, _mm256_loadu_ps(data.data + (dataOffset - i)), _mm256_loadu_ps(&inWeightsGradient[wIndex])));
					wIndex -= 8;
				}
			}
		}

		deltaIndex -= columnCount;
	}
}

template<uint32_t... u>
template<typename Activation, typename Loss, typename Optimizer>
inline void neural_network<u...>::fit(ann_data data, activation<Activation>&& activation, loss<Loss> loss, optimizer<Optimizer>&& optimizer, bool averageGradient, uint32_t epochs, uint32_t minibatchSize)
{
	assert(data.training_data && data.training_labels && data.training_count);

	if(minibatchSize == 0)
		minibatchSize = data.training_count;

	LOG_INFO("training count == %u, minibatchSize == %u", data.training_count, minibatchSize);

	vector_t weightsGradient(m_weights_count, 0.0f);
	vector_t biasesGradient(m_biases_count, 0.0f);

	std::vector<std::thread> threadPool;

	auto threadCount = std::max(std::thread::hardware_concurrency(), 4u);
	threadPool.reserve(threadCount);
	uint32_t step = minibatchSize / threadCount;

	/* to avoid using a mutex or having data races we will create one gradient per thread,
	 * update them individualy, then average them out before updating the parameters
	 */

	matrix_t perThreadWeightsGradients(threadCount);
	matrix_t perThreadBiasesGradients(threadCount);

	/* note that we are using a vector of vectors as opposed to a single one with offsets.
	 * that is to avoid false sharing by having ranges assigned to different threads too close to one another
	 */

	std::atomic<uint32_t> workFlag = 0;
	uint32_t workMask = 0;
	std::atomic<bool> hasWork = true, threadPaused = false;

	for (uint32_t i = 0; i < threadCount; ++i)
	{
		perThreadWeightsGradients[i].resize(weightsGradient.size(), 0.0f);
		perThreadBiasesGradients[i].resize(biasesGradient.size(), 0.0f);
		workMask |= 1u << i;
	}

	/* each thread will have a separate block of contiguous data points */
	std::vector<std::pair<uint32_t, uint32_t>> sharedDataIndices(threadCount);

	LOG_WARN("thead count == %u, training count per thread == %u", threadCount, minibatchSize / threadCount);

	for (uint32_t i = 0; i < threadCount; ++i)
	{
		threadPool.push_back(
			std::thread([this, activation, loss,
				&data, &workFlag, &hasWork, &threadPaused, id = i,
					&wGradient = perThreadWeightsGradients[i],
						&bGradient = perThreadBiasesGradients[i],
							&dataIndices = sharedDataIndices[i]]
								() mutable
		{
			while (hasWork)
			{
				while (threadPaused)
					{ std::this_thread::yield(); }
					
				auto& [localBegin, localEnd] = dataIndices;

				if (localBegin < localEnd)
				{
					uint32_t index = localBegin++;

					batch_data dataPoint;
					dataPoint.data = data.training_data + (data.data_offset * index);
					dataPoint.labels = data.training_labels + (data.labels_offset * index);
					dataPoint.count = 1;

					forward_pass(dataPoint, activation.static_class(), loss.static_class(), wGradient, bGradient);

					if (localBegin == localEnd)
					{
						workFlag |= 1u << id;
					}
				}
				else
				{
					std::this_thread::yield();
				}
			}
		}));
	}

	const uint32_t iterationsPerEpoch = data.training_count / minibatchSize;
	float iteration = 0.0f;

	LOG_INFO("iterations per epoch == %u", iterationsPerEpoch);

	/* set gradients to zero */
	simd::set_to_zero(weightsGradient.data(), weightsGradient.size());
	simd::set_to_zero(biasesGradient.data(), biasesGradient.size());

	int32_t currentEpoch = 0;
	while(epochs--)
	{
		float iterationPerMinibatch = 0.0f;

		for(uint32_t i = 0; i < data.training_count; i += minibatchSize)
		{
			threadPaused = true;

			for (size_t loc = 0; loc < sharedDataIndices.size(); ++loc)
			{
				sharedDataIndices[loc].second = i + (step * (loc + 1)); // second is end (i.e. one after the last valid)
				sharedDataIndices[loc].first = sharedDataIndices[loc].second - step; // first is begin
			}

			threadPaused = false;

			while (workFlag != workMask) { /* wait for forward pass */ }

			/* weights gradients */
			{
				float leftover = weightsGradient.size() % 8;

				/* sum up... */
				for (auto& gradient : perThreadWeightsGradients)
				{
					for (size_t j = 0; j < weightsGradient.size(); j += 8)
					{
						_mm256_storeu_ps(&weightsGradient[j], _mm256_add_ps(_mm256_loadu_ps(&weightsGradient[j]), _mm256_loadu_ps(&gradient[j])));
						_mm256_storeu_ps(&gradient[j], _mm256_setzero_ps());
					}

					if(leftover)
					{
						size_t offset = weightsGradient.size() - leftover;
            			for(size_t j = offset; j < weightsGradient.size(); j++)
						{
                			weightsGradient[j] += gradient[j];
							gradient[j] = 0.0f;
						}
					}
				}

				/* ... and divide (if applicable) */
				if(averageGradient)
				{
					__m256 minibatch256 = _mm256_set1_ps((float)minibatchSize);
					for (size_t j = 0; j < weightsGradient.size(); j += 8)
						_mm256_storeu_ps(&weightsGradient[j], _mm256_div_ps(_mm256_loadu_ps(&weightsGradient[j]), minibatch256));

					if(leftover)
					{
						size_t offset = weightsGradient.size() - leftover;
						for(size_t j = offset; j < weightsGradient.size(); j++)
							weightsGradient[j] /= (float)minibatchSize;
					}
				}
			}

			/* biases gradients */
			for (size_t j = 0; j < biasesGradient.size(); ++j)
			{
				for (size_t k = 0; k < perThreadBiasesGradients.size(); ++k)
				{
					biasesGradient[j] += perThreadBiasesGradients[k][j];
					perThreadBiasesGradients[k][j] = 0.0f;
				}
				
				if(averageGradient)
					biasesGradient[j] /= (float)minibatchSize;
			}

			optimizer.back_propagate(m_layout, weightsGradient, biasesGradient, m_weights, m_biases, iterationPerMinibatch);

			++iteration;
			++iterationPerMinibatch;

			/* reset gradients */
			simd::set_to_zero(weightsGradient.data(), weightsGradient.size());
			simd::set_to_zero(biasesGradient.data(), biasesGradient.size());

			workFlag = 0U;
		}

		float validationCost = test({data.validation_data, data.validation_labels, data.validation_count}, activation.static_class(), loss.static_class());
		LOG("EPOCH %u | cost(%s): %.4f", currentEpoch, loss.tag(), validationCost);
		currentEpoch++;
	}

	hasWork = false;
	for (auto& thread : threadPool)
		thread.join();
}

template<uint32_t... u>
template<typename Activation, typename Loss>
float neural_network<u...>::test(batch_data validationData, activation<Activation>&& activation, loss<Loss> loss)
{
	if(!validationData.data || !validationData.labels || !validationData.count)
	{
		LOG_ERROR("invalid test data");
		return std::numeric_limits<float>::lowest();
	}

	alignas(CACHELINE_SIZE) vector_t neuronOutputs(m_neuron_count, 0.0f);

	float cost = 0.0f;

	for(uint32_t i = 0; i < validationData.count; i++)
	{
		size_t biasesOffset = 0, weightsOffset = 0, outputOffset = 0;

		const float* input = &validationData.data[i];
		for (auto& [rowCount, columnCount] : m_layout)
		{
			simd::vec_mat_mul(input, &m_weights[weightsOffset], &neuronOutputs[outputOffset], rowCount, columnCount);
			activation.add_bias_activation(&neuronOutputs[outputOffset], &m_biases[biasesOffset], columnCount);

			input = neuronOutputs.data() + outputOffset;

			outputOffset += columnCount;
			biasesOffset += columnCount;
			weightsOffset += columnCount * rowCount;
		}

		if constexpr (s_single_value_output)
		{
			cost += loss(neuronOutputs[outputOffset - m_output_count], validationData.labels[i]);
		}
		else
		{
			cost += loss(&neuronOutputs[outputOffset - m_output_count], validationData.labels + i, m_output_count);
		}
	}

	return cost / float(validationData.count);
}
