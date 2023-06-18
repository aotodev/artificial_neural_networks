#pragma once

#include "core.hpp"

#include <immintrin.h>
#include <vector>

namespace simd {

    inline __m256 pow2_ps(__m256 m)
    {
        return _mm256_mul_ps(m, m);
    }

    inline float getf(__m256& vec, size_t index) { return *(((float*)&vec) + index); }

    inline float accumulate(float* inVec, size_t count)
    {
        __m256 temp = _mm256_setzero_ps();
        size_t leftover = count % 8;

        for (size_t i = 0; i < (count - leftover); i+=8)
            temp = _mm256_add_ps(_mm256_loadu_ps(&inVec[i]), temp);

        __m128 temp128 = _mm_add_ps(_mm256_castps256_ps128(temp), _mm256_extractf128_ps(temp, 1));
        __m128 high = _mm_movehl_ps(temp128, temp128);

        temp128 = _mm_add_ps(temp128, high);
        high =  _mm_shuffle_ps(temp128, temp128, 0x1);

        float mValue = _mm_cvtss_f32(_mm_add_ps(temp128, high));

        if (leftover)
        {
            size_t offset = count - leftover;
            for(size_t i = offset; i < count; i++)
                mValue += inVec[i];
        }

        return mValue;
    }

    inline float accumulate(__m256 inVec)
    {
        __m128 temp128 = _mm_add_ps(_mm256_castps256_ps128(inVec), _mm256_extractf128_ps(inVec, 1));
        __m128 high = _mm_movehl_ps(temp128, temp128);

        temp128 = _mm_add_ps(temp128, high);
        high =  _mm_shuffle_ps(temp128, temp128, 0x1);

        return _mm_cvtss_f32(_mm_add_ps(temp128, high));
    }


/* std::vector<__m256> will be alligned correctly, this warning does not help */
#ifdef APP_COMPILER_GNUC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

    inline void vec_mat_mul(const float* inVec, const float* inMatrix, float* outVec, size_t m, size_t n)
    {
        /* n __m256s(8 floats per element) */
        alignas(CACHELINE_SIZE) std::vector<__m256> tempResults(n, _mm256_setzero_ps());

        size_t nLeftover = n % 8;
        size_t mLeftover = m % 8;

        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < (m - mLeftover); j+=8)
            {
                tempResults[i] = _mm256_fmadd_ps(_mm256_loadu_ps(&inVec[j]), _mm256_loadu_ps(&inMatrix[i * m + j]), tempResults[i]);
            }

            if(mLeftover)
            {
                /* fill an array of size 8 with the remaing values followed by 0s */
                size_t offset = m - mLeftover;

                float vecTemp[8] = { 0.0f };
                float matTemp[8] = { 0.0f };


                for (size_t k = offset; k < m; k++)
                {
                    vecTemp[k - offset] = inVec[k];
                    matTemp[k - offset] = inMatrix[i * m + k];
                }

                tempResults[i] = _mm256_fmadd_ps(_mm256_loadu_ps(vecTemp), _mm256_loadu_ps(matTemp), tempResults[i]);
            }
        }

        for (size_t i = 0, count = 0; i < (n - nLeftover); i += 8, count++)
        {
            //reduce 8 to 4
            __m256 vec_0(_mm256_add_ps(
                _mm256_set_m128(_mm256_castps256_ps128(tempResults[i + 1]), _mm256_castps256_ps128(tempResults[i + 0])),
                _mm256_set_m128(_mm256_extractf128_ps(tempResults[i + 1], 1), _mm256_extractf128_ps(tempResults[i + 0], 1))));

            __m256 vec_1(_mm256_add_ps(
                _mm256_set_m128(_mm256_castps256_ps128(tempResults[i + 3]), _mm256_castps256_ps128(tempResults[i + 2])),
                _mm256_set_m128(_mm256_extractf128_ps(tempResults[i + 3], 1), _mm256_extractf128_ps(tempResults[i + 2], 1))));

            __m256 vec_2(_mm256_add_ps(
                _mm256_set_m128(_mm256_castps256_ps128(tempResults[i + 5]), _mm256_castps256_ps128(tempResults[i + 4])),
                _mm256_set_m128(_mm256_extractf128_ps(tempResults[i + 5], 1), _mm256_extractf128_ps(tempResults[i + 4], 1))));

            __m256 vec_3(_mm256_add_ps(
                _mm256_set_m128(_mm256_castps256_ps128(tempResults[i + 7]), _mm256_castps256_ps128(tempResults[i + 6])),
                _mm256_set_m128(_mm256_extractf128_ps(tempResults[i + 7], 1), _mm256_extractf128_ps(tempResults[i + 6], 1))));

            //reduce 4 to 2
            __m256 vec_4(_mm256_add_ps(
                _mm256_setr_ps(getf(vec_0, 0), getf(vec_0, 1), getf(vec_0, 4), getf(vec_0, 5), getf(vec_1, 0), getf(vec_1, 1), getf(vec_1, 4), getf(vec_1, 5)),
                _mm256_setr_ps(getf(vec_0, 2), getf(vec_0, 3), getf(vec_0, 6), getf(vec_0, 7), getf(vec_1, 2), getf(vec_1, 3), getf(vec_1, 6), getf(vec_1, 7))));


            __m256 vec_5(_mm256_add_ps(
                _mm256_setr_ps(getf(vec_2, 0), getf(vec_2, 1), getf(vec_2, 4), getf(vec_2, 5), getf(vec_3, 0), getf(vec_3, 1), getf(vec_3, 4), getf(vec_3, 5)),
                _mm256_setr_ps(getf(vec_2, 2), getf(vec_2, 3), getf(vec_2, 6), getf(vec_2, 7), getf(vec_3, 2), getf(vec_3, 3), getf(vec_3, 6), getf(vec_3, 7))));

            //final values
             auto final_vector = _mm256_add_ps(
                _mm256_setr_ps(getf(vec_4, 0), getf(vec_4, 2), getf(vec_4, 4), getf(vec_4, 6), getf(vec_5, 0), getf(vec_5, 2), getf(vec_5, 4), getf(vec_5, 6)),
                _mm256_setr_ps(getf(vec_4, 1), getf(vec_4, 3), getf(vec_4, 5), getf(vec_4, 7), getf(vec_5, 1), getf(vec_5, 3), getf(vec_5, 5), getf(vec_5, 7)));

             _mm256_storeu_ps(&outVec[i], final_vector);
        }

        if (nLeftover)
        {
            size_t offset = n - nLeftover;
            for (size_t i = offset; i < n; i++)
            {
                outVec[i] = accumulate(tempResults[i]);
            }
        }
    }

#ifdef APP_COMPILER_GNUC
#pragma GCC diagnostic pop
#endif


    inline void set_to_zero(float* inVec, size_t count)
    {
        size_t leftover = count % 8;

        for (size_t i = 0; i < (count - leftover); i+=8)
            _mm256_storeu_ps(&inVec[i], _mm256_setzero_ps());

        if (leftover)
        {
            size_t offset = count - leftover;
            for (size_t i = offset; i < count; i++)
                inVec[i] = 0.0f;
        }
    }

    inline void set_range_value(float* inVec, size_t count, float value)
    {
        size_t leftover = count % 8;

        __m256 mValue = _mm256_set1_ps(value);

        for (size_t i = 0; i < (count - leftover); i+=8)
            _mm256_storeu_ps(&inVec[i], mValue);

        if (leftover)
        {
            size_t offset = count - leftover;
            for (size_t i = offset; i < count; i++)
                inVec[i] = value;
        }       
    }

}

#include "vendor/avx_mathfun.h"

#if defined(APP_COMPILER_GNUC ) || defined(APP_COMPILER_CLANG)

/* _mm256_pow_ps is an intel extension only available on intel and MSVC compilers */
#define _mm256_pow_ps(a, b) pow_ps(a, b)
#endif
