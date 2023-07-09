#pragma once

#if defined(__WIN32__) || defined(_WIN32) || defined(__CYGWIN32__)
#define APP_WINDOWS
#endif

#if !defined(APP_COMPILER_GNUC) && !defined(APP_COMPILER_CLANG) && !defined(APP_COMPILER_MSVC)

#if defined(__clang__)
#define APP_COMPILER_CLANG

#elif defined(__GNUC__) || defined(__GNUG__)
#define APP_COMPILER_GNUC

#elif defined(_MSC_VER)
#define APP_COMPILER_MSVC
#endif

#endif

#define CAT(Arg1, Arg2) CAT_INTERNAL(Arg1, Arg2)
#define CAT_INTERNAL(Arg1, Arg2) Arg1##Arg2

#define BIT(x) (1UL << x)
#define CACHELINE_SIZE 64ULL

#ifndef VERBOSITY
#define VERBOSITY 2
#endif

#include <cstdio>

#if VERBOSITY > 1
#define LOG_VERBOSE(message, ...) printf(message "\n" __VA_OPT__(,) __VA_ARGS__);
#else
#define LOG_VERBOSE(...)
#endif

#if VERBOSITY > 0
#ifdef APP_WINDOWS
#define LOG_INFO(message, ...) printf(message "\n" __VA_OPT__(,) __VA_ARGS__);
#else
#define LOG_INFO(message, ...) printf("\033[0;32m" message "\033[0m\n" __VA_OPT__(,) __VA_ARGS__);
#endif // windows

#else
#define LOG_INFO(...)
#endif

#define LOG(message, ...) printf(message "\n" __VA_OPT__(,) __VA_ARGS__);

#ifdef APP_WINDOWS
#define LOG_WARN(message, ...) printf(message "\n" __VA_OPT__(,) __VA_ARGS__);
#define LOG_ERROR(message, ...) printf(message "\n" __VA_OPT__(,) __VA_ARGS__);
#else

#define LOG_WARN(message, ...) printf("\033[0;33m" message "\033[0m\n" __VA_OPT__(,) __VA_ARGS__);
#define LOG_ERROR(message, ...) printf("\033[0;97;101m" message "\033[0m\n" __VA_OPT__(,) __VA_ARGS__);
#endif // windows


//////////////////////////////////--TYPES--//////////////////////////////////////////////
#include <vector>

typedef std::vector<float> vector_t;
typedef std::vector<std::vector<float>> matrix_t;

#include <array>
#include <utility>
#include <stdint.h>

/* each first/second pair represents the input/output count of each layer of synapses */
template<uint32_t n>
using layout_t = std::array<std::pair<uint32_t, uint32_t>, n>;
