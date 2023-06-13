cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(ann_time_series VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------------------
# Build type & definitions
# ---------------------------------------------------------------------------------------
option(BUILD_DEBUG "debug" ON)
option(BUILD_RELEASE "release" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(WIN32)
	set(BASE_BIN_PATH "${CMAKE_SOURCE_DIR}/build/windows")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	set(BASE_BIN_PATH "${CMAKE_SOURCE_DIR}/build/linux")
endif()

macro(set_outputs_macro buildType)
	set(LIBRARY_OUTPUT_PATH "${BASE_BIN_PATH}/${buildType}")
	set(EXECUTABLE_OUTPUT_PATH "${BASE_BIN_PATH}/${buildType}")

	# MSVC appends Debug/Release folders at the end of the specified path
	if(MSVC)
		foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
			string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
			set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${BASE_BIN_PATH}/${buildType}" )
			set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${BASE_BIN_PATH}/${buildType}" )
			set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${BASE_BIN_PATH}/${buildType}" )
		endforeach()
	endif()

	message(STATUS "Building ${CMAKE_PROJECT_NAME} in ${buildType} Mode")
endmacro()

if(BUILD_DEBUG)
	add_compile_definitions(APP_DEBUG)
	set(CMAKE_BUILD_TYPE Debug)
	set_outputs_macro(debug)

elseif(BUILD_RELEASE)
	add_compile_definitions(APP_RELEASE)
	add_compile_definitions(NDEBUG)
	set(CMAKE_BUILD_TYPE Release)
	set_outputs_macro(release)

endif()

file(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/data)
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${EXECUTABLE_OUTPUT_PATH})

# ---------------------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------------------
if(MSVC)
add_compile_definitions(APP_COMPILER_MSVC)
	if(BUILD_DEBUG)
		set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} /Od /Z7 /FA /MTd /arch:AVX")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od /Z7 /FA /MTd /arch:AVX")

	else()
		add_compile_options($<$<CONFIG:Release>:/MT>)
		set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /FA /MT /arch:AVX")

	endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	add_compile_definitions(APP_COMPILER_CLANG)

	if(BUILD_DEBUG)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmove -Wshadow -msimd128 -mavx -march=native -pthread")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread -v")

	else() #BUILD_RELEASE
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -msimd128 -mavx -march=native -pthread -s") #--strip-all
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")

	endif()


elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	add_compile_definitions(APP_COMPILER_GNUC)

	if(BUILD_DEBUG)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow -msse -mavx -march=native -pthread")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")

	else() #BUILD_RELEASE
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -msse -mavx -march=native -pthread -s")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")

	endif()

endif() #COMPILER

# ---------------------------------------------------------------------------------------
# add executable
# ---------------------------------------------------------------------------------------
set(ProjectSourceFiles
	main.cpp
)

add_executable(${CMAKE_PROJECT_NAME} ${ProjectSourceFiles})

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR})
