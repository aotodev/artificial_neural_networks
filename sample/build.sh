#!/bin/bash
#...

export CXX=/bin/g++
export C=/bin/gcc

mkdir -p build/linux/build_files/release
cd build/linux/build_files/release

cmake -DBUILD_DEBUG=OFF -DBUILD_RELEASE=ON ../../../../
cmake --build . --config Release

read -rsp $'Press escape to continue...\n' -d $'\e'
