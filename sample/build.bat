@echo off
if not exist build mkdir build
cd build

if not exist windows mkdir windows
cd windows

if not exist build_files mkdir build_files
cd build_files

if not exist release mkdir release
cd release

call cmake -DBUILD_DEBUG=OFF -DBUILD_RELEASE=ON ../../../../
call cmake --build . --config Release
PAUSE
