@echo off
echo Cleaning OpenCV build caches...

:: Delete local build directory
if exist build (
    echo Removing build folder...
    rmdir /s /q build
) else (
    echo Build folder not found
)

:: Delete CMake global cache
if exist "%LOCALAPPDATA%\CMake" (
    echo Removing CMake cache...
    rmdir /s /q "%LOCALAPPDATA%\CMake"
) else (
    echo CMake cache folder not found
)

if exist "%LOCALAPPDATA%\cmake" (
    rmdir /s /q "%LOCALAPPDATA%\cmake"
) else (
    echo CMake user cache not found
)

:: Delete old OpenCV install folders
if exist "C:\opencv" (
    rmdir /s /q "C:\opencv"
) else (
    echo C:\opencv not found
)

if exist "C:\Program Files\opencv" (
    rmdir /s /q "C:\Program Files\opencv"
) else (
    echo C:\Program Files\opencv not found
)

echo Cleaning done!
pause
