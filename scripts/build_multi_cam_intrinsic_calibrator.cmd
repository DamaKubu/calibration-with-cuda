@echo off
setlocal

rem Builds the multi-camera intrinsic calibrator (C++/OpenCV) using cl.exe.
rem Expected to be run inside an x64 MSVC developer environment (VsDevCmd).

if "%OPENCV_INSTALL%"=="" (
  set "OPENCV_INSTALL=C:\ProgramData\opencv\install"
)

set "OPENCV_INC=%OPENCV_INSTALL%\include"
set "OPENCV_LIB=%OPENCV_INSTALL%\x64\vc17\lib"

if not exist "%OPENCV_INC%\opencv2" (
  echo OpenCV headers not found at "%OPENCV_INC%".
  echo Set OPENCV_INSTALL to your OpenCV install directory.
  exit /b 3
)

if not exist "%OPENCV_LIB%" (
  echo OpenCV libs not found at "%OPENCV_LIB%".
  echo Set OPENCV_INSTALL to your OpenCV install directory.
  exit /b 3
)

pushd "%~dp0.."

cl.exe /Zi /EHsc /nologo /std:c++17 /I"%OPENCV_INC%" ^
  /Fe"main\intrinsic\multi_cam_intrinsic_calibrator.exe" ^
  "main\intrinsic\multi_cam_intrinsic_calibrator.cpp" ^
  /link /LIBPATH:"%OPENCV_LIB%" opencv_calib3d.lib opencv_core.lib opencv_highgui.lib opencv_imgproc.lib opencv_imgcodecs.lib opencv_videoio.lib

set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
