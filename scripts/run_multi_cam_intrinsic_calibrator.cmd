@echo off
setlocal

rem Runs the compiled executable; expects OPENCV_INSTALL to point to OpenCV install.
if "%OPENCV_INSTALL%"=="" (
  set "OPENCV_INSTALL=C:\ProgramData\opencv\install"
)

set "PATH=%OPENCV_INSTALL%\x64\vc17\bin;%PATH%"

pushd "%~dp0.."
"main\intrinsic\multi_cam_intrinsic_calibrator.exe" %*
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
