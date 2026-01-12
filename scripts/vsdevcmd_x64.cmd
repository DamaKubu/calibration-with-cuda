@echo off
setlocal enabledelayedexpansion

rem Locates Visual Studio via vswhere and initializes an x64 Native Tools environment.
rem Usage:
rem   vsdevcmd_x64.cmd               (just sets up env; returns)
rem   vsdevcmd_x64.cmd cl.exe ...    (sets up env; runs command)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo vswhere not found at "%VSWHERE%".
  echo Install Visual Studio Build Tools or Visual Studio 2022.
  exit /b 2
)

for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
  set "VSINSTALL=%%I"
)

if "%VSINSTALL%"=="" (
  echo Could not locate a Visual Studio installation with C++ tools.
  exit /b 3
)

set "VSDEVCMD=%VSINSTALL%\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEVCMD%" (
  echo VsDevCmd.bat not found at "%VSDEVCMD%".
  exit /b 4
)

call "%VSDEVCMD%" -no_logo -arch=x64 -host_arch=x64

if "%~1"=="" (
  rem No command provided; just return with environment set for the current cmd.exe.
  exit /b 0
)

set "_CMD=%~1"
shift
call "%_CMD%" %*
exit /b %errorlevel%
