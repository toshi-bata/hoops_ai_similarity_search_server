@echo off
REM scs_points - Windows terminal build. No Visual Studio IDE needed: the MSVC
REM compiler alone is enough (Build Tools for Visual Studio 2022). This mirrors
REM the Linux Makefile - one compiler invocation for a single-file C++17 console
REM program, with no build system in between.
REM
REM   copy _win_env.bat.example _win_env.bat   REM then edit the SDK path
REM   build.bat                                REM writes ..\..\bin\scs_points.exe
REM   build.bat clean
setlocal
pushd "%~dp0"

if /i "%~1"=="clean" (
  if exist "build_win" rmdir /s /q "build_win"
  if exist "..\..\bin\scs_points.exe" del /q "..\..\bin\scs_points.exe"
  echo Cleaned.
  popd & exit /b 0
)

REM --- 1. SDK location -------------------------------------------------------
if exist "_win_env.bat" call "_win_env.bat"
if not defined HVISUALIZE_WEB_INSTALL_DIR (
  echo ERROR: HVISUALIZE_WEB_INSTALL_DIR is not set.
  echo        copy _win_env.bat.example _win_env.bat   and edit the SDK path.
  popd & exit /b 1
)
set "SC_ROOT=%HVISUALIZE_WEB_INSTALL_DIR%\authoring\libsc"

if not exist "%SC_ROOT%\include\sc_store.h" (
  echo ERROR: sc_store.h not found under %SC_ROOT%\include.
  echo        Check HVISUALIZE_WEB_INSTALL_DIR in _win_env.bat.
  popd & exit /b 1
)
if not exist "hoops_license.h" (
  echo ERROR: hoops_license.h is missing ^(download it from the Tech Soft 3D Developer Zone^).
  popd & exit /b 1
)

REM --- 2. MSVC environment ---------------------------------------------------
REM vswhere ships with every Visual Studio 2017+ installer, including the
REM standalone Build Tools, so it is the supported way to find the toolset
REM without hard-coding an install path.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo ERROR: vswhere.exe not found. Install "Build Tools for Visual Studio 2022"
  echo        with the "Desktop development with C++" workload.
  popd & exit /b 1
)
set "VSINSTALL="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL (
  echo ERROR: no MSVC x64 toolset found. Install the "Desktop development with C++"
  echo        workload in the Visual Studio Installer.
  popd & exit /b 1
)
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo ERROR: vcvars64.bat failed.
  popd & exit /b 1
)

REM --- 3. Compile ------------------------------------------------------------
REM Flags carried over from the retired scs_points.vcxproj (Release^|x64):
REM   /std:c++17 /O2 /GL + /LTCG  - the project had WholeProgramOptimization on,
REM                                 so the produced binary matches what the IDE
REM                                 build produced
REM   /MD                         - MultiThreadedDLL, the MSVC runtime the exe
REM                                 expects (see bin\README.md)
REM   /bigobj /Zc:__cplusplus     - were in AdditionalOptions
REM   /DUNICODE /D_UNICODE        - CharacterSet was Unicode
REM The trailing \\ in /Fo is deliberate: a lone backslash before the closing
REM quote would escape it for the compiler's own command-line parser.
if not exist "..\..\bin" mkdir "..\..\bin"
if not exist "build_win\Release" mkdir "build_win\Release"

cl /nologo /std:c++17 /O2 /GL /EHsc /MD /W3 /bigobj /Zc:__cplusplus ^
   /DNDEBUG /DUNICODE /D_UNICODE ^
   /I"%SC_ROOT%\include" ^
   /Fo"build_win\Release\\" ^
   /Fe"..\..\bin\scs_points.exe" ^
   main.cpp ^
   /link /LTCG /LIBPATH:"%SC_ROOT%\lib\win64" ^
   sc.lib sc_utils.lib sc_utils_text.lib hps_core.lib zlib.lib
if errorlevel 1 (
  echo BUILD FAILED.
  popd & exit /b 1
)

echo Built ..\..\bin\scs_points.exe
popd & exit /b 0
