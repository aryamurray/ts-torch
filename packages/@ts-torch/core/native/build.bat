@echo off
REM Build script for ts-torch native library (Windows)

setlocal enabledelayedexpansion

REM Default values
set BUILD_TYPE=Release
set LIBTORCH_PATH=
set INSTALL_PREFIX=
set CLEAN=0
set BUILD_EXAMPLES=0
set GENERATOR=

REM Parse command line arguments
:parse_args
if "%~1"=="" goto check_args
if /i "%~1"=="--debug" (
    set BUILD_TYPE=Debug
    shift
    goto parse_args
)
if /i "%~1"=="--release" (
    set BUILD_TYPE=Release
    shift
    goto parse_args
)
if /i "%~1"=="--libtorch" (
    set LIBTORCH_PATH=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--prefix" (
    set INSTALL_PREFIX=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--clean" (
    set CLEAN=1
    shift
    goto parse_args
)
if /i "%~1"=="--examples" (
    set BUILD_EXAMPLES=1
    shift
    goto parse_args
)
if /i "%~1"=="--vs2019" (
    set GENERATOR=-G "Visual Studio 16 2019"
    shift
    goto parse_args
)
if /i "%~1"=="--vs2022" (
    set GENERATOR=-G "Visual Studio 17 2022"
    shift
    goto parse_args
)
if /i "%~1"=="--mingw" (
    set GENERATOR=-G "MinGW Makefiles"
    shift
    goto parse_args
)
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help

echo Unknown option: %~1
echo Use --help for usage information
exit /b 1

:show_help
echo Usage: %~nx0 [options]
echo.
echo Options:
echo   --debug           Build in Debug mode
echo   --release         Build in Release mode (default)
echo   --libtorch PATH   Path to LibTorch installation (required)
echo   --prefix PATH     Installation prefix
echo   --clean           Clean build directory before building
echo   --examples        Build examples after library
echo   --vs2019          Use Visual Studio 2019 generator
echo   --vs2022          Use Visual Studio 2022 generator
echo   --mingw           Use MinGW Makefiles generator
echo   --help, -h        Show this help message
exit /b 0

:check_args
REM Auto-detect LibTorch path if not specified
if "%LIBTORCH_PATH%"=="" (
    REM Try project root /libtorch first
    set LIBTORCH_PATH=%~dp0..\..\..\..\libtorch
    if not exist "!LIBTORCH_PATH!" (
        echo Error: LibTorch path is required
        echo Please specify it with --libtorch C:\path\to\libtorch
        echo Or place libtorch at the project root: ts-tools/libtorch
        exit /b 1
    )
    echo Auto-detected LibTorch at project root
)

if not exist "%LIBTORCH_PATH%" (
    echo Error: LibTorch directory not found: %LIBTORCH_PATH%
    exit /b 1
)

echo === Building ts-torch Native Library ===
echo Build type: %BUILD_TYPE%
echo LibTorch path: %LIBTORCH_PATH%

REM Clean if requested
if %CLEAN%==1 (
    if exist build (
        echo Cleaning build directory...
        rmdir /s /q build
    )
)

REM Create build directory
if not exist build mkdir build
cd build

REM Configure CMake
echo Configuring CMake...
set CMAKE_ARGS=-DCMAKE_PREFIX_PATH="%LIBTORCH_PATH%" -DCMAKE_BUILD_TYPE=%BUILD_TYPE%

if not "%INSTALL_PREFIX%"=="" (
    set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_INSTALL_PREFIX="%INSTALL_PREFIX%"
)

if not "%GENERATOR%"=="" (
    set CMAKE_ARGS=%CMAKE_ARGS% %GENERATOR%
)

cmake %CMAKE_ARGS% ..
if errorlevel 1 (
    echo CMake configuration failed
    cd ..
    exit /b 1
)

REM Build
echo Building...
cmake --build . --config %BUILD_TYPE% -j
if errorlevel 1 (
    echo Build failed
    cd ..
    exit /b 1
)

echo Build successful!
echo.
echo Library location: %CD%\%BUILD_TYPE%\

REM Build examples if requested
if %BUILD_EXAMPLES%==1 (
    echo.
    echo Building examples...

    if not exist "..\examples" (
        echo Examples directory not found
        cd ..
        exit /b 1
    )

    REM Install library to local prefix
    set LOCAL_PREFIX=%CD%\install
    echo Installing library to local prefix for examples...
    cmake --install . --prefix "!LOCAL_PREFIX!" --config %BUILD_TYPE%
    if errorlevel 1 (
        echo Installation failed
        cd ..
        exit /b 1
    )

    REM Build examples
    if not exist examples_build mkdir examples_build
    cd examples_build

    cmake -DCMAKE_PREFIX_PATH="!LOCAL_PREFIX!;%LIBTORCH_PATH%" ..\..\examples
    if errorlevel 1 (
        echo Examples CMake configuration failed
        cd ..\..
        exit /b 1
    )

    cmake --build . --config %BUILD_TYPE% -j
    if errorlevel 1 (
        echo Examples build failed
        cd ..\..
        exit /b 1
    )

    echo Examples built successfully!
    echo Example binaries: %CD%\%BUILD_TYPE%\
    echo.
    echo Run example with:
    echo   cd %CD%\%BUILD_TYPE%
    echo   simple_example.exe

    cd ..
)

cd ..
echo.
echo All done!

endlocal
