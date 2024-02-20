# cmake_minimum_required(VERSION 3.10)

# project(LinearSVM)

# set(OpenCV_DIR C:/Users/kamir/OneDrive/Desktop/Coding Projects/C++ Stuff/OpenCV Projects/opencv/build)
# find_package(OpenCV REQUIRED)

# include_directories( ${OpenCV_INCLUDE_DIRS} )

# add_executable(${PROJECT_NAME} src/main.cpp)

# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

###################################
cmake_minimum_required(VERSION 3.10)
project(LinearSVM)

# Check if OpenCV is already installed
find_package(OpenCV QUIET)

# If OpenCV is not found, download and build it
if(NOT OpenCV_FOUND)
    message(STATUS "OpenCV not found. Downloading and building from source.")

    # Set the directory where OpenCV will be downloaded and built
    set(OPENCV_INSTALL_DIR "${CMAKE_BINARY_DIR}/opencv_install")

    # Download and build OpenCV
    include(FetchContent)
    FetchContent_Declare(
        opencv
        GIT_REPOSITORY https://github.com/opencv/opencv.git
        GIT_TAG <4.9.0> # specify the version you want to download
    )
    FetchContent_GetProperties(opencv)
    if(NOT opencv_POPULATED)
        FetchContent_Populate(opencv)
        add_subdirectory(${opencv_SOURCE_DIR} ${opencv_BINARY_DIR})
    endif()

    # Add OpenCV to the project
    set(OpenCV_DIR "${OPENCV_INSTALL_DIR}/share/OpenCV")
    find_package(OpenCV REQUIRED)
else()
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
endif()

# Add your executable target and link against OpenCV libraries
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
