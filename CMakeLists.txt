cmake_minimum_required(VERSION 3.10)
project(Final_CV)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})  # Importa tutti i path corretti

set(HEADER_FILES 
    include/json.hpp
)
set(SOURCE_FILES 
    src/labelImage.cpp 
    src/main.cpp
)
add_library(mylib ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(mylib 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(mylib PRIVATE ${OpenCV_LIBS})
add_executable(labelImage src/labelImage.cpp)
target_include_directories(labelImage PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(labelImage PRIVATE mylib)