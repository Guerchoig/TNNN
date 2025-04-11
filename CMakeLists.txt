cmake_minimum_required(VERSION 3.10)

set(PATCH_VERSION "0" CACHE INTERNAL "Patch version")
set(PROJECT_VESRION 0.0.${PATCH_VERSION})

project(TNN VERSION ${PROJECT_VESRION})

# find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(SFML 2.5 COMPONENTS audio graphics window system REQUIRED)

add_subdirectory(atomic_queue)
add_executable(test src/test.cpp atomic_queue)
add_library(TNN src/brain.cpp src/input_output.cpp src/mnist_set.cpp atomic_queue)


set_target_properties(TNN test PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)



target_include_directories(TNN
    PRIVATE "${CMAKE_BINARY_DIR}" 
            "${CMAKE_CURRENT_SOURCE_DIR}/include" 
            "${CMAKE_SOURCE_DIR}/atomic_queue/include/atomic_queue"
)

target_include_directories(test
    PRIVATE "${CMAKE_BINARY_DIR}" 
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/atomic_queue/include/atomic_queue"
)

# target_link_libraries(reservoir_sampling PRIVATE
#     reservoir_sampling
# )
target_link_libraries(test TNN )

target_link_libraries(TNN sfml-graphics sfml-window sfml-system)

if (MSVC)
    target_compile_options(TNN PRIVATE
        /W4
       )
       target_compile_options(test PRIVATE
        /W4
       )
else ()
    target_compile_options(TNN PRIVATE
        -Wall -Wextra -pedantic -Werror
    )
   target_compile_options(test PRIVATE
       -Wall -Wextra -pedantic -Werror
   )

endif()

# Pass MNIST data directory to main.cpp
target_compile_definitions(test PRIVATE MNIST_DATA_LOCATION="${CMAKE_CURRENT_SOURCE_DIR}/MNIST/")

# install(TARGETS TNN RUNTIME DESTINATION bin)

# set(CPACK_GENERATOR DEB)

# set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
# set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
# set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")

# set(CPACK_PACKAGE_CONTACT example@example.com)

# include(CPack)

# enable_testing()

# add_test(test_version test_version)
