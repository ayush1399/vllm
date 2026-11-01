# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

find_package(Git REQUIRED)

foreach(CURRENT_PATCH IN ITEMS "${PATCH_FILE}" "${SECOND_PATCH_FILE}")
  if(NOT CURRENT_PATCH)
    continue()
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${CURRENT_PATCH}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE ALREADY_APPLIED
    OUTPUT_QUIET ERROR_QUIET)
  if(ALREADY_APPLIED EQUAL 0)
    continue()
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply "${CURRENT_PATCH}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    COMMAND_ERROR_IS_FATAL ANY)
endforeach()
