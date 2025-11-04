cmake_minimum_required(VERSION 3.21)
project(cuda_detection LANGUAGES CXX)

# If you have multiple toolkits, pass -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
find_package(CUDAToolkit REQUIRED)

# --- Derive include and lib dirs ------------------------------------------------
# include_dir: usually one path
list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDA_DETECT_INCLUDE_DIR)

# lib_dirs: prefer .../lib/x64 if present, else CUDAToolkit_LIBRARY_DIR
set(CUDA_DETECT_LIB_DIR "${CUDAToolkit_LIBRARY_DIR}")
if(EXISTS "${CUDAToolkit_LIBRARY_DIR}/x64")
  set(CUDA_DETECT_LIB_DIR "${CUDAToolkit_LIBRARY_DIR}/x64")
endif()

# bin dir for DLLs
set(_BIN "${CUDAToolkit_BIN_DIR}")

# --- Version tag for NVRTC-builtins (12.6 -> 126) ------------------------------
string(REPLACE "." ";" _ver_list "${CUDAToolkit_VERSION}")
list(GET _ver_list 0 _maj)
list(LENGTH _ver_list _len)
if(_len GREATER 1)
  list(GET _ver_list 1 _min)
else()
  set(_min 0)
endif()
math(EXPR _TAG "${_maj} * 10 + ${_min}")

# --- dll_cuda (driver DLL) -----------------------------------------------------
# Prefer System32 (64-bit)
find_file(CUDA_DETECT_DLL_CUDA
  NAMES nvcuda.dll
  PATHS "C:/Windows/System32" "C:/Windows/SysWOW64"  # fallback shows 32-bit if only that exists
  NO_DEFAULT_PATH)

# --- dll_cudart (runtime DLL) --------------------------------------------------
# Try versioned names first, then any cudart64*.dll in the toolkit bin
set(CUDA_DETECT_DLL_CUDART "")
foreach(_cand
  "cudart64_${_maj}${_min}.dll"
  "cudart64.dll")
  find_file(CUDA_DETECT_DLL_CUDART NAMES "${_cand}" HINTS "${_BIN}" NO_DEFAULT_PATH)
  if(CUDA_DETECT_DLL_CUDART)
    break()
  endif()
endforeach()
if(NOT CUDA_DETECT_DLL_CUDART)
  file(GLOB _cudart_glob LIST_DIRECTORIES OFF "${_BIN}/cudart64*.dll")
  if(_cudart_glob)
    list(GET _cudart_glob 0 CUDA_DETECT_DLL_CUDART)
  endif()
endif()

# --- dll_nvrtc (you asked for the *builtins* DLL specifically) -----------------
# Search for nvrtc-builtins matching our tag, else first available
set(CUDA_DETECT_DLL_NVRTC "")
file(GLOB _builtins_glob LIST_DIRECTORIES OFF "${_BIN}/nvrtc-builtins*.dll")
if(_builtins_glob)
  foreach(_dll IN LISTS _builtins_glob)
    get_filename_component(_name "${_dll}" NAME)
    if(_name MATCHES "nvrtc-builtins.*${_TAG}")
      set(CUDA_DETECT_DLL_NVRTC "${_dll}")
      break()
    endif()
  endforeach()
  if(NOT CUDA_DETECT_DLL_NVRTC)
    list(GET _builtins_glob 0 CUDA_DETECT_DLL_NVRTC)
  endif()
endif()

# --- Sanity hints (optional warnings) -----------------------------------------
if(NOT CUDA_DETECT_DLL_CUDA)
  message(WARNING "nvcuda.dll not found in System32/SysWOW64.")
endif()
if(NOT CUDA_DETECT_DLL_CUDART)
  message(WARNING "cudart64*.dll not found in ${_BIN}.")
endif()
if(NOT CUDA_DETECT_DLL_NVRTC)
  message(WARNING "nvrtc-builtins*.dll not found in ${_BIN}.")
endif()

# --- Produce outputs in your desired schema -----------------------------------
# lib_names are the import-library basenames
set(_LIB_NAMES_PY "{'cuda': 'cuda', 'cudart': 'cudart', 'nvrtc': 'nvrtc'}")

# Escape backslashes for Python/JSON strings
function(_escape_for_py in_var out_var)
  string(REPLACE "\\" "\\\\" _tmp "${${in_var}}")
  set(${out_var} "${_tmp}" PARENT_SCOPE)
endfunction()

_escape_for_py(CUDA_DETECT_DLL_CUDA   _DLL_CUDA_PY)
_escape_for_py(CUDA_DETECT_DLL_CUDART _DLL_CUDART_PY)
_escape_for_py(CUDA_DETECT_DLL_NVRTC  _DLL_NVRTC_PY)
_escape_for_py(CUDA_DETECT_INCLUDE_DIR _INCLUDE_PY)
_escape_for_py(CUDA_DETECT_LIB_DIR     _LIBDIR_PY)

# Python-style dict (exact key names as you requested)
set(_DICT_PY
  "{'dll_cuda': '${_DLL_CUDA_PY}', "
  "'dll_cudart': '${_DLL_CUDART_PY}', "
  "'dll_nvrtc': '${_DLL_NVRTC_PY}', "
  "'include_dir': '${_INCLUDE_PY}', "
  "'lib_dirs': '${_LIBDIR_PY}', "
  "'lib_names': ${_LIB_NAMES_PY}}"
)

message(STATUS "CUDA_DETECTION=${_DICT_PY}")

# Also emit a JSON file (cuda_detect.json) in the build dir for programmatic use
# Note: using the same values; JSON requires quoted keys and escaped paths.
set(_JSON
  "{\n"
  "  \"dll_cuda\": \"${_DLL_CUDA_PY}\",\n"
  "  \"dll_cudart\": \"${_DLL_CUDART_PY}\",\n"
  "  \"dll_nvrtc\": \"${_DLL_NVRTC_PY}\",\n"
  "  \"include_dir\": \"${_INCLUDE_PY}\",\n"
  "  \"lib_dirs\": \"${_LIBDIR_PY}\",\n"
  "  \"lib_names\": {\"cuda\": \"cuda\", \"cudart\": \"cudart\", \"nvrtc\": \"nvrtc\"}\n"
  "}\n")
file(WRITE "${CMAKE_BINARY_DIR}/cuda_detect.json" "${_JSON}")

# Dummy target so configure+build does something
add_custom_target(show_detect ALL
  COMMENT "CUDA detection complete. See message above or cuda_detect.json.")
