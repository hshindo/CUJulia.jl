# Automatically generated using Clang.jl wrap_c, version 0.0.0

#using Compat

# begin enum ANONYMOUS_1
typealias ANONYMOUS_1 UInt32
const NVRTC_SUCCESS = (UInt32)(0)
const NVRTC_ERROR_OUT_OF_MEMORY = (UInt32)(1)
const NVRTC_ERROR_PROGRAM_CREATION_FAILURE = (UInt32)(2)
const NVRTC_ERROR_INVALID_INPUT = (UInt32)(3)
const NVRTC_ERROR_INVALID_PROGRAM = (UInt32)(4)
const NVRTC_ERROR_INVALID_OPTION = (UInt32)(5)
const NVRTC_ERROR_COMPILATION = (UInt32)(6)
const NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = (UInt32)(7)
const NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = (UInt32)(8)
const NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = (UInt32)(9)
const NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = (UInt32)(10)
const NVRTC_ERROR_INTERNAL_ERROR = (UInt32)(11)
# end enum ANONYMOUS_1

# begin enum nvrtcResult
typealias nvrtcResult UInt32
const NVRTC_SUCCESS = (UInt32)(0)
const NVRTC_ERROR_OUT_OF_MEMORY = (UInt32)(1)
const NVRTC_ERROR_PROGRAM_CREATION_FAILURE = (UInt32)(2)
const NVRTC_ERROR_INVALID_INPUT = (UInt32)(3)
const NVRTC_ERROR_INVALID_PROGRAM = (UInt32)(4)
const NVRTC_ERROR_INVALID_OPTION = (UInt32)(5)
const NVRTC_ERROR_COMPILATION = (UInt32)(6)
const NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = (UInt32)(7)
const NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = (UInt32)(8)
const NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = (UInt32)(9)
const NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = (UInt32)(10)
const NVRTC_ERROR_INTERNAL_ERROR = (UInt32)(11)
# end enum nvrtcResult

typealias _nvrtcProgram Void
typealias nvrtcProgram Ptr{_nvrtcProgram}
