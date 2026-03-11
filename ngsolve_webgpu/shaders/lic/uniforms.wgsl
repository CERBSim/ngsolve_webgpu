// Uniforms are global variables that are constant for all invocations of a shader.
// They are used to store configuration data (no mesh etc.)
// Each uniform must have a unique binding group and number
// They are used to pass data from the CPU to the GPU (variable names are only relevant within the shader code)

// mesh uniforms
// struct LineIntegralConvolutionUniforms {
//   width: u32,         // canvas width
//   height: u32,        // canvas height
//   kernel_length: u32,
//   oriented: u32,      // 0: not oriented, 1: oriented
//   thickness: u32,     // thickness of the lines (only used for oriented)

//   padding0: u32,
//   padding1: u32,
//   padding2: u32,
// };
