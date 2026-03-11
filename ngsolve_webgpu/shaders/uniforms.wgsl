// Uniforms are global variables that are constant for all invocations of a shader.
// They are used to store configuration data (no mesh etc.)
// Each uniform must have a unique binding group and number
// They are used to pass data from the CPU to the GPU (variable names are only relevant within the shader code)

// legacy uniforms
@group(0) @binding(12) var<storage> vertices : array<f32>;
@group(0) @binding(13) var<storage> trigs : array<u32>;

// mesh uniforms
@group(0) @binding(20) var<uniform> u_mesh : MeshUniforms;
@group(0) @binding(23) var<storage> u_trigs : array<Trig>;
@group(0) @binding(25) var<storage> u_tets : array<Tet>;

struct MeshUniforms {
  subdivision: u32,
  shrink: f32,

  padding0: f32,
  padding1: f32,
};

struct LineIntegralConvolutionUniforms {
  width: u32,         // canvas width
  height: u32,        // canvas height
  kernel_length: u32,
  oriented: u32,      // 0: not oriented, 1: oriented
  thickness: u32,     // thickness of the lines (only used for oriented)

  padding0: u32,
  padding1: u32,
  padding2: u32,
};
// Mesh element structures

struct Edge { p: array<u32, 2> }; // Inner edge, for wireframe
struct Seg { p: array<u32, 2>, nr: u32, index: u32 };
struct Trig { p: array<u32, 3>, nr: u32, index: u32 };
struct Quad { p: array<u32, 4>, nr: u32, index: u32 };
struct Tet { p: array<u32, 4>, nr: u32, index: u32 };
struct Pyramid { p: array<u32, 5>, nr: u32, index: u32 };
struct Prism { p: array<u32, 6>, nr: u32, index: u32 };
struct Hex { p: array<u32, 8>, nr: u32, index: u32 };


// legacy mesh structs
struct EdgeP1 { p: array<f32, 6> };
struct TrigP1 { p: array<f32, 9>, index: i32 }; // 3 vertices with 3 coordinates each, don't use vec3 due to 16 byte alignment
struct TrigP2 { p: array<f32, 18>, index: i32 };

struct ClippingTrig {
  nr: u32,            // volume element number
  lam: array<f32, 9>, // 3 vertices with 3 barycentric coordinates each
};
