#import camera
#import light
#import clipping

@group(0) @binding(90) var<storage> u_vertices : array<f32>;
@group(0) @binding(91) var<storage> u_normals : array<f32>;
@group(0) @binding(92) var<storage> u_indices : array<u32>;
@group(0) @binding(93) var<storage> u_colors: array<f32>;

struct GeoFragmentInput {
  @builtin(position) position: vec4<f32>,
    @location(0) p: vec3<f32>,
    @location(1) n: vec3<f32>,
    @location(2) @interpolate(flat) id: u32,
    @location(3) @interpolate(flat) index: u32,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertId: u32,
               @builtin(instance_index) trigId: u32) -> GeoFragmentInput {
  let point = vec3<f32>(u_vertices[trigId * 9 + vertId * 3],
                        u_vertices[trigId * 9 + vertId * 3 + 1],
                        u_vertices[trigId * 9 + vertId * 3 + 2]);
  let normal = vec3<f32>(u_normals[trigId * 9 + vertId * 3],
                         u_normals[trigId * 9 + vertId * 3 + 1],
                         u_normals[trigId * 9 + vertId * 3 + 2]);
  let position = cameraMapPoint(point);
  return GeoFragmentInput(position,
                          point,
                          normal,
                          trigId, u_indices[trigId]);
}

@fragment
fn fragment_main(input: GeoFragmentInput) -> @location(0) vec4<f32> {
  let color = vec4<f32>(u_colors[input.index * 4],
                        u_colors[input.index * 4 + 1],
                        u_colors[input.index * 4 + 2],
                        u_colors[input.index * 4 + 3]);
  if (color.a == 0.) {
    discard;
  }
  return lightCalcColor(input.p, input.n, color);
}

@fragment
fn fragmentQueryIndex(input: GeoFragmentInput) -> @location(0) vec2<u32>
{
  return vec2<u32>(input.index, 2u);
}
