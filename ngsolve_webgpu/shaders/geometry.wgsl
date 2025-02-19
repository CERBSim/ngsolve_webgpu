
@group(0) @binding(90) var<storage> u_vertices : array<f32>;
@group(0) @binding(91) var<storage> u_normals : array<f32>;
@group(0) @binding(92) var<storage> u_indices : array<u32>;

struct GeoFragmentInput {
  @builtin(position) position: vec4<f32>,
    @location(0) p: vec3<f32>,
    @location(1) n: vec3<f32>,
    @location(2) @interpolate(flat) id: u32,
    @location(3) @interpolate(flat) index: u32,
};

@vertex
fn vertexGeo(@builtin(vertex_index) vertId: u32, @builtin(instance_index) trigId: u32) -> GeoFragmentInput {
  let point = vec3<f32>(u_vertices[trigId * 9 + vertId * 3],
                        u_vertices[trigId * 9 + vertId * 3 + 1],
                        u_vertices[trigId * 9 + vertId * 3 + 2]);
  let normal = -vec3<f32>(u_normals[trigId * 9 + vertId * 3],
                         u_normals[trigId * 9 + vertId * 3 + 1],
                         u_normals[trigId * 9 + vertId * 3 + 2]);
  let position = cameraMapPoint(point);
  return GeoFragmentInput(position,
                          point,
                          normal,
                          trigId, u_indices[trigId]);
}

@fragment
fn fragmentGeo(input: GeoFragmentInput) -> @location(0) vec4<f32> {
  return lightCalcColor(input.n, vec4f(0., 1., 0., 1.));
}
