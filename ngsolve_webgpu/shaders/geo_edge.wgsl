#import camera
#import clipping

@group(0) @binding(90) var<storage> u_vertices: array<f32>;
@group(0) @binding(91) var<storage> u_color: array<f32>;
@group(0) @binding(92) var<uniform> u_thickness: f32;
@group(0) @binding(93) var<storage> u_indices: array<u32>;

struct GeoEdgeInput
{
  @builtin(position) position: vec4<f32>,
  @location(1) @interpolate(flat) index: u32,
};


@vertex
fn vertex_main(@builtin(vertex_index) vertId: u32,
               @builtin(instance_index) instanceId: u32) -> GeoEdgeInput
{
  let p1 = vec3f(u_vertices[instanceId * 6],
                 u_vertices[instanceId * 6 + 1],
                 u_vertices[instanceId * 6 + 2]);
  let p2 = vec3f(u_vertices[instanceId * 6 + 3],
                 u_vertices[instanceId * 6 + 4],
                 u_vertices[instanceId * 6 + 5]);
  let tp1 = cameraMapPoint(p1);
  let tp2 = cameraMapPoint(p2);
  let v = normalize(tp2.xy*tp1.w - tp1.xy*tp2.w);
  let v2 = vec2<f32>(-v.y, v.x);
  var pos: vec4<f32>;
  if(vertId == 0) {
    pos = vec4<f32>(tp1.xy + (-0.5*v.xy + v2) * u_thickness/2.*tp1.w, tp1.zw); }
  else if(vertId == 1) {
    pos = vec4<f32>(tp1.xy + (-0.5*v.xy + - v2) * u_thickness/2.*tp1.w, tp1.zw); }
  else if(vertId == 2) {
    pos = vec4<f32>(tp2.xy + (0.5*v.xy + v2) * u_thickness/2.*tp2.w, tp2.zw); }
  else {
    pos = vec4<f32>(tp2.xy + (0.5*v.xy - v2) * u_thickness/2.*tp2.w, tp2.zw); }
  
  return GeoEdgeInput(pos, u_indices[instanceId]);
}

@fragment
fn fragment_main(input: GeoEdgeInput) -> @location(0) vec4<f32> {
  if (u_color[input.index*4+3] == 0.0) {
    discard;
  }
  return vec4<f32>(u_color[input.index * 4],
                   u_color[input.index * 4 + 1],
                   u_color[input.index * 4 + 2],
                   u_color[input.index * 4 + 3]);
}

@fragment
fn fragmentQueryIndex(input: GeoEdgeInput) -> @location(0) vec2<u32> {
  return vec2<u32>(input.index, 1u);
}
