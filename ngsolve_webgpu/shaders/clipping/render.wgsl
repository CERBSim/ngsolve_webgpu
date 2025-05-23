#import camera
#import light
#import colormap
#import clipping
#import ngsolve/clipping/common

@group(0) @binding(24) var<storage> subtrigs: array<SubTrig>;

struct VertexOutputClip {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) n: vec3<f32>,
  @location(2) lam: vec3<f32>,
  @location(3) @interpolate(flat) elnr: u32
};

@vertex
fn vertex_main(@builtin(vertex_index) vertId: u32,
               @builtin(instance_index) trigId: u32)
  -> VertexOutputClip
{
  let trig = subtrigs[trigId];
  let points = get_tet_points(trig.id);
  var lam = vec4<f32>(trig.lam[vertId], 1.);
  lam[3] = 1.0 - lam[0] - lam[1] - lam[2];
  var p = vec3<f32>(0.0, 0.0, 0.0);
  for(var i = 0u; i < 4u; i = i + 1u) {
    p = p + lam[i] * points[i];
  }
  
  return VertexOutputClip(cameraMapPoint(p), p, u_clipping.plane.xyz, lam.xyz,
                      trig.id);
}

@fragment
fn fragment_main(input: VertexOutputClip) -> @location(0) vec4<f32>
{
  let value = evalTet(&u_function_values_3d, input.elnr, 0, input.lam);
  return lightCalcColor(input.p, input.n, getColor(value));
}


