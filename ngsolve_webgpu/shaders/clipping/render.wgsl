#import camera
#import light
#import colormap
#import clipping
#import ngsolve/clipping/common
#ifdef SYMMETRY
#import ngsolve/symmetry
#endif SYMMETRY

@group(0) @binding(24) var<storage> subtrigs: array<SubTrig>;
@group(0) @binding(55) var<uniform> u_component: i32;

struct VertexOutputClip {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) n: vec3<f32>,
  @location(2) lam: vec3<f32>,
  @location(3) @interpolate(flat) elnr: u32,
  @location(4) @interpolate(flat) value_sign: f32,
};

@vertex
fn vertex_clipping(@builtin(vertex_index) vertId: u32,
               @builtin(instance_index) trigId_: u32)
  -> VertexOutputClip
{
#ifdef SYMMETRY
  let trigId = symGetElementIndex(trigId_);
#else SYMMETRY
  let trigId = trigId_;
#endif SYMMETRY
  let trig = subtrigs[trigId];
  let points = getTetPoints(trig.id);
  var lam = vec4<f32>(trig.lam[vertId], 1.);
  lam[3] = 1.0 - lam[0] - lam[1] - lam[2];
  var p = vec3<f32>(0.0, 0.0, 0.0);
  for(var i = 0u; i < 4u; i = i + 1u) {
    p = p + lam[i] * points[i];
  }
  var n = u_clipping.plane.xyz;
#ifdef SYMMETRY
  p = symApplyPosition(p, trigId_);
  n = symApplyNormal(n, trigId_);
#endif SYMMETRY
  
#ifdef SYMMETRY
  let value_sign = symGetValueSign(trigId_);
#else SYMMETRY
  let value_sign = 1.0;
#endif SYMMETRY
  return VertexOutputClip(cameraMapPoint(p), p, n, lam.xyz, trig.id, value_sign);
}

@fragment
fn fragment_clipping(input: VertexOutputClip) -> @location(0) vec4<f32>
{
  let value = evalTet(&u_function_values_3d, input.elnr, u_component, input.lam) * input.value_sign;
  return lightCalcColor(input.p, input.n, getColor(value));
}
