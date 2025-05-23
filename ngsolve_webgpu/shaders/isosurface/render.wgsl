#import ngsolve/clipping/render

struct MeshFragmentInput {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) color_val: f32,
  @location(1) p: vec3<f32>,
  @location(2) n: vec3<f32>,
  @location(3) @interpolate(flat) id: u32,
  @location(4) @interpolate(flat) index: u32,
};

@vertex
fn vertex_isosurface(@builtin(vertex_index) vertId: u32,
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

  var p0 = (1.-trig.lam[0][0]-trig.lam[0][1]-trig.lam[0][2]) * points[3];
  for(var i=0u; i<3u; i=i+1u) {
    p0 = p0 + trig.lam[0][i] * points[i];
  }
  var p1 = (1.-trig.lam[1][0]-trig.lam[1][1]-trig.lam[1][2]) * points[3];
  for(var i=0u; i<3u; i=i+1u) {
    p1 = p1 + trig.lam[1][i] * points[i];
  }
  var p2 = (1.-trig.lam[2][0]-trig.lam[2][1]-trig.lam[2][2]) * points[3];
  for(var i=0u; i<3u; i=i+1u) {
    p2 = p2 + trig.lam[2][i] * points[i];
  }
  let n = normalize(cross(p1 - p0, p2 - p0));

  return VertexOutputClip(cameraMapPoint(p), p, -n, lam.xyz,
                      trig.id);
}


@fragment
fn fragment_isosurface(input: VertexOutputClip) -> @location(0) vec4<f32>
{
  checkClipping(input.p);
  let value = evalTet(&u_function_values_3d, input.elnr, 0, input.lam);
  return lightCalcColor(input.p, input.n, getColor(value));
}
