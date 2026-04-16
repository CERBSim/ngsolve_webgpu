#import ngsolve/clipping/render
#ifdef SYMMETRY
#import ngsolve/symmetry
#endif SYMMETRY

@vertex
fn vertex_isosurface(@builtin(vertex_index) vertId: u32,
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
  var n = normalize(cross(p1 - p0, p2 - p0));

#ifdef SYMMETRY
  p = symApplyPosition(p, trigId_);
  n = symApplyNormal(n, trigId_);
#endif SYMMETRY
#ifdef SYMMETRY
  let value_sign = symGetValueSign(trigId_);
#else SYMMETRY
  let value_sign = 1.0;
#endif SYMMETRY
  return VertexOutputClip(cameraMapPoint(p), p, -n, lam.xyz,
                      trig.id, value_sign);
}


@fragment
fn fragment_isosurface(input: VertexOutputClip) -> @location(0) vec4<f32>
{
  checkClipping(input.p);
  let value = evalTet(&u_function_values_3d, input.elnr, 0, input.lam) * input.value_sign;
  return lightCalcColor(input.p, input.n, getColor(value));
}
