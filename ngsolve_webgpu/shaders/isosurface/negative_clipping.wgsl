#import ngsolve/clipping/render

@group(0) @binding(80) var<storage> levelset: array<f32>;

@fragment
fn fragment_neg_clip(input: VertexOutputClip) -> @location(0) vec4<f32>
{
  let lvlset = evalTet(&levelset, input.elnr, 0, input.lam);
  if(lvlset < 0.) {
    discard;
  }
  let value = evalTet(&u_function_values_3d, input.elnr, 0, input.lam);
  return lightCalcColor(input.p, input.n, getColor(value));
}
