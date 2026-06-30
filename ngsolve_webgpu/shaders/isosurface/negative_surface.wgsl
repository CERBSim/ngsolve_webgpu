#import ngsolve/mesh/render
#import ngsolve/isosurface/common

@group(0) @binding(80) var<storage> levelset_values : array<f32>;

@fragment
fn fragmentCheckLevelset(input: VertexOutput2d) -> @location(0) vec4<f32> {
    checkClipping(input.p);
    let p = &u_function_values_2d;
    let value = evalTrig(p, input.instanceId, 0, input.lam) * input.value_sign;
    let pl = &levelset_values;
    let levelset = evalTrig(pl, input.instanceId, 0, input.lam);
    if(levelset < u_iso.value) {
      discard;
    }
    return lightCalcColor(input.p, input.n, getColor(value));
}
