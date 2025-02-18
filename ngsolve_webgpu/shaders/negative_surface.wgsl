
@group(0) @binding(80) var<storage> levelset_values : array<f32>;

@fragment
fn fragmentCheckLevelset(input: VertexOutput2d) -> @location(0) vec4<f32> {
    checkClipping(input.p);
    let p = &trig_function_values;
    let value = evalTrig(p, input.id, 1u, input.lam);
    let pl = &levelset_values;
    let levelset = evalTrig(pl, input.id, 1u, input.lam);
    if (levelset > 0.0) {
      discard;
    }
    return lightCalcColor(-input.n, getColor(value));
}
