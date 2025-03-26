
struct MeshFragmentInput {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) color_val: f32,
  @location(1) p: vec3<f32>,
  @location(2) n: vec3<f32>,
  @location(3) @interpolate(flat) id: u32,
  @location(4) @interpolate(flat) index: u32,
};

@group(0) @binding(82) var<storage, read> subtrigs: array<SubTrig>;
@group(0) @binding(84) var<storage> draw_function_values: array<f32>;


fn calcMeshFace(color_val: f32, p: array<vec3<f32>, 3>, vertId: u32, nr: u32, index: u32) -> MeshFragmentInput {
    let n = cross(p[1] - p[0], p[2] - p[0]);
    let point = p[vertId];
    let position = calcPosition(point);
    return MeshFragmentInput(position, color_val, point, n, nr, index);
}

fn calcPosition(p: vec3<f32>) -> vec4<f32> {
    return u_view.model_view_projection * vec4<f32>(p, 1.0);
}

@vertex
fn vertexIsoSurface(@builtin(vertex_index) vertId: u32, @builtin(instance_index) trigid: u32) -> MeshFragmentInput {
  let trig = subtrigs[trigid];
  let p = get_tet_points(trig.id);
  var points = array<vec3<f32>, 3>(vec3f(0.0), vec3f(0.0), vec3f(0.0));
  for(var k = 0u; k < 3u; k++)
    {
      let lam = vec4f(trig.lam[k], 1.0 - trig.lam[k].x - trig.lam[k].y - trig.lam[k].z);
      points[k] = lam.x * p[0] + lam.y * p[1] + lam.z * p[2] + lam.w * p[3];
    }
  let mylam = vec4f(trig.lam[vertId], 1.0 -trig.lam[vertId].x - trig.lam[vertId].y - trig.lam[vertId].z);
  let func_value = mylam.x * draw_function_values[trig.id * 4] + mylam.y * draw_function_values[trig.id * 4 + 1] + mylam.z * draw_function_values[trig.id * 4 + 2] + mylam.w * draw_function_values[trig.id * 4 + 3];
  return calcMeshFace(func_value, points, vertId, 0,0);
}

@fragment
fn fragmentIsoSurface(input: MeshFragmentInput) -> @location(0) vec4<f32> {
  let p = input.p;
  checkClipping(input.p);
  let n4 = u_view.normal_mat * vec4(input.n, 1.0);
  let n = normalize(n4.xyz);
  let brightness = clamp(dot(n, normalize(vec3<f32>(-1., -3., -3.))), .0, 1.) * 0.7 + 0.3;
  let color = getColor(input.color_val).xyz * brightness;
  return vec4<f32>(color, 1.);
}

