#import camera
#import clipping
#import ngsolve/highlight

@group(0) @binding(90) var<storage> u_vertices: array<f32>;
@group(0) @binding(91) var<storage> u_vertex_color: array<f32>;
@group(0) @binding(92) var<uniform> u_vertex_thickness: f32;

const VERTEX_DEPTH_OFFSET: f32 = 4.0e-4;

struct GeoVertexInput
{
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) @interpolate(flat) index: u32,
  @location(2) p: vec3<f32>,
};


@vertex
fn vertex_main(@builtin(vertex_index) vertId: u32,
               @builtin(instance_index) vertIndex: u32) -> GeoVertexInput
{
  let point = vec3<f32>(u_vertices[vertIndex * 3u],
                        u_vertices[vertIndex * 3u + 1u],
                        u_vertices[vertIndex * 3u + 2u]);
  var pos = cameraMapPoint(point);
  var vt = u_vertex_thickness;
#ifdef SELECT_PIPELINE
  vt = vt * 3.0;
#endif SELECT_PIPELINE

  var local = vec2<f32>(-1.0, 1.0);
  if(vertId == 1) { local = vec2<f32>(-1.0, -1.0); }
  else if(vertId == 2) { local = vec2<f32>(1.0, 1.0); }
  else if(vertId == 3) { local = vec2<f32>(1.0, -1.0); }

  pos.x += local.x * 0.5 * vt / u_camera.aspect;
  pos.y += local.y * 0.5 * vt;
  pos.z -= VERTEX_DEPTH_OFFSET * pos.w;

  return GeoVertexInput(pos, local, vertIndex, point);
}

@fragment
fn fragment_main(input: GeoVertexInput) -> @location(0) vec4<f32> {
  checkClipping(input.p);
  let r2 = dot(input.local, input.local);
  let r = sqrt(r2);
  let aa = max(fwidth(r), 1e-5);
  let coverage = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, r);
  if (coverage <= 0.0) {
    discard;
  }

  let a = u_vertex_color[input.index * 4 + 3];
  if (a == 0.) {
    discard;
  }
  let base = vec3<f32>(u_vertex_color[input.index * 4],
                       u_vertex_color[input.index * 4 + 1],
                       u_vertex_color[input.index * 4 + 2]);

  // shade the billboard as a small sphere lit from the upper left
  let normal = vec3<f32>(input.local, sqrt(max(1.0 - r2, 0.0)));
  let light_dir = normalize(vec3<f32>(-0.4, 0.5, 0.8));
  let diffuse = max(dot(normal, light_dir), 0.0);
  let ambient = 0.4;
  let half_vec = normalize(light_dir + vec3<f32>(0.0, 0.0, 1.0));
  let specular = pow(max(dot(normal, half_vec), 0.0), 24.0) * 0.4;
  let shaded = base * (ambient + (1.0 - ambient) * diffuse) + vec3<f32>(specular);

  let color = applyHighlight(vec4<f32>(shaded, a), input.index, input.index);
  // premultiplied alpha, faded at the rim for anti-aliasing
  let alpha = color.a * coverage;
  return vec4<f32>(color.rgb * alpha, alpha);
}

@fragment
fn fragmentQueryIndex(input: GeoVertexInput) -> @location(0) vec4<u32> {
  checkClipping(input.p);
  if (dot(input.local, input.local) > 1.0) {
    discard;
  }
  return vec4<u32>(@RENDER_OBJECT_ID@, bitcast<u32>(input.position.z), 0u, input.index);
}
