#import camera
#import clipping

@group(0) @binding(90) var<storage> u_vertices: array<f32>;
@group(0) @binding(91) var<storage> u_vertex_color: array<f32>;
@group(0) @binding(92) var<uniform> u_vertex_thickness: f32;

struct GeoVertexInput
{
  @builtin(position) position: vec4<f32>,
  @location(1) pos: vec4<f32>,
  @location(2) pos2: vec4<f32>,
  @location(3) p: vec3<f32>,
  @location(4) @interpolate(flat) index: u32,
};


@vertex
fn vertex_main(@builtin(vertex_index) vertId: u32,
               @builtin(instance_index) vertIndex: u32) -> GeoVertexInput
{
  let point = vec3<f32>(u_vertices[vertIndex * 3u],
                        u_vertices[vertIndex * 3u + 1u],
                        u_vertices[vertIndex * 3u + 2u]);
  let pos = cameraMapPoint(point);
  var position = pos;
  // draw a rectangle, we discard points outside the circle in the fragment shader
  if(vertId == 0)
    {
      position.x -= u_vertex_thickness/2.;
      position.y += u_vertex_thickness/2.;
    }
  else if(vertId == 1)
    {
      position.x -= u_vertex_thickness/2.;
      position.y -= u_vertex_thickness/2.;
    }
  else if(vertId == 2)
    {
      position.x += u_vertex_thickness/2.;
      position.y += u_vertex_thickness/2.;
    }
  else if(vertId == 3)
    {
      position.x += u_vertex_thickness/2.;
      position.y -= u_vertex_thickness/2.;
    }

  return GeoVertexInput(position, position, pos, point, vertIndex);
}

@fragment
fn fragment_main(input: GeoVertexInput) -> @location(0) vec4<f32> {
  if(length(input.pos.xy - input.pos2.xy) > u_vertex_thickness/2.) {
    discard;
  }
  if (u_vertex_color[input.index*4+3] == 0.) {
    discard;
  }

  return vec4<f32>(u_vertex_color[input.index * 4],
                   u_vertex_color[input.index * 4 + 1],
                   u_vertex_color[input.index * 4 + 2],
                   u_vertex_color[input.index * 4 + 3]);
}

@fragment
fn fragmentQueryIndex(input: GeoVertexInput) -> @location(0) vec2<u32> {
  if(length(input.pos.xy - input.pos2.xy) > u_vertex_thickness/2.) {
    discard;
  }
  return vec2<u32>(input.index, 0u);
}
