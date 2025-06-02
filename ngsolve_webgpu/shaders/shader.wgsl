#import ngsolve/eval/trig

struct VertexOutput1d {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) lam: f32,
  @location(2) @interpolate(flat) id: u32,
};

struct VertexOutput2d {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) lam: vec2<f32>,
  @location(2) @interpolate(flat) id: u32,
  @location(3) n: vec3<f32>,
};

struct VertexOutput3d {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) lam: vec3<f32>,
  @location(2) @interpolate(flat) id: u32,
  @location(3) n: vec3<f32>,
};

@vertex
fn vertexEdgeP1(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) edgeId: u32) -> VertexOutput1d {
    let edge = edges_p1[edgeId];
    var p: vec3<f32> = vec3<f32>(edge.p[3 * vertexId], edge.p[3 * vertexId + 1], edge.p[3 * vertexId + 2]);

    var lam: f32 = 0.0;
    if vertexId == 0 {
        lam = 1.0;
    }

    var position = cameraMapPoint(p);
    return VertexOutput1d(position, p, lam, edgeId);
}

fn calcTrig(p: array<vec3<f32>, 3>, vertexId: u32, trigId: u32) -> VertexOutput2d {
    let subdivision = u_subdivision;
    let h = 1.0 / f32(subdivision);

    var lam = vec2f(0.0, 0.0);
    if (vertexId % 3u) < 2 {
        lam[vertexId % 3u] += h;
    }

    var position: vec3f;
    var normal: vec3f;

    if subdivision == 1 {
        position = p[vertexId];
        if (u_deformation_values_2d[0] != -1.) {
          let pos_and_gradients = u_deformation_scale * evalTrigVec3Grad(&u_deformation_values_2d, trigId, lam);
          position += u_deformation_scale * pos_and_gradients[0];
          var v1 = p[0] - p[2] + u_deformation_scale * pos_and_gradients[1];
          var v2 = p[1] - p[2] + u_deformation_scale * pos_and_gradients[2];
          normal = normalize(cross(v1, v2));
        }
        else {
          normal = cross(p[1] - p[0], p[2] - p[0]);
        }
    } else {
        var subTrigId: u32 = vertexId / 3u;
        var ix = subTrigId % subdivision;
        var iy = subTrigId / subdivision;
        lam += h * vec2f(f32(ix), f32(iy));
        if ix + iy >= subdivision {
            lam[0] = 1.0 - lam[0];
            lam[1] = 1.0 - lam[1];
        }


        let data = &u_curvature_values_2d;
        var pos_and_gradients = evalTrigVec3Grad(data, trigId, lam);
        if (u_deformation_values_2d[0] != -1.) {
          pos_and_gradients += u_deformation_scale * evalTrigVec3Grad(&u_deformation_values_2d, trigId, lam);
        }
        position = pos_and_gradients[0];
        normal = normalize(cross(pos_and_gradients[1], pos_and_gradients[2]));
    }

    let mapped_position = cameraMapPoint(position);

    return VertexOutput2d(mapped_position, position, lam, trigId, normal);
}


@vertex
fn vertexTrigP1Indexed(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) trigId: u32) -> VertexOutput2d {
    var vid = 3 * vec3u(
        trigs[4 * trigId + 0],
        trigs[4 * trigId + 1],
        trigs[4 * trigId + 2]
    );

    var p = array<vec3<f32>, 3>(
        vec3<f32>(vertices[vid[0] ], vertices[vid[0] + 1], vertices[vid[0] + 2]),
        vec3<f32>(vertices[vid[1] ], vertices[vid[1] + 1], vertices[vid[1] + 2]),
        vec3<f32>(vertices[vid[2] ], vertices[vid[2] + 1], vertices[vid[2] + 2])
    );
    return calcTrig(p, vertexId, trigId);
}

@vertex
fn vertexWireframe2d(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) trigId: u32) -> VertexOutput2d {
    var vid = 3 * vec3u(
        trigs[4 * trigId + 0],
        trigs[4 * trigId + 1],
        trigs[4 * trigId + 2]
    );

    var p = array<vec3<f32>, 3>(
        vec3<f32>(vertices[vid[0] ], vertices[vid[0] + 1], vertices[vid[0] + 2]),
        vec3<f32>(vertices[vid[1] ], vertices[vid[1] + 1], vertices[vid[1] + 2]),
        vec3<f32>(vertices[vid[2] ], vertices[vid[2] + 1], vertices[vid[2] + 2])
    );

    let subdivision = u_subdivision;
    let h = 1./ f32(subdivision);
    var lam = vec2f(0.0, 0.0);
    var position: vec3f;
    var side = vertexId / subdivision;
    if (side >= 2u) {
      side = 2u;
    }
    var subId = vertexId - subdivision * side;
    if(side == 0u)
      {
        lam[0] = h * f32(subId);
        lam[1] = 0.;
      }
    else {
      if(side == 1u)
      {
        lam[0] = 1.0 - h * f32(subId);
        lam[1] = h * f32(subId);
      }
    else
      {
        lam[0] = 0.;
        lam[1] = 1. - h * f32(subId);
      }
    }
    if(subdivision == 1)
      {
        position = p[(vertexId+2)%3u];
      }
    else
      {
        position = evalTrigVec3(&u_curvature_values_2d, trigId, lam);
      }
    if (u_deformation_values_2d[0] != -1.) {
      position += u_deformation_scale * evalTrigVec3(&u_deformation_values_2d, trigId, lam);
    }
    return VertexOutput2d(cameraMapPoint(position), position, lam, trigId,
                          normalize(cross(p[1] - p[0], p[2] - p[0])));
}


@fragment
fn fragmentTrig(input: VertexOutput2d) -> @location(0) vec4<f32> {
    checkClipping(input.p);
    let p = &u_function_values_2d;
    let value = evalTrig(p, input.id, u_function_component, input.lam);
    return lightCalcColor(input.p, input.n, getColor(value));
}

@fragment
fn fragmentEdge(@location(0) p: vec3<f32>) -> @location(0) vec4<f32> {
    checkClipping(p);
    return vec4<f32>(0, 0, 0, 1.0);
}
