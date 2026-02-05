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
  @location(4) @interpolate(flat) index: u32,
  @location(5) @interpolate(flat) instanceId: u32,
};

struct VertexOutput3d {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) lam: vec3<f32>,
  @location(2) @interpolate(flat) id: u32,
  @location(3) n: vec3<f32>,
};

struct Triangle {
  p: array<vec3<f32>, 3>,
  nr: u32,
  npElement: u32,
  trigOfElement: u32,
  index: u32,
};

fn loadTriangle(vertexId: u32, instanceId: u32) -> Triangle {
    let MESHDATA_OFFSET : u32 = 2;
    var tri: Triangle;

    var trigId = instanceId;
    let numElements = trigs[0];

    let isSecondTrigOfQuad = (trigId >= numElements);
    if(isSecondTrigOfQuad) {
      trigId = trigId - numElements;
      trigId = trigs[MESHDATA_OFFSET + numElements*4 + trigId];
    }

    tri.nr = trigId;

    tri.index = trigs[4 * trigId + 3 + MESHDATA_OFFSET];
    let signedIndex = bitcast<i32>(tri.index);
    if(signedIndex < 0) {
      tri.npElement = 4;
      tri.index = trigs[u32(-signedIndex) + 1];
    }
    else {
      tri.npElement = 3;
    }

    var vid = vec3u(0,0,0);

    tri.trigOfElement = 0;
    vid = vec3u(
          trigs[4 * trigId + 0 + MESHDATA_OFFSET],
          trigs[4 * trigId + 1 + MESHDATA_OFFSET],
          trigs[4 * trigId + 2 + MESHDATA_OFFSET]
      );
      
    if(tri.npElement==4){
        if(isSecondTrigOfQuad) {
            tri.trigOfElement = 1;
            vid = vec3u(
                  trigs[4 * trigId + 2 + MESHDATA_OFFSET],
                  trigs[u32(-signedIndex)],
                  trigs[4 * trigId + 1 + MESHDATA_OFFSET],
              );
        }
        else {
            vid = vec3u(
                  trigs[4 * trigId + 0 + MESHDATA_OFFSET],
                  trigs[4 * trigId + 1 + MESHDATA_OFFSET],
                  trigs[u32(-signedIndex)],
              );
        }
          
    }
    
    vid = 3 * vid;
    tri.p = array<vec3<f32>, 3>(
        vec3<f32>(vertices[vid[0] ], vertices[vid[0] + 1], vertices[vid[0] + 2]),
        vec3<f32>(vertices[vid[1] ], vertices[vid[1] + 1], vertices[vid[1] + 2]),
        vec3<f32>(vertices[vid[2] ], vertices[vid[2] + 1], vertices[vid[2] + 2])
    );
    return tri;
}

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

fn calcTriLam(tri: Triangle, vertexId: u32, h: f32) -> vec2<f32> {
    let i3 = vertexId % 3u;
    var lam = vec2f(0.0, 0.0);
    
    if(tri.npElement ==3) {
        if (i3) < 2 {
            lam[i3] += h;
        }
    }
    else {
        if(i3 > 0u) {
            lam[i3-1] += h;
        }
    }
    
    return lam;
}

fn calcTrig(tri: Triangle, vertexId: u32, instanceId: u32)
  -> VertexOutput2d {
    let p = tri.p;
    let trigId = tri.nr;
    let index = tri.index;
    let subdivision = u_subdivision;
    let h = 1.0 / f32(subdivision);

    var lam = calcTriLam(tri, vertexId, h);

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

    
    if(tri.npElement == 4 && tri.trigOfElement == 0)
        {
            // lam.x += 0.5;
            // position = vec3f(0., 0., 0.);
        }

    let mapped_position = cameraMapPoint(position);
    return VertexOutput2d(mapped_position, position, lam, trigId, normal,
                          index, instanceId);
}


@vertex
fn vertexTrigP1Indexed(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) instanceId: u32) -> VertexOutput2d {

    let tri = loadTriangle(vertexId, instanceId);
    return calcTrig(tri, vertexId, instanceId);
}

@vertex
fn vertexWireframe2d(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) trigId: u32) -> VertexOutput2d {
    // let MESHDATA_OFFSET : u32 = 2;
    let tri = loadTriangle(vertexId, trigId);

    let index = tri.index;

    let subdivision = u_subdivision;
    let h = 1./ f32(subdivision);
    var lam = vec2f(0.0, 0.0);
    var position: vec3f;
    
    lam = calcTriLam(tri, vertexId, h);

    if(subdivision == 1)
      {
        var pi = (vertexId+2) % 3u;

        // For quads, don't draw the diagonal edge (p2-p0) 
        // in order to do that, just draw the "last" vertex at p2 again, in case we are at p0
        if(tri.npElement == 4u && pi == 1u) {
          pi = 2u;
        }

        position = tri.p[pi];
      }
    else
      {
        position = evalTrigVec3(&u_curvature_values_2d, tri.nr, lam);
      }
    if (u_deformation_values_2d[0] != -1.) {
      position += u_deformation_scale * evalTrigVec3(&u_deformation_values_2d, tri.nr, lam);
    }
    return VertexOutput2d(cameraMapPoint(position), position, lam, tri.nr,
                          normalize(cross(tri.p[1] - tri.p[0], tri.p[2] - tri.p[0])),
                          index, trigId);
}


@fragment
fn fragmentTrig(input: VertexOutput2d) -> @location(0) vec4<f32> {
    checkClipping(input.p);
    let p = &u_function_values_2d;
    let value = evalTrig(p, input.instanceId, u_function_component, input.lam);
    let color = getColor(value);
    if(color.a < 0.01) {
        discard;
    }
    return lightCalcColor(input.p, input.n, color);
}

@fragment
fn fragmentEdge(@location(0) p: vec3<f32>) -> @location(0) vec4<f32> {
    checkClipping(p);
    return vec4<f32>(0, 0, 0, 1.0);
}
