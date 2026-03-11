@group(0) @binding(110) var<storage> mesh_data : array<u32>;
@group(0) @binding(20) var<uniform> u_mesh : MeshUniforms;

struct MeshUniforms {
  subdivision: u32,
  shrink: f32,

  padding0: f32,
  padding1: f32,
};


struct Triangle {
  p: array<vec3<f32>, 3>,
  nr: u32,
  npElement: u32,
  trigOfElement: u32,
  index: u32,
};

fn getVertex(vertexId: u32) -> vec3f {
    let offset = mesh_data[0];
    
    return vec3f(
        bitcast<f32>(mesh_data[offset + 3*vertexId]),
        bitcast<f32>(mesh_data[offset + 3*vertexId + 1]),
        bitcast<f32>(mesh_data[offset + 3*vertexId + 2])
    );
}

fn loadTriangle(instanceId: u32) -> Triangle {
    let MESHDATA_OFFSET : u32 = 2;
    var tri: Triangle;

    var trigId = instanceId;
    
    let offset_2d = mesh_data[1];
    
    let numElements = mesh_data[offset_2d];

    let isSecondTrigOfQuad = (trigId >= numElements);
    if(isSecondTrigOfQuad) {
      trigId = trigId - numElements;
      trigId = mesh_data[offset_2d + MESHDATA_OFFSET + numElements*4 + trigId];
    }

    tri.nr = trigId;

    tri.index = mesh_data[ offset_2d + 4 * trigId + 3 + MESHDATA_OFFSET];
    let signedIndex = bitcast<i32>(tri.index);
    if(signedIndex < 0) {
      tri.npElement = 4;
      tri.index = mesh_data[offset_2d + u32(-signedIndex) + 1];
    }
    else {
      tri.npElement = 3;
    }

    var vid = vec3u(0,0,0);

    tri.trigOfElement = 0;
    vid = vec3u(
          mesh_data[offset_2d + 4 * trigId + 0 + MESHDATA_OFFSET],
          mesh_data[offset_2d + 4 * trigId + 1 + MESHDATA_OFFSET],
          mesh_data[offset_2d + 4 * trigId + 2 + MESHDATA_OFFSET]
      );
      
    if(tri.npElement==4){
        if(isSecondTrigOfQuad) {
            tri.trigOfElement = 1;
            vid = vec3u(
                  mesh_data[offset_2d + 4 * trigId + 2 + MESHDATA_OFFSET],
                  mesh_data[offset_2d + u32(-signedIndex) + 1],
                  mesh_data[offset_2d + 4 * trigId + 1 + MESHDATA_OFFSET],
              );
        }
        else {
            vid = vec3u(
                  mesh_data[offset_2d + 4 * trigId + 0 + MESHDATA_OFFSET],
                  mesh_data[offset_2d + 4 * trigId + 1 + MESHDATA_OFFSET],
                  mesh_data[offset_2d + u32(-signedIndex) + 1],
              );
        }
          
    }
    
    tri.p = array<vec3<f32>, 3>(
        getVertex(vid[0]),
        getVertex(vid[1]),
        getVertex(vid[2])
    );
    return tri;
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

