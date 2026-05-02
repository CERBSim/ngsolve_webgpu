#import clipping
#import colormap
#import camera
#import light

#import ngsolve/eval/trig
#import ngsolve/mesh/utils
#ifdef SYMMETRY
#import ngsolve/symmetry
#endif SYMMETRY

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
  @location(6) @interpolate(flat) value_sign: f32,
};

struct VertexOutput3d {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) lam: vec3<f32>,
  @location(2) @interpolate(flat) id: u32,
  @location(3) n: vec3<f32>,
};

@vertex
fn vertexTrigP1Indexed(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) instanceId_: u32) -> VertexOutput2d {
#ifdef SYMMETRY
    let instanceId = symGetElementIndex(instanceId_);
#else SYMMETRY
    let instanceId = instanceId_;
#endif SYMMETRY
    let tri = loadTriangle(instanceId);
    return calcTrig(tri, vertexId, instanceId, instanceId_);
}

@vertex
fn vertexWireframe2d(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) trigId_: u32) -> VertexOutput2d {
#ifdef SYMMETRY
    let trigId = symGetElementIndex(trigId_);
#else SYMMETRY
    let trigId = trigId_;
#endif SYMMETRY
    // let MESHDATA_OFFSET : u32 = 2;
    let tri = loadTriangle(trigId);

    let index = tri.index;

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


    if(subdivision == 1 || mesh.is_curved == 0u)
      {
        var pi = (vertexId+2) % 3u;

        // For quads, don't draw the diagonal edge (p2-p0) 
        // in order to do that, just draw the "last" vertex at p1 again
        if(tri.npElement == 4u && vertexId == 3u) {
          // todo: fix curved quads
          lam = calcTriLam(tri, vertexId, h);
          pi = 1u;
        }

        position = tri.p[pi];
      }
    else
      {
        position = evalTrigVec3(&mesh.data, trigId, lam, mesh.offset_curvature_2d);
      }
    if (u_deformation_values_2d[0] != -1.) {
      position += u_deformation_scale * evalTrigVec3Complex(&u_deformation_values_2d, trigId, lam, 0u);
    }
    var normal = normalize(cross(tri.p[1] - tri.p[0], tri.p[2] - tri.p[0]));
#ifdef SYMMETRY
    position = symApplyPosition(position, trigId_);
    normal = symApplyNormal(normal, trigId_);
#endif SYMMETRY
#ifdef SYMMETRY
    let value_sign = symGetValueSign(trigId_);
#else SYMMETRY
    let value_sign = 1.0;
#endif SYMMETRY
    return VertexOutput2d(cameraMapPoint(position), position, lam, tri.nr,
                          normal, index, trigId, value_sign);
}


@fragment
fn fragmentTrig(input: VertexOutput2d) -> @location(0) vec4<f32> {
    checkClipping(input.p);
    let p = &u_function_values_2d;
    let value = evalTrig(p, input.instanceId, u_function_component, input.lam) * input.value_sign;
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

struct MeshFragmentInput {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) n: vec3<f32>,
  @location(2) @interpolate(flat) id: u32,
  @location(3) @interpolate(flat) index: u32,
};

// A triangle as part of a 3d element (thus, 3 barycentric coordinates)
struct SubTrig {
  lam: array<vec3f, 3>,
  id: u32,
}

struct ClipTetResult {
  n: u32,
  trigs: array<SubTrig, 2>,
}

fn clipCheckOrientation(trig: SubTrig, values: array<f32, 4>) -> SubTrig {
  var result = trig;
  let n = cross(
    trig.lam[1] - trig.lam[0],
    trig.lam[2] - trig.lam[0]
  );

  let p = 1.0/3.*(trig.lam[0] + trig.lam[1] + trig.lam[2]) + n;
  let p4 = 1.0 - p.x - p.y - p.z;
  let value = p.x * values[0] + p.y * values[1] + p.z * values[2] + p4 * values[3];

  if(value<0.0) {
    result.lam[1] = trig.lam[2];
    result.lam[2] = trig.lam[1];
  }

  return result;
}

// clip tet such that the clip triangle(s) have value 0 everywhere
fn clipTet(lam: array<vec3f, 4>, values: array<f32, 4>, ei: u32) -> ClipTetResult {
    var trigs = ClipTetResult(0, array<SubTrig, 2>(SubTrig(array<vec3f, 3>(vec3f(0.0), vec3f(0.0), vec3f(0.0)), 0), SubTrig(array<vec3f, 3>(vec3f(0.0), vec3f(0.0), vec3f(0.0)), 0)));
    var p_pos = array<u32, 4>(0u, 0u, 0u, 0u);
    var p_neg = array<u32, 4>(0u, 0u, 0u, 0u);

    var n_pos: u32 = 0u;
    var n_neg: u32 = 0u;

    for (var i = 0u; i < 4u; i++) {
      if (values[i] > 0.0) {
            p_pos[n_pos] = i;
            n_pos++;
        } else {
            p_neg[n_neg] = i;
            n_neg++;
        }
    }

    if n_pos == 4u || n_neg == 4u {
        return trigs;
    }

    if n_pos == 3u {
        trigs.n = 1u;
        n_pos = 1u;
        n_neg = 3u;
        p_pos[3] = p_neg[0];
        p_neg[0] = p_pos[0];
        p_neg[1] = p_pos[1];
        p_neg[2] = p_pos[2];
        p_pos[0] = p_pos[3];
    }

    if n_pos == 1u {
        trigs.n = 1u;
        trigs.trigs[0].id = ei;
        for (var i = 0u; i < 3u; i++) {
            let t = values[p_pos[0] ] / (values[p_pos[0] ] - values[p_neg[i] ]);
            let lam_trig = mix(lam[p_pos[0] ], lam[p_neg[i] ], t);
            trigs.trigs[0].lam[i] = lam_trig;
        }

        trigs.trigs[0] = clipCheckOrientation(trigs.trigs[0], values);
        return trigs;
    }

  // two points before, two points behind clipping plane
  // -> two triangles
    trigs.n = 2u;
    var pairs = array<vec2u,4>(
        vec2u(p_pos[1], p_neg[0]),
        vec2u(p_pos[0], p_neg[0]),
        vec2u(p_pos[0], p_neg[1]),
        vec2u(p_pos[1], p_neg[1])
    );
    var points: array<vec3f, 4> = array<vec3f, 4>(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
    for (var i = 0; i < 4; i++) {
        let p0 = pairs[i].x;
        let p1 = pairs[i].y;
        let t = values[p0 ] / (values[p0] - values[p1]);
        let lam_trig = mix(lam[p0], lam[p1], t);
        points[i] = lam_trig;
    }
    trigs.trigs[0].id = ei;
    trigs.trigs[0].lam = array(points[0], points[1], points[2]);
    trigs.trigs[1].id = ei;
    trigs.trigs[1].lam = array(points[0], points[2], points[3]);

    trigs.trigs[0] = clipCheckOrientation(trigs.trigs[0], values);
    trigs.trigs[1] = clipCheckOrientation(trigs.trigs[1], values);
    return trigs;
}


@fragment
fn fragmentMesh(input: MeshFragmentInput) -> @location(0) vec4<f32> {
    let color = getColor(f32(input.index));
#ifdef OPAQUE_PASS
    if (color.a < 1.0) { discard; }
#endif OPAQUE_PASS
#ifdef TRANSPARENT_PASS
    if (color.a >= 1.0) { discard; }
#endif TRANSPARENT_PASS
    return lightCalcColor(input.p, input.n, color);
}

@group(0) @binding(54) var<storage> u_mesh_color: vec4<f32>;

@fragment
fn fragment2dElement(input: VertexOutput2d) -> @location(0) vec4<f32> {
  checkClipping(input.p);
  let color = getColor(f32(input.index) + 0.5);
  if(color.a < 0.01) {
    discard;
  }
#ifdef OPAQUE_PASS
  if (color.a < 1.0) { discard; }
#endif OPAQUE_PASS
#ifdef TRANSPARENT_PASS
  if (color.a >= 1.0) { discard; }
#endif TRANSPARENT_PASS
  return lightCalcColor(input.p, input.n, color);
}

#ifdef SELECT_PIPELINE
@fragment fn select2dElement(
    input: VertexOutput2d
) -> @location(0) vec4<u32> {
    checkClipping(input.p);
    let color = getColor(f32(input.index) + 0.5);
    if(color.a < 0.01) {
      discard;
    }
    return vec4<u32>(@RENDER_OBJECT_ID@, bitcast<u32>(input.fragPosition.z), 0, 0);
}
#endif SELECT_PIPELINE

@fragment
fn fragmentWireframe2d(input: VertexOutput2d) -> @location(0) vec4<f32> {
  checkClipping(input.p);
  return lightCalcColor(input.p, input.n, u_mesh_color);
}

fn calcTrig(tri: Triangle, vertexId: u32, instanceId: u32, rawInstanceId: u32)
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
          let pos_and_gradients = u_deformation_scale * evalTrigVec3GradComplex(&u_deformation_values_2d, instanceId, lam, 0u);
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


        var pos_and_gradients = evalTrigVec3Grad(&mesh.data, instanceId, lam, mesh.offset_curvature_2d);
        if (u_deformation_values_2d[0] != -1.) {
          pos_and_gradients += u_deformation_scale * evalTrigVec3GradComplex(&u_deformation_values_2d, instanceId, lam, 0u);
        }
        position = pos_and_gradients[0];
        normal = normalize(cross(pos_and_gradients[1], pos_and_gradients[2]));
    }

    
    if(tri.npElement == 4 && tri.trigOfElement == 0)
        {
            // lam.x += 0.5;
            // position = vec3f(0., 0., 0.);
        }

#ifdef SYMMETRY
    position = symApplyPosition(position, rawInstanceId);
    normal = symApplyNormal(normal, rawInstanceId);
#endif SYMMETRY
    let mapped_position = cameraMapPoint(position);
#ifdef SYMMETRY
    let value_sign = symGetValueSign(rawInstanceId);
#else SYMMETRY
    let value_sign = 1.0;
#endif SYMMETRY
    return VertexOutput2d(mapped_position, position, lam, trigId, normal,
                          index, instanceId, value_sign);
}