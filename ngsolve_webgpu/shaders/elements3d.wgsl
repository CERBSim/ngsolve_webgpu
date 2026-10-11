#import clipping
#import camera
#import light
#import colormap
#import ngsolve/highlight
#import ngsolve/eval/common3d
#import ngsolve/mesh/render
#import ngsolve/eval/tet
#import ngsolve/eval/hex
#import ngsolve/eval/prism
#import ngsolve/eval/pyramid
#ifdef SYMMETRY
#import ngsolve/symmetry
#endif SYMMETRY

// Node index → reference tet barycentric coordinates
// NGSolve convention: ref (0,0,0)→node3, (1,0,0)→node0, (0,1,0)→node1, (0,0,1)→node2
const NODE_REF = array<vec3f, 4>(
    vec3f(1.0, 0.0, 0.0),
    vec3f(0.0, 1.0, 0.0),
    vec3f(0.0, 0.0, 1.0),
    vec3f(0.0, 0.0, 0.0),
);

const NODE_REF_HEX = array<vec3f, 8>(
    vec3f(0,0,0), vec3f(1,0,0), vec3f(1,1,0), vec3f(0,1,0),
    vec3f(0,0,1), vec3f(1,0,1), vec3f(1,1,1), vec3f(0,1,1),
);
const NODE_REF_PRISM = array<vec3f, 6>(
    vec3f(1,0,0), vec3f(0,1,0), vec3f(0,0,0),
    vec3f(1,0,1), vec3f(0,1,1), vec3f(0,0,1),
);
const NODE_REF_PYRAMID = array<vec3f, 5>(
    vec3f(0,0,0), vec3f(1,0,0), vec3f(1,1,0), vec3f(0,1,0), vec3f(0,0,1),
);

// Reference coordinates of local vertex i of an element with np vertices
fn nodeRef(np: u32, i: u32) -> vec3f {
    if (np == 5u) { return NODE_REF_PYRAMID[i]; }
    if (np == 6u) { return NODE_REF_PRISM[i]; }
    if (np == 8u) { return NODE_REF_HEX[i]; }
    return NODE_REF[i];
}

// Start of the true map coefficients of 3D element elementId in mesh.data,
// -1 if the element is drawn with straight sub-tets.
fn trueMapOffset(elementId: u32) -> i32 {
#ifdef TRUE_MAP_NONE
    return -1;
#else TRUE_MAP_NONE
    if ((mesh.is_curved & 2u) == 0u) {
        return -1;
    }
    let oc3d = mesh.offset_curvature_3d;
    if (bitcast<i32>(mesh.data[oc3d + 1u]) <= 0) {
        return -1;
    }
    let lookup_val = bitcast<i32>(mesh.data[oc3d + 2u + elementId]);
    if (lookup_val < 0) {
        return -1;
    }
    let n_total = bitcast<u32>(mesh.data[mesh.offset_3d_data]);
    return i32(oc3d + 2u + n_total) + lookup_val;
#endif TRUE_MAP_NONE
}

fn trueMapOrder() -> i32 {
    return bitcast<i32>(mesh.data[mesh.offset_curvature_3d]);
}

// True map and its derivatives in reference coordinates: mat4x3f(pos, dF/dx, dF/dy, dF/dz)
#ifdef TRUE_MAP_ORDER1
// all true maps are P1 (curve order 1): closed forms, much cheaper to compile than de Casteljau
fn _trueMapCoeff(coeff_start: u32, i: u32) -> vec3f {
    let b = coeff_start + 3u * i;
    return vec3f(mesh.data[b], mesh.data[b + 1u], mesh.data[b + 2u]);
}

// trilinear on [0,1]^3, coefficient ix + 2 iy + 4 iz
fn _trilinearGrad(cs: u32, l: vec3f) -> mat4x3f {
    let c000 = _trueMapCoeff(cs, 0u); let c100 = _trueMapCoeff(cs, 1u);
    let c010 = _trueMapCoeff(cs, 2u); let c110 = _trueMapCoeff(cs, 3u);
    let c001 = _trueMapCoeff(cs, 4u); let c101 = _trueMapCoeff(cs, 5u);
    let c011 = _trueMapCoeff(cs, 6u); let c111 = _trueMapCoeff(cs, 7u);
    let c00 = mix(c000, c100, l.x); let c10 = mix(c010, c110, l.x);
    let c01 = mix(c001, c101, l.x); let c11 = mix(c011, c111, l.x);
    let c0 = mix(c00, c10, l.y); let c1 = mix(c01, c11, l.y);
    let dx = mix(mix(c100 - c000, c110 - c010, l.y), mix(c101 - c001, c111 - c011, l.y), l.z);
    let dy = mix(c10 - c00, c11 - c01, l.z);
    return mat4x3f(mix(c0, c1, l.z), dx, dy, c1 - c0);
}

fn evalTrueMapGrad(np: u32, coeff_start: u32, ref_coord: vec3f) -> mat4x3f {
    if (np == 6u) {
        // z-layer major, triangle dofs weighted by (x, y, 1-x-y)
        let x = ref_coord.x; let y = ref_coord.y; let z = ref_coord.z;
        let v0 = _trueMapCoeff(coeff_start, 0u); let v1 = _trueMapCoeff(coeff_start, 1u);
        let v2 = _trueMapCoeff(coeff_start, 2u); let v3 = _trueMapCoeff(coeff_start, 3u);
        let v4 = _trueMapCoeff(coeff_start, 4u); let v5 = _trueMapCoeff(coeff_start, 5u);
        let p0 = x * v0 + y * v1 + (1.0 - x - y) * v2;
        let p1 = x * v3 + y * v4 + (1.0 - x - y) * v5;
        return mat4x3f(mix(p0, p1, z), mix(v0 - v2, v3 - v5, z), mix(v1 - v2, v4 - v5, z), p1 - p0);
    }
    if (np == 5u) {
        let noz = max(1.0 - ref_coord.z, 1e-6);
        let c = _pyramidCollapsed(ref_coord);
        let g = _trilinearGrad(coeff_start, c);
        return mat4x3f(g[0], g[1] / noz, g[2] / noz, g[3] + (c.x * g[1] + c.y * g[2]) / noz);
    }
    // hex, tets have no order 1 true map
    return _trilinearGrad(coeff_start, ref_coord);
}

fn evalTrueMap(np: u32, coeff_start: u32, ref_coord: vec3f) -> vec3f {
    return evalTrueMapGrad(np, coeff_start, ref_coord)[0];
}
#else TRUE_MAP_ORDER1
fn evalTrueMapGrad(np: u32, coeff_start: u32, ref_coord: vec3f) -> mat4x3f {
    let order = trueMapOrder();
    if (np == 5u) { return evalPyramidVec3GradAt(coeff_start, order, ref_coord); }
    if (np == 6u) { return evalPrismVec3GradAt(coeff_start, order, ref_coord); }
    if (np == 8u) { return evalHexVec3GradAt(coeff_start, order, ref_coord); }
    return evalTetVec3GradAtDirect(coeff_start, order, ref_coord);
}

fn evalTrueMap(np: u32, coeff_start: u32, ref_coord: vec3f) -> vec3f {
    let order = trueMapOrder();
    if (np == 5u) { return evalPyramidVec3At(coeff_start, order, ref_coord); }
    if (np == 6u) { return evalPrismVec3At(coeff_start, order, ref_coord); }
    if (np == 8u) { return evalHexVec3At(coeff_start, order, ref_coord); }
    return evalTetVec3AtDirect(coeff_start, order, ref_coord);
}

#endif TRUE_MAP_ORDER1

// Sub-tet barycentric lam (first three) -> reference coordinates of the parent element
fn subTetRefCoord(tet: Tetrahedron, lam: vec3f) -> vec3f {
    return lam.x * nodeRef(tet.np, tet.pi[0]) + lam.y * nodeRef(tet.np, tet.pi[1])
         + lam.z * nodeRef(tet.np, tet.pi[2]) + (1.0 - lam.x - lam.y - lam.z) * nodeRef(tet.np, tet.pi[3]);
}

// Decode face instance → element, face, subdivision
// Instance layout:
//   [0, 4*numElements)                         first 4 faces per element (flat, subdivision=1)
//   [4*numElements, +4*nCurved)                 curved tet faces (subdivided)
//   [+4*nCurved, +2*nPyra+4*nPrism+8*nHex)     non-tet extra faces (subdivided)
struct FaceInfo {
    elementId: u32,
    faceId: u32,
    subdivision: u32,
    inFlatBatch: u32,
}

fn decodeFaceInstance(instanceId: u32) -> FaceInfo {
    var info: FaceInfo;
    info.inFlatBatch = 0u;

    let offset_3d = mesh.offset_3d_data;
    let numElements = bitcast<u32>(mesh.data[offset_3d]);
    let numPyra = bitcast<u32>(mesh.data[offset_3d + 2u]);
    let numPrims = bitcast<u32>(mesh.data[offset_3d + 3u]);
    let numHex = bitcast<u32>(mesh.data[offset_3d + 4u]);
    let IDX = offset_3d + 5u + 5u * numElements;  // INDEX_SORTED_BY_TYPE

    var nCurved = 0u;
    if ((mesh.is_curved & 2u) != 0u) {
        nCurved = bitcast<u32>(mesh.data[mesh.offset_curvature_3d + 1u]);
    }

    let extraStart = 4u * numElements;
    let curvedEnd = extraStart + 4u * nCurved;
    let pyramidStart = curvedEnd;
    let prismStart = pyramidStart + 2u * numPyra;
    let hexStart = prismStart + 4u * numPrims;

    if (instanceId < extraStart) {
        // First 4 faces per element (flat)
        info.faceId = instanceId % 4u;
        info.elementId = instanceId / 4u;
        info.subdivision = 1u;
        info.inFlatBatch = 1u;
    } else if (instanceId < curvedEnd) {
        // Curved tet faces
        let local = instanceId - extraStart;
        info.faceId = local % 4u;
        info.elementId = bitcast<u32>(mesh.data[IDX + (local / 4u)]);
        info.subdivision = u_mesh.subdivision;
    } else if (instanceId < prismStart) {
        // Pyramid extra faces
        let local = instanceId - pyramidStart;
        info.faceId = local % 2u + 4u;
        info.elementId = bitcast<u32>(mesh.data[IDX + nCurved + (local / 2u)]);
        info.subdivision = u_mesh.subdivision;
    } else if (instanceId < hexStart) {
        // Prism extra faces
        let local = instanceId - prismStart;
        info.faceId = local % 4u + 4u;
        info.elementId = bitcast<u32>(mesh.data[IDX + nCurved + numPyra + (local / 4u)]);
        info.subdivision = u_mesh.subdivision;
    } else {
        // Hex extra faces
        let local = instanceId - hexStart;
        info.faceId = local % 8u + 4u;
        info.elementId = bitcast<u32>(mesh.data[IDX + nCurved + numPyra + numPrims + (local / 8u)]);
        info.subdivision = u_mesh.subdivision;
    }

    return info;
}

@vertex
fn vertex_main(@builtin(vertex_index) vertId: u32,
               @builtin(instance_index) instanceId_: u32)
  -> MeshFragmentInput
{
#ifdef SYMMETRY
    let instanceId = symGetElementIndex(instanceId_);
#else SYMMETRY
    let instanceId = instanceId_;
#endif SYMMETRY

    let zero = MeshFragmentInput(vec4f(0), vec3f(0), vec3f(0), 0u, 0u, vec3f(0), 0u);
    let info = decodeFaceInstance(instanceId);

    // Discard excess sub-triangles (needed when single-draw symmetry fallback
    // invokes flat instances with more vertices than subdivision² sub-triangles)
    let subTrigId = vertId / 3u;
    if (subTrigId >= info.subdivision * info.subdivision) {
        return zero;
    }

    var element = getElem(info.elementId);

#ifdef REGION_VISIBILITY
    if (regionAlphaVol(element.index) == 0.0) {
        return zero;
    }
#endif REGION_VISIBILITY

    let center = getCenter(element);

#ifdef FACET_CF
    // FACET_CF: no vertex-level clipping — fragment shader clips per-pixel
#else FACET_CF
    if (!calcClipping(center)) {
        return zero;
    }
#endif FACET_CF

    // Check if element is drawn with its true map
    let true_map = trueMapOffset(info.elementId);
    let is_curved = true_map >= 0;
    let curved_coeff_offset = u32(max(true_map, 0));

    if (info.inFlatBatch == 1u && is_curved) {
        return zero;
    }

    var face = getFace(element, info.faceId);

    // Subdivision grid
    let subdivision = info.subdivision;
    let h = 1.0 / f32(subdivision);
    let localVert = vertId % 3u;
    let ix = subTrigId % subdivision;
    let iy = subTrigId / subdivision;

    // Face-local (u, v) in reference triangle
    // localVert: 0→(h,0), 1→(0,h), 2→(0,0) then shifted by (ix,iy)*h
    var u = select(0.0, h, localVert == 0u) + f32(ix) * h;
    var v = select(0.0, h, localVert == 1u) + f32(iy) * h;
    if (ix + iy >= subdivision) {
        u = 1.0 - u;
        v = 1.0 - v;
    }

    var position: vec3f;
    var normal: vec3f;
    var ref_lam = vec3f(0.0);

    if (is_curved) {
        var fv: vec3u;
        if (element.np == 4u) {
            fv = TET_FACES[info.faceId];
        } else if (element.np == 5u) {
            fv = PYRA_FACES[info.faceId];
        } else if (element.np == 6u) {
            fv = PRISM_FACES[info.faceId];
        } else {
            fv = HEX_FACES[info.faceId];
        }
        let R0 = nodeRef(element.np, fv[0]);
        let R1 = nodeRef(element.np, fv[1]);
        let R2 = nodeRef(element.np, fv[2]);

        let lam = (1.0 - u - v) * R0 + u * R1 + v * R2;
        ref_lam = lam;

        let result = evalTrueMapGrad(element.np, curved_coeff_offset, lam);

        position = result[0];
        let dlam_du = R1 - R0;
        let dlam_dv = R2 - R0;
        let t1 = result[1] * dlam_du.x + result[2] * dlam_du.y + result[3] * dlam_du.z;
        let t2 = result[1] * dlam_dv.x + result[2] * dlam_dv.y + result[3] * dlam_dv.z;
        normal = cross(t2, t1);
    } else {
        // Flat: linear interpolation of face vertex positions
        position = (1.0 - u - v) * face.p[0] + u * face.p[1] + v * face.p[2];
        normal = cross(face.p[2] - face.p[0], face.p[1] - face.p[0]);
        if (element.np == 4u) {
            let fv = TET_FACES[info.faceId];
            let R0f = NODE_REF[fv[0]]; let R1f = NODE_REF[fv[1]]; let R2f = NODE_REF[fv[2]];
            ref_lam = (1.0 - u - v) * R0f + u * R1f + v * R2f;
        }
    }

    // Apply shrink toward element center
    position = mix(center, position, u_mesh.shrink);

#ifdef SYMMETRY
    position = symApplyPosition(position, instanceId_);
    normal = symApplyNormal(normal, instanceId_);
#endif SYMMETRY

    let mapped_position = cameraMapPoint(position);
#ifdef FACET_CF
    // Map shader faceId to MtAE face index (MtAE face g = opposite local vertex g)
    const MTAE_FOR_SHADER = array<u32, 4>(3u, 2u, 0u, 1u);
    let faceIndex = info.elementId * 4u + MTAE_FOR_SHADER[info.faceId];

    // Transform (u,v) to extraction barycentric coords.
    // MtAE sorts face vertices by ascending global vertex index.
    // evalTrig bary = (lam.x, lam.y, 1-lam.x-lam.y).
    // Shader position = (1-u-v)*V_a + u*V_b + v*V_c  (TET_FACES[f] = a,b,c)
    // So wa=1-u-v, wb=u, wc=v.  We need xi=weight of min-global, eta=weight of mid-global.
    let fv = TET_FACES[info.faceId];
    let ga = element.p[fv[0]];
    let gb = element.p[fv[1]];
    let gc = element.p[fv[2]];
    let wa = 1.0 - u - v;
    let wb = u;
    let wc = v;
    var ext_lam: vec2f;
    if (ga <= gb && ga <= gc) {
        ext_lam = vec2f(wa, select(wc, wb, gb <= gc));
    } else if (gb <= gc) {
        ext_lam = vec2f(wb, select(wc, wa, ga <= gc));
    } else {
        ext_lam = vec2f(wc, select(wb, wa, ga <= gb));
    }
    return MeshFragmentInput(mapped_position, position, normal, instanceId, element.index, vec3f(ext_lam, 0.0), faceIndex);
#else FACET_CF
    return MeshFragmentInput(mapped_position, position, normal, instanceId, element.index, ref_lam, info.elementId);
#endif FACET_CF
}

@fragment
fn fragment_main(input: MeshFragmentInput) -> @location(0) vec4<f32>
{
  let color = getColor(f32(input.index));
  if(color.a < 0.01) {
    discard;
  }
  return lightCalcColor(input.p, input.n, applyHighlight(color, input.id, input.index));
}

#ifdef FACET_CF
@group(0) @binding(81) var<storage> u_facet_values : array<f32>;

@fragment
fn fragment_facet_cf(input: MeshFragmentInput) -> @location(0) vec4<f32>
{
    checkClipping(input.p);
    let faceIndex = input.elementId;
    let value = evalTrig(&u_facet_values, faceIndex, u_function_component, input.lam.xy);
    let color = getColor(value);
    if(color.a < 0.01) {
        discard;
    }
    return lightCalcColor(input.p, -input.n, color);
}
#endif FACET_CF

#ifdef SELECT_PIPELINE
@fragment fn select3dElement(
    input: MeshFragmentInput
) -> @location(0) vec4<u32> {
    checkClipping(input.p);
    let color = getColor(f32(input.index));
    if(color.a < 0.01) {
      discard;
    }
    return vec4<u32>(@RENDER_OBJECT_ID@, bitcast<u32>(input.fragPosition.z), input.id, input.index);
}
#endif SELECT_PIPELINE
