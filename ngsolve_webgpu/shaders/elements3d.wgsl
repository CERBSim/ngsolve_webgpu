#import clipping
#import camera
#import light
#import colormap
#import ngsolve/eval/common3d
#import ngsolve/mesh/render
#import ngsolve/eval/tet
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

// Decode face instance → element, face, subdivision
// Instance layout:
//   [0, 4*numElements)                         first 4 faces per element (flat, subdivision=1)
//   [4*numElements, +4*nCurved)                 curved tet faces (subdivided)
//   [+4*nCurved, +2*nPyra+4*nPrism+8*nHex)     non-tet extra faces (subdivided)
struct FaceInfo {
    elementId: u32,
    faceId: u32,
    subdivision: u32,
}

fn decodeFaceInstance(instanceId: u32) -> FaceInfo {
    var info: FaceInfo;

    let offset_3d = mesh.offset_3d_data;
    let numElements = bitcast<u32>(mesh.data[offset_3d]);
    let numPyra = bitcast<u32>(mesh.data[offset_3d + 2u]);
    let numPrims = bitcast<u32>(mesh.data[offset_3d + 3u]);
    let numHex = bitcast<u32>(mesh.data[offset_3d + 4u]);
    let IDX = offset_3d + 5u + 5u * numElements;  // INDEX_SORTED_BY_TYPE

    var nCurved = 0u;
    if (mesh.is_curved != 0u) {
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

    let zero = MeshFragmentInput(vec4f(0), vec3f(0), vec3f(0), 0u, 0u);
    let info = decodeFaceInstance(instanceId);

    // Discard excess sub-triangles (needed when single-draw symmetry fallback
    // invokes flat instances with more vertices than subdivision² sub-triangles)
    let subTrigId = vertId / 3u;
    if (subTrigId >= info.subdivision * info.subdivision) {
        return zero;
    }

    var element = getElem(info.elementId);
    let center = getCenter(element);

    if (!calcClipping(center)) {
        return zero;
    }

    // Check if element is a curved tet
    var is_curved = false;
    var curved_local_id = 0u;
    if (mesh.is_curved != 0u && element.np == 4u) {
        let oc3d = mesh.offset_curvature_3d;
        let n_curved = bitcast<i32>(mesh.data[oc3d + 1u]);
        if (n_curved > 0) {
            let lookup_val = bitcast<i32>(mesh.data[oc3d + 2u + info.elementId]);
            if (lookup_val >= 0) {
                is_curved = true;
                curved_local_id = u32(lookup_val);
            }
        }
    }

    // In flat range: skip curved tets (they are rendered in the subdivided range)
    if (info.subdivision == 1u && is_curved) {
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

    if (is_curved) {
        // Map face (u, v) → tet reference coords via face vertex ref mapping
        let fv = TET_FACES[info.faceId];
        let R0 = NODE_REF[fv[0]];
        let R1 = NODE_REF[fv[1]];
        let R2 = NODE_REF[fv[2]];
        let lam = (1.0 - u - v) * R0 + u * R1 + v * R2;

        let oc3d = mesh.offset_curvature_3d;
        let order = bitcast<i32>(mesh.data[oc3d]);
        let ndof = u32((order + 1) * (order + 2) * (order + 3) / 6);
        let numElements = bitcast<u32>(mesh.data[mesh.offset_3d_data]);

        // result = [position, dF/dlam.x, dF/dlam.y, dF/dlam.z]
        let result = evalTetVec3GradAt(oc3d, numElements, curved_local_id, ndof, lam);
        position = result[0];

        // Face tangent vectors via chain rule: t = J * dlam/d(u or v)
        let dlam_du = R1 - R0;
        let dlam_dv = R2 - R0;
        let t1 = result[1] * dlam_du.x + result[2] * dlam_du.y + result[3] * dlam_du.z;
        let t2 = result[1] * dlam_dv.x + result[2] * dlam_dv.y + result[3] * dlam_dv.z;
        normal = cross(t2, t1);
    } else {
        // Flat: linear interpolation of face vertex positions
        position = (1.0 - u - v) * face.p[0] + u * face.p[1] + v * face.p[2];
        normal = cross(face.p[2] - face.p[0], face.p[1] - face.p[0]);
    }

    // Apply shrink toward element center
    position = mix(center, position, u_mesh.shrink);

#ifdef SYMMETRY
    position = symApplyPosition(position, instanceId_);
    normal = symApplyNormal(normal, instanceId_);
#endif SYMMETRY

    let mapped_position = cameraMapPoint(position);
    return MeshFragmentInput(mapped_position, position, normal, instanceId, element.index);
}

@fragment
fn fragment_main(input: MeshFragmentInput) -> @location(0) vec4<f32>
{
  let color = getColor(f32(input.index));
  if(color.a < 0.01) {
    discard;
  }
  return lightCalcColor(input.p, input.n, color);
}
