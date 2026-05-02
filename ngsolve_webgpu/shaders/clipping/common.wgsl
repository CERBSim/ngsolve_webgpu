#import ngsolve/eval/tet
#import ngsolve/elements3d

struct Tet { p: array<u32, 4>, nr: u32, index: u32 };

fn getTetPoints(ei: u32) -> array<vec3f, 4> {
    let tet = getTetrahedron(ei);
    var p = array(getVertex(tet.p[0]), getVertex(tet.p[1]), getVertex(tet.p[2]), getVertex(tet.p[3]));
    let lam = array(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, 0.0));
    if (mesh.is_curved != 0u) {
        let oc3d = mesh.offset_curvature_3d;
        let n_curved = bitcast<i32>(mesh.data[oc3d + 1u]);
        if (n_curved > 0) {
            let lookup_val = bitcast<i32>(mesh.data[oc3d + 2u + tet.id]);
            if (lookup_val >= 0) {
                let order = bitcast<i32>(mesh.data[oc3d]);
                let ndof = u32((order + 1) * (order + 2) * (order + 3) / 6);
                let numElements = bitcast<u32>(mesh.data[mesh.offset_3d_data]);
                let local_id = u32(lookup_val);
                for (var pi = 0; pi < 4; pi++) {
                    p[pi] = evalTetVec3At(oc3d, numElements, local_id, ndof, lam[pi]);
                }
            }
        }
    }
    if(u_deformation_values_3d[0] != -1) {
        for(var pi = 0; pi < 4; pi++) {
            for(var xi = 0; xi < 3; xi++) {
                p[pi][xi] += evalTetComplex(&u_deformation_values_3d, ei, xi, lam[pi]);
            }
        }
    }
    return p;
}


