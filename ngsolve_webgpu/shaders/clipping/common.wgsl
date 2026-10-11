#import ngsolve/eval/tet
#import ngsolve/eval/hex
#import ngsolve/eval/prism
#import ngsolve/elements3d

struct Tet { p: array<u32, 4>, nr: u32, index: u32 };

fn getTetPoints(ei: u32) -> array<vec3f, 4> {
    let tet = getTetrahedron(ei);
    var p = array(getVertex(tet.p[0]), getVertex(tet.p[1]), getVertex(tet.p[2]), getVertex(tet.p[3]));
    let lam = array(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, 0.0));
    let true_map = trueMapOffset(tet.id);
    if (true_map >= 0) {
        for (var vi = 0; vi < 4; vi++) {
            p[vi] = evalTrueMap(tet.np, u32(true_map), nodeRef(tet.np, tet.pi[vi]));
        }
    }
    if(u_deformation_values_3d[0] != -1) {
        for(var pi = 0; pi < 4; pi++) {
            for(var xi = 0; xi < 3; xi++) {
                p[pi][xi] += u_deformation_scale * evalTetComplex(&u_deformation_values_3d, ei, xi, lam[pi]);
            }
        }
    }
    return p;
}

// Point with barycentric coordinates lam in sub-tet ei of an element with a true map,
// including the deformation
fn subTetPoint(ei: u32, tet: Tetrahedron, true_map: u32, lam: vec3f) -> vec3f {
    var p = evalTrueMap(tet.np, true_map, subTetRefCoord(tet, lam));
    if(u_deformation_values_3d[0] != -1) {
        for(var xi = 0; xi < 3; xi++) {
            p[xi] += u_deformation_scale * evalTetComplex(&u_deformation_values_3d, ei, xi, lam);
        }
    }
    return p;
}
