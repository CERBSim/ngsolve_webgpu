#import ngsolve/eval/tet
#import ngsolve/elements3d

struct Tet { p: array<u32, 4>, nr: u32, index: u32 };

fn getTetPoints(ei: u32) -> array<vec3f, 4> {
    let tet = getTetrahedron(ei);
    var p = array(getVertex(tet.p[0]), getVertex(tet.p[1]), getVertex(tet.p[2]), getVertex(tet.p[3]));
    let lam = array(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, 0.0));
    if(u_deformation_values_3d[0] != -1) {
        for(var pi = 0; pi < 4; pi++) {
            for(var xi = 0; xi < 3; xi++) {
                p[pi][xi] += evalTetComplex(&u_deformation_values_3d, ei, xi, lam[pi]);
            }
        }
    }
    return p;
}


