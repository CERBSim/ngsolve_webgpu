#import clipping
#import camera
#import light
#import colormap
#import ngsolve/eval/common3d

@group(0) @binding(12) var<storage> vertices : array<f32>;
@group(0) @binding(20) var<uniform> u_mesh : MeshUniforms;
@group(0) @binding(25) var<storage> u_tets : array<u32>;
                        
const TET_FACES = array(
    vec3u(0u, 2u, 1u),
    vec3u(0u, 1u, 3u),
    vec3u(1u, 2u, 3u),
    vec3u(2u, 0u, 3u)
);

const PYRA_FACES = array<vec3<u32>, 6>(
    vec3(0u, 1u, 2u),
    vec3(0u, 2u, 3u),

    vec3(0u, 1u, 4u),
    vec3(1u, 2u, 4u),
    vec3(2u, 3u, 4u),
    vec3(3u, 0u, 4u)
);


const PRISM_FACES = array<vec3<u32>, 8>(
    vec3(0u, 1u, 2u),
    vec3(3u, 5u, 4u),

    vec3(0u, 1u, 4u),
    vec3(0u, 4u, 3u),

    vec3(1u, 2u, 5u),
    vec3(1u, 5u, 4u),

    vec3(2u, 0u, 3u),
    vec3(2u, 3u, 5u)
);

const HEX_FACES = array<vec3<u32>, 12>(
    vec3(0u, 1u, 2u),
    vec3(0u, 2u, 3u),

    vec3(4u, 5u, 6u),
    vec3(4u, 6u, 7u),

    vec3(0u, 3u, 7u),
    vec3(0u, 7u, 4u),

    vec3(1u, 5u, 6u),
    vec3(1u, 6u, 2u),

    vec3(0u, 1u, 5u),
    vec3(0u, 5u, 4u),

    vec3(3u, 2u, 6u),
    vec3(3u, 6u, 7u)
);


struct MeshUniforms {
  subdivision: u32,
  shrink: f32,

  padding0: f32,
  padding1: f32,
};

struct Tet { p: array<u32, 4>, index: u32, padding: u32};

struct MeshFragmentInput {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) n: vec3<f32>,
  @location(2) @interpolate(flat) id: u32,
  @location(3) @interpolate(flat) index: u32,
};

fn calcMeshFace(p: array<vec3<f32>, 3>,
                vertId: u32, nr: u32, index: u32,
                lams: array<vec3<f32>,3>) -> MeshFragmentInput
{
    let n = cross(p[2] - p[0], p[1] - p[0]);
    let point = p[vertId % 3];
    let position = cameraMapPoint(point);
    return MeshFragmentInput(position, point, n, nr, index);
}


/*
@vertex
fn vertex_main_old(@builtin(vertex_index) vertId: u32,
               @builtin(instance_index) elId: u32)
  -> MeshFragmentInput
{
    const N: u32 = 4u;
    let faceId: u32 = vertId / 3u;
    let el = u_tets[elId];
    var p: array<vec3<f32>, 4>;

    var lam = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var center = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < N; i++) {
        let n = 3u * el.p[i];
        p[i] = vec3<f32>(vertices[n], vertices[n + 1u], vertices[n + 2u]);
        center += p[i] / f32(N);
    }


    for (var i = 0u; i < 4u; i++) {
        p[i] = mix(center, p[i], u_mesh.shrink);
        if(calcClipping(p[i]) == false) {
          // one vertex is clipped away, skip rendering this tet
          return MeshFragmentInput(vec4<f32>(0.0, 0.0, 0.0, 0.0),
                                   vec3<f32>(0.0, 0.0, 0.0),
                                   vec3<f32>(0.0, 0.0, 0.0),
                                   0u, 0u);
        }
    }
    var lams = array<vec3<f32>, 4>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 0.0, 0.0),
    );
    let pi = TET_FACES[faceId];
    let points = array<vec3<f32>, 3>(p[pi[0]], p[pi[1]], p[pi[2]]);

    return calcMeshFace(points, vertId, elId, el.index,
                        array<vec3<f32>, 3> (lams[pi[0]], lams[pi[1]], lams[pi[2]]));
}
*/

@vertex
fn vertex_main(@builtin(vertex_index) vertId: u32,
               @builtin(instance_index) elId: u32)
  -> MeshFragmentInput
{
    let face = loadFaces(vertId, elId);
    var lams = array<vec3<f32>, 4>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 0.0, 0.0),
    );

    return calcMeshFace(face.p, vertId, elId, face.index,
                        array<vec3<f32>, 3> (lams[0], lams[1], lams[2]));
}

@fragment
fn fragment_main(input: MeshFragmentInput) -> @location(0) vec4<f32>
{
  checkClipping(input.p);
  let color = getColor(f32(input.index));
  if(color.a < 0.01) {
    discard;
  }
  return lightCalcColor(input.p, input.n, color);
}
