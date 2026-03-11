#import clipping
#import camera
#import light
#import colormap
#import ngsolve/eval/common3d
#import ngsolve/mesh/render

// @group(0) @binding(12) var<storage> vertices : array<f32>;
// @group(0) @binding(25) var<storage> u_tets : array<u32>;
                        

// struct Tet { p: array<u32, 4>, index: u32, padding: u32};

fn calcMeshFace(p: array<vec3<f32>, 3>,
                vertId: u32, nr: u32, index: u32,
                lams: array<vec3<f32>,3>) -> MeshFragmentInput
{
    let n = cross(p[2] - p[0], p[1] - p[0]);
    let point = p[vertId % 3];
    let position = cameraMapPoint(point);
    return MeshFragmentInput(position, point, n, nr, index);
}

@vertex
fn vertex_main(@builtin(vertex_index) vertId: u32,
               @builtin(instance_index) instanceId: u32)
  -> MeshFragmentInput
{
    let MESHDATA_OFFSET: u32 = 5;
    let offset_3d = bitcast<u32>(mesh.offset_3d_data);
    let numElements = bitcast<u32>(mesh.data[offset_3d + 0]);
    let numTets = bitcast<u32>(mesh.data[offset_3d + 1]);
    let numPyra = bitcast<u32>(mesh.data[offset_3d + 2]);
    let numPrims = bitcast<u32>(mesh.data[offset_3d + 3]);
    let numHex = bitcast<u32>(mesh.data[offset_3d + 4]);
    let INDEX_SORTED_BY_TYPE = MESHDATA_OFFSET+5*numElements;

    var elementId= instanceId;
    var faceId= 0u;

    let pyramidStart = 4u*numElements;
    let prismStart = pyramidStart + 2u *numPyra;
    let hexStart = prismStart + 4u * numPrims;

    if (elementId < pyramidStart) {
        faceId = elementId % 4u;
        elementId= elementId / 4u;
    }
    // extra 2 triangles of any pyramid
    else if (elementId< prismStart) {
        let local = elementId - pyramidStart;
        faceId = local % 2u + 4u;
        elementId= bitcast<u32>(mesh.data[offset_3d + INDEX_SORTED_BY_TYPE + (local / 2u)]);
        
    }

    // extra 4 triangles of any prism
    else if (elementId < hexStart) {
        let local = elementId - prismStart;
        faceId = local % 4u + 4u;
        elementId= bitcast<u32>(mesh.data[offset_3d + INDEX_SORTED_BY_TYPE + numPyra + (local / 4u)]);
    }

    // extra 8 triangles of any hex
    else {
        let local = elementId - hexStart;
        faceId = local % 8u + 4u;
        elementId= bitcast<u32>(mesh.data[offset_3d + INDEX_SORTED_BY_TYPE + numPyra + numPrims + (local / 8u)]);
    }

    let element = getElem(elementId);
    let center = getCenter(element);
    var face = getFace(element, faceId);

    if(calcClipping(center) == false) {
          // one vertex is clipped away, skip rendering this tet
          return MeshFragmentInput(vec4<f32>(0.0, 0.0, 0.0, 0.0),
                                   vec3<f32>(0.0, 0.0, 0.0),
                                   vec3<f32>(0.0, 0.0, 0.0),
                                   0u, 0u);
    }

    for (var i = 0u; i < 3u; i++) {
        face.p[i] = mix(center, face.p[i], u_mesh.shrink);
    }
    
    var lams = array<vec3<f32>, 4>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 0.0, 0.0),
    );


    return calcMeshFace(face.p, vertId, instanceId, face.index,
                        array<vec3<f32>, 3> (lams[0], lams[1], lams[2]));
}

@fragment
fn fragment_main(input: MeshFragmentInput) -> @location(0) vec4<f32>
{
  //checkClipping(input.p);
  let color = getColor(f32(input.index));
  if(color.a < 0.01) {
    discard;
  }
  return lightCalcColor(input.p, input.n, color);
}


