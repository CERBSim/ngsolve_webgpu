@group(0) @binding(110) var<storage> mesh : MeshData;
@group(0) @binding(20) var<uniform> u_mesh : MeshUniforms;

struct MeshData {
    offset_vertices: u32,
    offset_2d_data: u32,
    offset_3d_data: u32,
    offset_curvature_2d: u32,
    offset_curvature_3d: u32,
    num_verts: u32,
    num_segments: u32,
    num_trigs: u32,
    num_quads: u32,
    num_tets: u32,
    num_pyramids: u32,
    num_prisms: u32,
    num_hexes: u32,
    is_curved: u32,
    data: array<f32>,
};

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

struct Element {
    p: array<u32, 8>,
    id: u32,
    np: u32,
    index: u32,
}

struct Tetrahedron {
    p: vec4u,
    id: u32,
    tetOfElement: u32,
    index: u32,
}

fn getVertex(vertexId: u32) -> vec3f {
    let offset = mesh.offset_vertices + 3 * vertexId;
    
    return vec3f(
        mesh.data[offset],
        mesh.data[offset + 1],
        mesh.data[offset + 2]
    );
}

fn loadTriangle(instanceId: u32) -> Triangle {
    let MESHDATA_OFFSET : u32 = 2;
    var tri: Triangle;

    var trigId = instanceId;
    
    let offset_2d = mesh.offset_2d_data;
    
    let numElements = bitcast<u32>(mesh.data[offset_2d]);

    let isSecondTrigOfQuad = (trigId >= numElements);
    if(isSecondTrigOfQuad) {
      trigId = trigId - numElements;
      trigId = bitcast<u32>(mesh.data[offset_2d + MESHDATA_OFFSET + numElements*4 + trigId]);
    }

    tri.nr = trigId;

    tri.index = bitcast<u32>(mesh.data[ offset_2d + 4 * trigId + 3 + MESHDATA_OFFSET]);
    let signedIndex = bitcast<i32>(tri.index);
    if(signedIndex < 0) {
      tri.npElement = 4;
      tri.index = bitcast<u32>(mesh.data[offset_2d + u32(-signedIndex) + 1]);
    }
    else {
      tri.npElement = 3;
    }

    tri.trigOfElement = 0;
    let trig_base = offset_2d + MESHDATA_OFFSET + 4u * trigId;

    var vid = vec3u(
          bitcast<u32>(mesh.data[trig_base + 0u]),
          bitcast<u32>(mesh.data[trig_base + 1u]),
          bitcast<u32>(mesh.data[trig_base + 2u])
      );

    if(tri.npElement==4){
        let pi3 = bitcast<u32>(mesh.data[offset_2d + u32(-signedIndex)]);
        if(isSecondTrigOfQuad) {
            tri.trigOfElement = 1;
            vid = vec3u(vid[2], pi3, vid[1]);
        }
        else {
            vid[2] = pi3;
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

fn getElem(elementId: u32) -> Element {
    let MESHDATA_OFFSET: u32 = 5;
    var elem: Element;
    let o3 = mesh.offset_3d_data;

    elem.id = elementId; // element to which I belong ;  is this useful in this case 
    elem.index = bitcast<u32>(mesh.data[o3 + 5 * elementId+ 4 + MESHDATA_OFFSET]); //offset

    let signedIndex = bitcast<i32>(elem.index); //sign of the offset

    var VerArray: array<u32, 8> = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        VerArray[i] = bitcast<u32>(mesh.data[o3 + 5 * elementId+ i + MESHDATA_OFFSET]);
    }

    if (signedIndex >= 0) {
        elem.np = 4;
    }
    else {
         let offset = o3 + u32(-signedIndex);

         elem.np = bitcast<u32>(mesh.data[offset]);
         elem.index = bitcast<u32>(mesh.data[offset + 1u]);

         for (var i: u32 = 0; i < (elem.np-4u); i = i + 1u) {
            VerArray[4+i] = bitcast<u32>(mesh.data[offset + 2 + i]);
            }
    }

    elem.p = VerArray;

    return elem;
}

fn getTetrahedron(tetId: u32) -> Tetrahedron {
    let MESHDATA_OFFSET: u32 = 5;
    var tet: Tetrahedron;
    let o3 = mesh.offset_3d_data;
    
    let n_els = bitcast<u32>(mesh.data[o3]);
    
    var elId = tetId;
    
    tet.tetOfElement = 0;
    
    if(elId >= n_els) {
        let num_pyramids = bitcast<u32>(mesh.data[o3 + 2]);
        let num_prisms = bitcast<u32>(mesh.data[o3 + 3]);
        let hex_start = num_pyramids + 2u * num_prisms;
        
        var extraElemId = elId-n_els;
        
        if(extraElemId < num_pyramids) {
            tet.tetOfElement = 1;
        } 
        else if (extraElemId < hex_start) {
            extraElemId = extraElemId - num_pyramids;
            tet.tetOfElement = extraElemId%2u + 1;
            extraElemId = extraElemId/2u + num_pyramids;
        }
        else {
            extraElemId = extraElemId - hex_start;
            tet.tetOfElement = extraElemId%5u + 1;
            extraElemId = extraElemId/5u + num_pyramids + num_prisms;
        }
        
        elId = bitcast<u32>(mesh.data[o3 + MESHDATA_OFFSET + 5u*n_els + extraElemId]);
        
        // tet.tetOfElement = 0;
    }
    
    let element = getElem(elId);
    tet.index = element.index;
    tet.id = element.id;
    
    var pi = vec4u(0,1,2,3);
    if(element.np == 5 ) {
        pi = PYRAMID_TETS[tet.tetOfElement];
    }
    else if(element.np == 6 ) {
        pi = PRISM_TETS[tet.tetOfElement];
    }
    else if (element.np == 8) {
        pi = HEX_TETS[tet.tetOfElement];
    }
    
    
    tet.p = vec4u( element.p[pi[0]], element.p[pi[1]], element.p[pi[2]], element.p[pi[3]] );
    return tet;
}

fn getElementVertex(element: Element, index: u32) ->  vec3<f32> {
    return getVertex(element.p[index]);
}

fn getCenter(element: Element) -> vec3<f32> {
    var center = getElementVertex(element, 0u);

    for (var i: u32 = 1u; i < element.np; i = i + 1u) {
        center = center + getElementVertex(element, i);
    }

    return center / f32(element.np);
}

fn getFace(element: Element, faceIndex: u32) -> Triangle {
    let MESHDATA_OFFSET: u32 = 5;
    var face: Triangle;

    face.nr = element.id; // element to which I belong
    //face.index = u_tets[5 * element.id + 4 + MESHDATA_OFFSET]; //offset
    face.trigOfElement = faceIndex;
    face.npElement = element.np;
    face.index = element.index;
  
    var LocalV: vec3<u32> = vec3<u32>(0, 0, 0); 

    if (face.npElement == 4) {
        LocalV = TET_FACES[face.trigOfElement];
    } else if (face.npElement == 5) {
        LocalV = PYRA_FACES[face.trigOfElement];
    } else if (face.npElement == 6) {
        LocalV = PRISM_FACES[face.trigOfElement];
    } else {
        LocalV = HEX_FACES[face.trigOfElement];
    }

    face.p = array<vec3<f32>, 3>(
        getElementVertex(element, LocalV[0]),
        getElementVertex(element, LocalV[1]),
        getElementVertex(element, LocalV[2])
    );

    return face;
}

const PYRAMID_TETS = array(
    vec4u(1, 3 ,4, 0),
    vec4u(1, 3 ,4, 2),
);

const PRISM_TETS = array(
    vec4u( 0,1,5,2),
    vec4u( 5,1,4,0),
    vec4u( 3,4,0,5),
);

const  HEX_TETS = array(
    vec4u( 1,3,4,0),
    vec4u( 7,6,2,5),
    vec4u( 5,7,1,4),
    vec4u( 7,2,3,1),
    vec4u( 4,3,7,1),
    vec4u( 5,2,7,1),
);

const TET_FACES = array(
    vec3u(0u, 2u, 1u),
    vec3u(0u, 1u, 3u),
    vec3u(1u, 2u, 3u),
    vec3u(2u, 0u, 3u)
);

const PYRA_FACES = array<vec3<u32>, 6>(
    //vec3(0u, 1u, 2u),
    vec3(0u, 2u, 1u),
    vec3(0u, 3u, 2u),

    vec3(0u, 1u, 4u),
    vec3(1u, 2u, 4u),
    vec3(2u, 3u, 4u),
    vec3(3u, 0u, 4u)
);


const PRISM_FACES = array<vec3<u32>, 8>(
    vec3(0u, 2u, 1u),
    vec3(3u, 4u, 5u),

    vec3(0u, 1u, 4u),
    vec3(0u, 4u, 3u),

    vec3(1u, 2u, 5u),
    vec3(1u, 5u, 4u),

    vec3(2u, 0u, 3u),
    vec3(2u, 3u, 5u)
);

const HEX_FACES = array<vec3<u32>, 12>(
    vec3(0u, 2u, 1u),
    vec3(3u, 2u, 0u),

    //BOTTOM FACE
    vec3(4u, 5u, 6u),
    vec3(4u, 6u, 7u),

    //Left lateral face
    vec3(0u, 7u, 3u),
    vec3(0u, 4u, 7u),

    //Right Face
    vec3(1u, 6u, 5u),
    vec3(1u, 2u, 6u),

    //First Face
    vec3(0u, 1u, 5u),
    vec3(0u, 5u, 4u),

    // 4th Lateral Face
    vec3(3u, 6u, 2u),
    vec3(3u, 7u, 6u)
);
