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

fn getElem(elementId: u32) -> Element {
    let MESHDATA_OFFSET: u32 = 5;
    var elem: Element;

    elem.id = elementId; // element to which I belong ;  is this useful in this case 
    elem.index = u_tets[5 * elementId+ 4 + MESHDATA_OFFSET]; //offset

    let signedIndex = bitcast<i32>(elem.index); //sign of the offset

    var VerArray: array<u32, 8> = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        VerArray[i] = u_tets[5 * elementId+ i + MESHDATA_OFFSET];
    }

    if (signedIndex >= 0) {
        elem.np = 4;
    }
    else {
         let offset = u32(-signedIndex);

         elem.np = u_tets[offset];
         elem.index = u_tets[offset + 1u];

         for (var i: u32 = 0; i < (elem.np-4u); i = i + 1u) {
            VerArray[4+i] = u_tets[offset + 2 + i];
            }
    }

    elem.p = VerArray;

    return elem;
}

fn getTetrahedron(tetId: u32) -> Tetrahedron {
    let MESHDATA_OFFSET: u32 = 5;
    var tet: Tetrahedron;
    
    let n_els = u_tets[0];
    
    var elId = tetId;
    
    tet.tetOfElement = 0;
    
    if(elId >= n_els) {
        let num_pyramids = u_tets[2];
        let num_prisms = u_tets[3];
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
        
        elId = u_tets[MESHDATA_OFFSET + 5u*n_els + extraElemId];
        
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

fn getPoint(element: Element, index: u32) ->  vec3<f32> {
    let a = 3u * element.p[index];
    return vec3<f32>(vertices[a ], vertices[a + 1], vertices[a + 2]);

}

fn getVertex(index: u32) ->  vec3<f32> {
    let a = 3u * index;
    return vec3<f32>(vertices[a ], vertices[a + 1], vertices[a + 2]);

}

fn getCenter(element: Element) -> vec3<f32> {
    var center = getPoint(element, 0u);

    for (var i: u32 = 1u; i < element.np; i = i + 1u) {
        center = center + getPoint(element, i);
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
        getPoint(element, LocalV[0]),
        getPoint(element, LocalV[1]),
        getPoint(element, LocalV[2])
    );

    return face;
}

/*
fn loadFaces_old(vertexId: u32, instanceId: u32) -> Triangle {
    let MESHDATA_OFFSET: u32 = 5;
    var face: Triangle;

    var elementId= instanceId;
    let numElements = u_tets[0];
    let numTets = u_tets[1];
    let numPyra = u_tets[2];
    let numPrims = u_tets[3];
    let numHex = u_tets[4];

    let INDEX_SORTED_BY_TYPE = MESHDATA_OFFSET+5*numElements;

    // Standard elements
    if (elementId< 4u * numElements) {
        face.trigOfElement = elementId% 4u;
        elementId= elementId/ 4u;
    }

    // extra 2 triangles of any pyramid
    else if (4u * numElements <= elementId&& elementId< 4u * numElements + 2u * numPyra) {
        let local = elementId- 4u * numElements;
        face.trigOfElement = local % 2u + 4u;
        // elementId= local / 2u;
        elementId= u_tets[INDEX_SORTED_BY_TYPE + (local / 2u)];
        
    }

    // extra 4 triangles of any prism
    else if (4u * numElements + 2u * numPyra <= elementId&& elementId< 4u * numElements + 2u * numPyra + 4u * numPrims) {
        let local = elementId- (4u * numElements + 2u * numPyra);
        face.trigOfElement = local % 4u + 4u;
        //elementId= local / 4u;
        elementId= u_tets[INDEX_SORTED_BY_TYPE + (local / 4u)];
    }

    // extra 8 triangles of any hex
    else {
        let local = elementId- (4u * numElements + 2u * numPyra + 4u * numPrims);
        face.trigOfElement = local % 8u + 4u;
        //elementId= local / 8u;
        elementId= u_tets[INDEX_SORTED_BY_TYPE + (local / 8u)];
    }

    face.nr = elementId; // element to which I belong
    face.index = u_tets[5 * elementId+ 4 + MESHDATA_OFFSET]; //offset

    let signedIndex = bitcast<i32>(face.index); //sign of the offset

    // Defining the array (BAD TYPE
    var VerArray: array<u32, 8> = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        VerArray[i] = u_tets[5 * elementId+ i + MESHDATA_OFFSET];
    }

    if (signedIndex >= 0) {
        face.npElement = 4;
    }
    else {
         let offset = u32(-signedIndex);

         face.npElement = u_tets[offset];
         face.index = u_tets[offset + 1u];

         for (var i: u32 = 0; i < (face.npElement-4u); i = i + 1u) {
            VerArray[4+i] = u_tets[offset + 2 + i];
            }
    }
  
    var LocalV: vec3<u32> = vec3<u32>(0, 0, 0); 

    // Element to which I belong : face.nr
    // The type of element it is : face.npElement
    // which triangle am I : face.TrigofElement

    if (face.npElement == 4) {
        LocalV = TET_FACES[face.trigOfElement];
    } else if (face.npElement == 5) {
        LocalV = PYRA_FACES[face.trigOfElement];
    } else if (face.npElement == 6) {
        LocalV = PRISM_FACES[face.trigOfElement];
    } else {
        LocalV = HEX_FACES[face.trigOfElement];
    }

    var vid = vec3u(0,0,0);

    vid = vec3u(VerArray[LocalV[0]],
                VerArray[LocalV[1]],
                VerArray[LocalV[2]]
    );

    vid = 3 * vid; 

    face.p = array<vec3<f32>, 3>(
        vec3<f32>(vertices[vid[0] ], vertices[vid[0] + 1], vertices[vid[0] + 2]),
        vec3<f32>(vertices[vid[1] ], vertices[vid[1] + 1], vertices[vid[1] + 2]),
        vec3<f32>(vertices[vid[2] ], vertices[vid[2] + 1], vertices[vid[2] + 2])
    );

    return face;
}
*/


const PYRAMID_TETS = array(
    vec4u(0, 1, 2, 4),
    vec4u(0, 2, 3, 4),
);

const PRISM_TETS = array(
    vec4u(0, 1, 2, 3),
    vec4u(1, 4, 5, 3),
    vec4u(1, 5, 2, 3),
);

// const PRISM_TETS = array(
//     vec4u(0, 0, 0, 0),
//     vec4u(0, 0, 0, 0),
//     vec4u(0, 1, 2, 3),
//     // vec4u(1, 5, 2, 3),
//     // vec4u(1, 4, 5, 3),
// );

const  HEX_TETS = array(
    vec4u( 0,2,7,3),
    vec4u( 6,5,1,4),
    vec4u( 4,6,0,7),
    vec4u( 6,1,2,0),
    vec4u( 7,2,6,0),
    vec4u( 4,1,6,0),
);