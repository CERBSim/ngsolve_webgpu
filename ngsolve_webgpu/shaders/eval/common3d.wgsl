

struct Triangle {
  p: array<vec3<f32>, 3>,
  nr: u32,
  npElement: u32,
  trigOfElement: u32,
  index: u32,
};

fn loadFaces(vertexId: u32, instanceId: u32) -> Triangle {
    let MESHDATA_OFFSET: u32 = 5;
    var face: Triangle;

    var elementId= instanceId;
    let numElements = u_tets[0];
    let numTets = u_tets[1];
    let numPyra = u_tets[2];
    let numPrims = u_tets[3];
    let numHex = u_tets[4];
    
    let INDEX_SORTED_BY_TYPE = MESHDATA_OFFSET+4*numTets+6*numPyra+8*numPrims+12*numHex;

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