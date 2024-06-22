import numpy as np
from skimage import measure
import trimesh
from pygltflib import GLTF2, Scene, Node, Mesh, Buffer, BufferView, Accessor, Primitive, Material, PbrMetallicRoughness

def niiToGLB(modelData, modelPath):
    # Threshold the data to create a binary mask
    threshold = np.mean(modelData)
    binary_data = modelData > threshold

    # Generate a 3D mesh using marching cubes
    verts, faces, _, _ = measure.marching_cubes(binary_data, level=0)

    # Create a Trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # Export the mesh to GLB
    gltf = GLTF2()

    # Create a buffer for the vertex and index data
    vertex_data = mesh.vertices.astype(np.float32).tobytes()
    index_data = mesh.faces.astype(np.uint32).tobytes()
    buffer = Buffer(byteLength=len(vertex_data) + len(index_data))

    # Add buffer view for vertex data
    vertex_buffer_view = BufferView(buffer=0, byteOffset=0, byteLength=len(vertex_data), target=34962)
    gltf.bufferViews.append(vertex_buffer_view)

    # Add buffer view for index data
    index_buffer_view = BufferView(buffer=0, byteOffset=len(vertex_data), byteLength=len(index_data), target=34963)
    gltf.bufferViews.append(index_buffer_view)

    # Add accessors for vertex and index data
    vertex_accessor = Accessor(bufferView=0, byteOffset=0, componentType=5126, count=len(mesh.vertices), type="VEC3", max=list(mesh.vertices.max(axis=0)), min=list(mesh.vertices.min(axis=0)))
    index_accessor = Accessor(bufferView=1, byteOffset=0, componentType=5125, count=len(mesh.faces) * 3, type="SCALAR")
    gltf.accessors.extend([vertex_accessor, index_accessor])

    # Create a PBR material with full opacity
    pbr = PbrMetallicRoughness(baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=0.0, roughnessFactor=1.0)
    material = Material(pbrMetallicRoughness=pbr, alphaMode='OPAQUE', doubleSided=True) # added doubleside to fix the hollow model
    gltf.materials.append(material)

    # Create a mesh primitive
    primitive = Primitive(attributes={"POSITION": 0}, indices=1, material=0)
    gltf_mesh = Mesh(primitives=[primitive])
    gltf.meshes.append(gltf_mesh)

    # Create a node
    node = Node(mesh=0)
    gltf.nodes.append(node)

    # Create a scene
    scene = Scene(nodes=[0])
    gltf.scenes.append(scene)
    gltf.scene = 0

    # Add buffer data
    gltf.buffers.append(buffer)

    # Assign the buffer data
    gltf.set_binary_blob(vertex_data + index_data)

    # Save GLB file
    gltf.save_binary(modelPath)

    print("conversion of nii to glb succeeded")