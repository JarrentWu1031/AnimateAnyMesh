import bpy
import bmesh
import os
import numpy as np
import torch
import math
import mathutils
from mathutils import Vector
import imageio
import imageio.v3 as iio
import glob

### Render DMesh with no gt texture
def create_mesh(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    for v in verts:
        bm.verts.new(v)
    bm.verts.ensure_lookup_table()
    for f in faces:
        try:
            bm.faces.new([bm.verts[i] for i in f])
        except:
            continue
    bm.to_mesh(mesh)
    mesh.update()
    return obj

def create_gradient_material():
    mat = bpy.data.materials.new(name="GradientMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    # Create nodes
    node_tex_coord = nodes.new(type='ShaderNodeTexCoord')
    node_mapping = nodes.new(type='ShaderNodeMapping')
    node_gradient = nodes.new(type='ShaderNodeTexGradient')
    node_color_ramp = nodes.new(type='ShaderNodeValToRGB')
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    # Set ColorRamp nodes
    color_ramp = node_color_ramp.color_ramp
    color_ramp.elements[0].position = 0.0
    color_ramp.elements[0].color = (1, 0, 0, 1)  # 红色
    color_ramp.elements[1].position = 1.0
    color_ramp.elements[1].color = (0, 0, 1, 1)  # 蓝色
    # Link nodes
    links.new(node_tex_coord.outputs['Generated'], node_mapping.inputs['Vector'])
    links.new(node_mapping.outputs['Vector'], node_gradient.inputs['Vector'])
    links.new(node_gradient.outputs['Color'], node_color_ramp.inputs['Fac'])
    links.new(node_color_ramp.outputs['Color'], node_bsdf.inputs['Base Color'])
    links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])
    return mat

def setup_lighting():
    # Delete scene light
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj) 
    # Create lights
    key_light = bpy.data.lights.new(name="Key_Light", type='SUN')
    key_light.energy = 3.0 
    key_light.angle = 0.5  
    key_light_obj = bpy.data.objects.new(name="Key_Light", object_data=key_light)
    bpy.context.collection.objects.link(key_light_obj)
    fill_light = bpy.data.lights.new(name="Fill_Light", type='AREA')
    fill_light.energy = 1.5  
    fill_light.size = 5.0  
    fill_light_obj = bpy.data.objects.new(name="Fill_Light", object_data=fill_light)
    bpy.context.collection.objects.link(fill_light_obj)
    rim_light = bpy.data.lights.new(name="Rim_Light", type='SPOT')
    rim_light.energy = 2.0
    rim_light.spot_size = math.radians(45)  
    rim_light.spot_blend = 0.5  
    rim_light_obj = bpy.data.objects.new(name="Rim_Light", object_data=rim_light)
    bpy.context.collection.objects.link(rim_light_obj)
    ambient_light = bpy.data.lights.new(name="Ambient_Light", type='AREA')
    ambient_light.energy = 0.5  
    ambient_light.size = 10.0
    ambient_light_obj = bpy.data.objects.new(name="Ambient_Light", object_data=ambient_light)
    bpy.context.collection.objects.link(ambient_light_obj)
    return key_light_obj, fill_light_obj, rim_light_obj, ambient_light_obj

def setup_camera_and_light_to_focus_object(camera, vertex_trajectories, frame_percentage=1.0, size=None, center=None, azi=0, ele=0):
    if size is None:
        center = np.mean(vertex_trajectories, axis=(0, 1))
        min_coords = np.min(vertex_trajectories, axis=(0, 1))
        max_coords = np.max(vertex_trajectories, axis=(0, 1))
        size = np.max(max_coords - min_coords)
    # Set camera
    fov = camera.data.angle
    distance = size / (2 * math.tan(fov / 2) * frame_percentage)
    base_offset = mathutils.Vector((0, -distance, distance / 2))
    azi_rad = math.radians(azi)
    ele_rad = math.radians(ele)
    rotation = mathutils.Euler((math.radians(ele), 0, math.radians(azi)), 'ZYX')
    final_offset = rotation.to_matrix() @ base_offset
    camera.location = mathutils.Vector(center) + final_offset
    direction = mathutils.Vector(center) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    # Set light
    key_light, fill_light, rim_light, ambient_light = setup_lighting()
    key_light.location = camera.location + mathutils.Vector((0, 0, distance/2))
    key_direction = mathutils.Vector(center) - key_light.location
    key_light.rotation_euler = key_direction.to_track_quat('-Z', 'Y').to_euler()
    fill_offset = mathutils.Vector((math.cos(azi_rad + math.pi/2), -math.sin(azi_rad + math.pi/2), 0))
    fill_light.location = camera.location + fill_offset * distance
    fill_direction = mathutils.Vector(center) - fill_light.location
    fill_light.rotation_euler = fill_direction.to_track_quat('-Z', 'Y').to_euler()
    rim_x = -distance * math.cos(ele_rad) * math.sin(azi_rad)
    rim_y = distance * math.cos(ele_rad) * math.cos(azi_rad)
    rim_light.location = mathutils.Vector(center) + mathutils.Vector((rim_x, rim_y, distance))
    rim_direction = mathutils.Vector(center) - rim_light.location
    rim_light.rotation_euler = rim_direction.to_track_quat('-Z', 'Y').to_euler()
    ambient_light.location = mathutils.Vector(center) + mathutils.Vector((0, 0, distance*1.5))
    ambient_direction = mathutils.Vector(center) - ambient_light.location
    ambient_light.rotation_euler = ambient_direction.to_track_quat('-Z', 'Y').to_euler()
    # Set global light
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.1, 0.1, 0.1, 1)
    bg.inputs[1].default_value = 0.2

def render_dynamic_mesh_direct_to_video(vertices, face_data, video_save_dir, save_name, azi=0, ele=0):
    # Clear scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    if "LookAtTarget" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["LookAtTarget"], do_unlink=True)
    # Load mesh data
    if hasattr(vertices, 'numpy'): vertices = vertices.numpy()
    if hasattr(face_data, 'numpy'): face_data = face_data.numpy()  
    vertex_trajectories = vertices
    # Create mesh & material
    initial_verts = vertex_trajectories[0]
    mesh_obj = create_mesh("DynamicMesh", initial_verts, face_data)
    gradient_mat = create_gradient_material()
    mesh_obj.data.materials.append(gradient_mat)
    # Set scene
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = len(vertex_trajectories) - 1
    # Set camera
    cam = bpy.data.objects['Camera']
    setup_camera_and_light_to_focus_object(cam, vertex_trajectories, azi=azi, ele=ele)
    # Create shape key animation 
    mesh_obj.shape_key_add(name='Basis')
    if mesh_obj.data.shape_keys.animation_data is None:
        mesh_obj.data.shape_keys.animation_data_create()
    for frame_idx in range(len(vertex_trajectories)):
        shape_key = mesh_obj.shape_key_add(name=f'Frame_{frame_idx}')
        shape_key.data.foreach_set('co', vertex_trajectories[frame_idx].flatten())
        shape_key.value = 0
        shape_key.keyframe_insert(data_path='value', frame=frame_idx - 1 if frame_idx > 0 else 0)
        shape_key.value = 1
        shape_key.keyframe_insert(data_path='value', frame=frame_idx)
        shape_key.value = 0
        shape_key.keyframe_insert(data_path='value', frame=frame_idx + 1)
        fcurve = mesh_obj.data.shape_keys.animation_data.action.fcurves.find(f'key_blocks["{shape_key.name}"].value')
        if fcurve:
            for kf_point in fcurve.keyframe_points:
                kf_point.interpolation = 'CONSTANT'
    # Set scene and camera
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = len(vertex_trajectories) - 1
    # Render settings
    scene.render.engine = 'CYCLES'
    scene.render.fps = 10
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.cycles.samples = 128
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    scene.cycles.device = "GPU"
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    # scene.render.image_settings.color_mode = 'RGB'
    scene.sequencer_colorspace_settings.name = 'sRGB'
    video_path = os.path.join(video_save_dir, f"{save_name}.mp4")
    scene.render.filepath = video_path
    # Start rendering
    print(f"Start rendering (10 FPS), the rendered results are saved to : {video_path}")
    bpy.ops.render.render(animation=True)
    print("Rendering finished!!!")
###

### Render DMesh with gt texture
def clear_scene():
    bpy.ops.wm.read_homefile(use_empty=True)
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

def import_glb(filepath):
    bpy.ops.import_scene.gltf(filepath=filepath)
    mesh_objects = [obj for obj in bpy.context.scene.objects if (obj.type == 'MESH' and obj.name != 'Cube' and obj.name != 'Icosphere')]
    if not mesh_objects:
        raise ValueError("No mesh objects found")
    for obj in mesh_objects:
        if obj.animation_data:
            obj.animation_data_clear()
        if obj.data.shape_keys:
            obj.shape_key_clear()
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    print(f"Loaded {len(mesh_objects)} mesh objects")
    for mesh_obj in mesh_objects:
        print(f"Mesh: {mesh_obj.name}, Vertices: {len(mesh_obj.data.vertices)}")
    return mesh_objects

def setup_scene_four_views(mesh_obj):
    """Four orthogonal view renderings"""
    cameras = []
    camera_positions = [
        ((0, -3, 0), (1.5708, 0, 0)),       # frontal
        ((3, 0, 0), (1.5708, 0, 1.5708)),    # right
        ((0, 3, 0), (1.5708, 0, 3.14159)),  # back
        ((-3, 0, 0), (1.5708, 0, -1.5708)), # left
    ]
    for pos, rot in camera_positions:
        bpy.ops.object.camera_add(location=pos)
        camera = bpy.context.object
        camera.rotation_euler = rot
        camera.data.type = 'ORTHO'  
        camera.data.ortho_scale = 3.0
        cameras.append(camera)
    # Lights
    light_settings = [
        ((0, -3, 2), (0.5, 0, 0), 5.0),      
        ((0, 3, 2), (-0.5, 0, math.pi), 5.0), 
        ((-3, 0, 2), (0.5, 0, math.pi/2), 5.0), 
        ((3, 0, 2), (0.5, 0, -math.pi/2), 5.0), 
        ((2, -2, 3), (0.7, 0.7, 0), 3.0),
        ((-2, -2, 3), (0.7, -0.7, 0), 3.0),
        ((2, 2, 3), (-0.7, 0.7, 0), 3.0),
        ((-2, 2, 3), (-0.7, -0.7, 0), 3.0),
    ]
    lights = []
    for pos, rot, energy in light_settings:
        bpy.ops.object.light_add(type='SUN', location=pos)
        light = bpy.context.object
        light.rotation_euler = rot
        light.data.energy = energy
        light.data.use_shadow = True
        light.data.shadow_soft_size = 2.0  # 增加软阴影大小
        lights.append(light)
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
    top_light = bpy.context.object
    top_light.rotation_euler = (0, 0, 0)
    top_light.data.energy = 7.0
    top_light.data.use_shadow = True
    top_light.data.shadow_soft_size = 2.5
    lights.append(top_light)
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    bg_node = nodes.new('ShaderNodeBackground')
    bg_node.inputs['Color'].default_value = (1, 1, 1, 1)
    bg_node.inputs['Strength'].default_value = 2.0  
    output_node = nodes.new('ShaderNodeOutputWorld')
    world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    # Rendering settings
    bpy.context.scene.render.engine = 'CYCLES'  
    bpy.context.scene.cycles.shadow_samples = 2  
    bpy.context.scene.cycles.light_sampling_threshold = 0.01  
    bpy.context.scene.cycles.samples = 64  
    bpy.context.scene.cycles.max_bounces = 1  
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transmission_bounces = 1
    bpy.context.scene.cycles.volume_bounces = 0
    bpy.context.scene.cycles.transparent_max_bounces = 1
    for obj in cameras + lights:
        obj.lock_location = (True, True, True)
        obj.lock_rotation = (True, True, True)
        obj.lock_scale = (True, True, True)
    return cameras

def setup_scene(mesh_obj, azi=0.0, ele=0.0):
    base_location = Vector((0, -3.0, 0))
    rotation = mathutils.Euler((math.radians(ele), 0, math.radians(azi)), 'ZYX')
    final_location = rotation.to_matrix() @ base_location
    cameras = []
    bpy.ops.object.camera_add(location=final_location)
    camera = bpy.context.object
    direction = Vector((0, 0, 0)) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y') # -Z是相机的朝向，Y是相机的上方向
    camera.rotation_euler = rot_quat.to_euler()
    # 使用原始代码中的相机设置。
    camera.data.type = 'ORTHO'  
    camera.data.ortho_scale = 3.0 # ortho_scale 保持不变
    cameras.append(camera)
    light_settings = [
        ((0, -3, 2), (0.5, 0, 0), 5.0),      
        ((0, 3, 2), (-0.5, 0, math.pi), 5.0), 
        ((-3, 0, 2), (0.5, 0, math.pi/2), 5.0), 
        ((3, 0, 2), (0.5, 0, -math.pi/2), 5.0), 
        ((2, -2, 3), (0.7, 0.7, 0), 3.0),
        ((-2, -2, 3), (0.7, -0.7, 0), 3.0),
        ((2, 2, 3), (-0.7, 0.7, 0), 3.0),
        ((-2, 2, 3), (-0.7, -0.7, 0), 3.0),
    ]
    lights = []
    for pos, rot, energy in light_settings:
        bpy.ops.object.light_add(type='SUN', location=pos)
        light = bpy.context.object
        light.rotation_euler = rot
        light.data.energy = energy
        light.data.use_shadow = True
        light.data.shadow_soft_size = 2.0  
        lights.append(light)
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
    top_light = bpy.context.object
    top_light.rotation_euler = (0, 0, 0)
    top_light.data.energy = 7.0
    top_light.data.use_shadow = True
    top_light.data.shadow_soft_size = 2.5
    lights.append(top_light)
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    bg_node = nodes.new('ShaderNodeBackground')
    bg_node.inputs['Color'].default_value = (1, 1, 1, 1)
    bg_node.inputs['Strength'].default_value = 2.0  
    output_node = nodes.new('ShaderNodeOutputWorld')
    world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    bpy.context.scene.render.engine = 'CYCLES' 
    bpy.context.scene.cycles.shadow_samples = 2 
    bpy.context.scene.cycles.light_sampling_threshold = 0.01  
    bpy.context.scene.cycles.samples = 64  
    bpy.context.scene.cycles.max_bounces = 1 
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transmission_bounces = 1
    bpy.context.scene.cycles.volume_bounces = 0
    bpy.context.scene.cycles.transparent_max_bounces = 1
    for obj in cameras + lights:
        obj.lock_location = (True, True, True)
        obj.lock_rotation = (True, True, True)
        obj.lock_scale = (True, True, True)
    return cameras

def setup_circle_camera(mesh_obj, frame_idx, total_frames=64):
    """Set surrounding cameras"""
    angle = (360.0 / total_frames) * frame_idx - 90.0
    angle_rad = math.radians(angle)
    radius = 3.0  
    x = radius * math.cos(angle_rad)
    y = radius * math.sin(angle_rad)
    z = 0.0  
    bpy.ops.object.camera_add(location=(x, y, z))
    camera = bpy.context.object
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = mesh_obj
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 3.0
    return camera

def set_transparent_background():
    bpy.context.scene.render.film_transparent = True
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    bg_node = nodes.new('ShaderNodeBackground')
    bg_node.inputs['Color'].default_value = (1, 1, 1, 0) 
    bg_node.inputs['Strength'].default_value = 0.0
    output_node = nodes.new('ShaderNodeOutputWorld')
    world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'

def setup_animation(output_path, num_frames=16):
    """Rendering settings"""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.render.filepath = output_path
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.fps = 10
    scene.frame_start = 0
    scene.frame_end = num_frames - 1
    scene.cycles.samples = 128
    scene.cycles.preview_samples = 32

def get_mesh_vertices(mesh_obj):
    vertices = []
    for vertex in mesh_obj.data.vertices:
        vertices.append([vertex.co[0], vertex.co[1], vertex.co[2]])
    return np.array(vertices)

def get_all_vertices(mesh_objects):
    all_vertices = []
    for mesh_obj in mesh_objects:
        vertices = get_mesh_vertices(mesh_obj)
        all_vertices.append(torch.from_numpy(vertices))
    return all_vertices

def get_mesh_faces(mesh_obj):
    faces = []
    for face in mesh_obj.data.polygons:
        faces.append(list(face.vertices))
    return np.array(faces)

def get_all_faces(mesh_objects):
    all_faces = []
    vertex_count = 0  
    for mesh_obj in mesh_objects:
        faces = get_mesh_faces(mesh_obj)
        if len(all_faces) > 0:
            faces = faces + vertex_count
        all_faces.append(torch.from_numpy(faces))
        vertex_count += len(mesh_obj.data.vertices)
    return all_faces

def move_vertices_with_trajectory(mesh_obj, frame, trajectories):
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='OBJECT')
    if mesh_obj.data.shape_keys is None:
        basis = mesh_obj.shape_key_add(name='Basis')
        basis.interpolation = 'KEY_LINEAR'  
    shape_key_name = f"Frame_{frame}"
    shape_key = mesh_obj.data.shape_keys.key_blocks.get(shape_key_name)
    if not shape_key:
        shape_key = mesh_obj.shape_key_add(name=shape_key_name)
        shape_key.interpolation = 'KEY_LINEAR'
    positions = trajectories[frame].numpy()
    for idx, pos in enumerate(positions):
        shape_key.data[idx].co = Vector(pos)

def drive_mesh_with_trajs_frames(mesh_objects, trajs, output_dir, azi=0.0, ele=0.0):
    try:
        os.makedirs(output_dir, exist_ok=True)
        view_names = [f'azi{int(azi)}_ele{int(ele)}']  
        frames_dirs = {}
        for view in view_names:
            frames_dir = os.path.join(output_dir, f'frames_azi{int(azi)}_ele{int(ele)}')
            os.makedirs(frames_dir, exist_ok=True)
            frames_dirs[view] = frames_dir
        print("Setting up animations for all meshes...")
        for i in range(len(mesh_objects)):
            mesh_obj = mesh_objects[i]
            traj = trajs[i]
            num_frames = traj.shape[0]
            for frame in range(num_frames):
                move_vertices_with_trajectory(mesh_obj, frame, traj)
            for frame in range(num_frames):
                for shape_key in mesh_obj.data.shape_keys.key_blocks[1:]:
                    shape_key.value = 0
                    shape_key.keyframe_insert("value", frame=frame)
                    if mesh_obj.data.shape_keys.animation_data and mesh_obj.data.shape_keys.animation_data.action:
                        fcurves = mesh_obj.data.shape_keys.animation_data.action.fcurves
                        for fc in fcurves:
                            for kf in fc.keyframe_points:
                                kf.interpolation = 'CONSTANT'
                current_shape_key = mesh_obj.data.shape_keys.key_blocks[f"Frame_{frame}"]
                current_shape_key.value = 1
                current_shape_key.keyframe_insert("value", frame=frame)
        print("Setting up scene...")
        cameras = setup_scene(mesh_objects[0], azi=azi, ele=ele)
        set_transparent_background()
        print("Setting up rendering...")
        scene = bpy.context.scene
        scene.render.image_settings.file_format = 'PNG'
        scene.render.resolution_x = 512
        scene.render.resolution_y = 512
        scene.render.resolution_percentage = 100
        scene.render.fps = 10
        scene.frame_start = 0
        scene.frame_end = num_frames - 1
        print("Rendering frames...")
        for frame in range(num_frames):
            scene.frame_set(frame)
            for cam, view_name in zip(cameras, view_names):
                scene.camera = cam
                frames_dir = frames_dirs[view_name]
                scene.render.filepath = os.path.join(frames_dir, f'frame_{frame:04d}.png')
                bpy.ops.render.render(write_still=True)
                print(f"Rendered {view_name} frame {frame}/{num_frames-1}")
        filename = output_dir.split('/')[-1]
        print("Converting frames to videos...")
        frames = []
        for view_name, frames_dir in frames_dirs.items():
            frame_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))
            if frame_files:
                for frame_file in frame_files:
                    img = iio.imread(frame_file)
                    frames.append(img)
            video_path = os.path.join(os.path.dirname(output_dir), '{}_azi{}_ele{}.mp4'.format(filename, int(azi), int(ele)))
            iio.imwrite(
                video_path,
                frames,
                fps=10,
                codec='libx264',
                quality=10,
                pixelformat='yuv420p'
            )
        print(f"All frames rendered successfully to: {output_dir}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

def drive_mesh_with_trajs_frames_five_views(mesh_objects, trajs, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        view_names = ['front', 'right', 'back', 'left', 'circle']  
        frames_dirs = {}
        for view in view_names:
            frames_dir = os.path.join(output_dir, f'frames_{view}')
            os.makedirs(frames_dir, exist_ok=True)
            frames_dirs[view] = frames_dir
        print("Setting up animations for all meshes...")
        for i in range(len(mesh_objects)):
            mesh_obj = mesh_objects[i]
            traj = trajs[i]
            num_frames = traj.shape[0]
            for frame in range(num_frames):
                move_vertices_with_trajectory(mesh_obj, frame, traj)
            for frame in range(num_frames):
                for shape_key in mesh_obj.data.shape_keys.key_blocks[1:]:
                    shape_key.value = 0
                    shape_key.keyframe_insert("value", frame=frame)
                    if mesh_obj.data.shape_keys.animation_data and mesh_obj.data.shape_keys.animation_data.action:
                        fcurves = mesh_obj.data.shape_keys.animation_data.action.fcurves
                        for fc in fcurves:
                            for kf in fc.keyframe_points:
                                kf.interpolation = 'CONSTANT'
                current_shape_key = mesh_obj.data.shape_keys.key_blocks[f"Frame_{frame}"]
                current_shape_key.value = 1
                current_shape_key.keyframe_insert("value", frame=frame)
        print("Setting up scene...")
        cameras = setup_scene_four_views(mesh_objects[0])
        set_transparent_background()
        print("Setting up rendering...")
        scene = bpy.context.scene
        scene.render.image_settings.file_format = 'PNG'
        scene.render.resolution_x = 512
        scene.render.resolution_y = 512
        scene.render.resolution_percentage = 100
        scene.render.fps = 10
        scene.frame_start = 0
        scene.frame_end = num_frames - 1
        print("Rendering frames...")
        for frame in range(num_frames):
            scene.frame_set(frame)
            for cam, view_name in zip(cameras, view_names[:-1]):
                scene.camera = cam
                frames_dir = frames_dirs[view_name]
                scene.render.filepath = os.path.join(frames_dir, f'frame_{frame:04d}.png')
                bpy.ops.render.render(write_still=True)
                print(f"Rendered {view_name} frame {frame}/{num_frames-1}")
        circle_frames_dir = frames_dirs['circle']
        for i in range(64):
            current_frame = i % num_frames
            scene.frame_set(current_frame)
            circle_camera = setup_circle_camera(mesh_objects[0], i)
            scene.camera = circle_camera
            scene.render.filepath = os.path.join(circle_frames_dir, f'frame_{i:02d}.png')
            bpy.ops.render.render(write_still=True)
            print(f"Rendered circle view frame {i}/{64}, angle {i}/64")
            bpy.data.objects.remove(circle_camera, do_unlink=True)
        filename = output_dir.split('/')[-1]
        print("Converting frames to videos...")
        frames_circle = []
        frames_4views = []
        for view_name, frames_dir in frames_dirs.items():
            if view_name == 'circle':
                frame_files = sorted(glob.glob(os.path.join(frames_dir, f'frame_*.png')))
                for frame_file in frame_files:
                    img = iio.imread(frame_file)
                    frames_circle.append(img)
                video_path = os.path.join(os.path.dirname(output_dir), '{}_animation_circle.mp4'.format(filename))
                iio.imwrite(
                    video_path,
                    frames_circle,
                    fps=10,
                    codec='libx264',
                    quality=10,
                    pixelformat='yuv420p'
                )
            else:
                frame_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))
                if frame_files:
                    for frame_file in frame_files:
                        img = iio.imread(frame_file)
                        frames_4views.append(img)
            video_path = os.path.join(os.path.dirname(output_dir), '{}_animation_4view.mp4'.format(filename))
            iio.imwrite(
                video_path,
                frames_4views,
                fps=10,
                codec='libx264',
                quality=10,
                pixelformat='yuv420p'
            )
        print(f"All frames rendered successfully to: {output_dir}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
###