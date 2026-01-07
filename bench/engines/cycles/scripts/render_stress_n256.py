import bpy
import math
import time

# Clear default scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Parameters
width = 640
height = 360
spp = 64
bounces = 4
num_spheres = 256

# Setup render settings
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x = width
scene.render.resolution_y = height
scene.cycles.samples = spp
scene.cycles.max_bounces = bounces
scene.cycles.device = 'GPU'

# Try to use CUDA
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
prefs.get_devices()
for device in prefs.devices:
    if device.type == 'CUDA':
        device.use = True

# Create camera
bpy.ops.object.camera_add(location=(0, -12, 3))
camera = bpy.context.object
camera.rotation_euler = (math.radians(80), 0, 0)
scene.camera = camera

# Create ground plane
bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
ground = bpy.context.object
mat_ground = bpy.data.materials.new(name="Ground")
mat_ground.use_nodes = True
mat_ground.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1)
ground.data.materials.append(mat_ground)

# Create sun light
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))

# Create spheres in grid
grid_size = math.ceil(math.sqrt(num_spheres))
spacing = 2.0
offset = (grid_size - 1) * spacing / 2.0

for i in range(num_spheres):
    x = (i % grid_size) * spacing - offset
    y = (i // grid_size) * spacing - offset
    z = 0.5

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(x, y, z))
    sphere = bpy.context.object

    r = abs(math.sin(i * 0.7)) * 0.8 + 0.2
    g = abs(math.sin(i * 1.3)) * 0.8 + 0.2
    b = abs(math.sin(i * 2.1)) * 0.8 + 0.2

    mat = bpy.data.materials.new(name=f"Sphere{i}")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (r, g, b, 1)
    sphere.data.materials.append(mat)

# Set output path (use forward slashes for Python)
scene.render.filepath = "C:/Users/Admin/projects/mind-ray/bench/engines/cycles/output/output"

# Time the render
start_time = time.perf_counter()
bpy.ops.render.render(write_still=True)
end_time = time.perf_counter()

wall_ms = (end_time - start_time) * 1000.0
print(f"CYCLES_RENDER_MS={wall_ms:.2f}")