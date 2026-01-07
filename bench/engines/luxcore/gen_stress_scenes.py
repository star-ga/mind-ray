#!/usr/bin/env python3
"""Generate LuxCore stress test scenes with N spheres."""

import math
import os

def generate_sphere_ply(filename, center, radius, subdivisions=16):
    """Generate a UV sphere as PLY file."""
    vertices = []
    faces = []

    cx, cy, cz = center

    # Generate vertices
    for i in range(subdivisions + 1):
        lat = math.pi * i / subdivisions
        for j in range(subdivisions):
            lon = 2 * math.pi * j / subdivisions
            x = cx + radius * math.sin(lat) * math.cos(lon)
            y = cy + radius * math.sin(lat) * math.sin(lon)
            z = cz + radius * math.cos(lat)
            vertices.append((x, y, z))

    # Generate faces (quads split into triangles)
    for i in range(subdivisions):
        for j in range(subdivisions):
            p1 = i * subdivisions + j
            p2 = i * subdivisions + (j + 1) % subdivisions
            p3 = (i + 1) * subdivisions + (j + 1) % subdivisions
            p4 = (i + 1) * subdivisions + j
            faces.append((p1, p2, p3))
            faces.append((p1, p3, p4))

    # Write PLY file (Unix line endings required)
    with open(filename, 'w', newline='\n') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar uint vertex_indices\n")
        f.write("end_header\n")

        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    return len(vertices), len(faces)


def generate_box_ply(filename, min_pt, max_pt):
    """Generate a box as PLY file."""
    x0, y0, z0 = min_pt
    x1, y1, z1 = max_pt

    vertices = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),  # bottom
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),  # top
    ]

    faces = [
        (0, 2, 1), (0, 3, 2),  # bottom
        (4, 5, 6), (4, 6, 7),  # top
        (0, 1, 5), (0, 5, 4),  # front
        (2, 3, 7), (2, 7, 6),  # back
        (0, 4, 7), (0, 7, 3),  # left
        (1, 2, 6), (1, 6, 5),  # right
    ]

    # Write PLY file (Unix line endings required)
    with open(filename, 'w', newline='\n') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar uint vertex_indices\n")
        f.write("end_header\n")

        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def generate_stress_scene(output_dir, n_spheres, spp=64, resolution=(640, 360)):
    """Generate a stress test scene with N spheres."""

    scene_dir = os.path.join(output_dir, f"stress_n{n_spheres}")
    os.makedirs(scene_dir, exist_ok=True)

    # Generate ground plane PLY
    ground_ply = os.path.join(scene_dir, "ground.ply")
    generate_box_ply(ground_ply, (-20, -20, -0.5), (20, 20, 0))

    # Generate area light PLY (ceiling light)
    light_ply = os.path.join(scene_dir, "light.ply")
    generate_box_ply(light_ply, (-3, -3, 9.9), (3, 3, 10))

    # Generate sphere PLY files and calculate positions
    sphere_positions = []
    grid_size = int(math.ceil(math.sqrt(n_spheres)))
    spacing = 2.5
    offset = (grid_size - 1) * spacing / 2

    sphere_idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if sphere_idx >= n_spheres:
                break
            x = i * spacing - offset
            y = j * spacing - offset
            z = 1.0  # sphere center height

            sphere_ply = os.path.join(scene_dir, f"sphere_{sphere_idx}.ply")
            generate_sphere_ply(sphere_ply, (x, y, z), 0.8, subdivisions=16)
            sphere_positions.append((sphere_idx, x, y, z))
            sphere_idx += 1

    # Generate .scn file
    scn_content = []
    scn_content.append("# LuxCore Stress Test Scene")
    scn_content.append(f"# N={n_spheres} spheres")
    scn_content.append("")

    # Camera
    cam_dist = max(15, grid_size * spacing * 0.8)
    scn_content.append('scene.camera.type = "perspective"')
    scn_content.append(f'scene.camera.lookat.orig = {cam_dist} {-cam_dist} {cam_dist * 0.6}')
    scn_content.append(f'scene.camera.lookat.target = 0 0 1')
    scn_content.append('scene.camera.fieldofview = 50')
    scn_content.append("")

    # Materials
    scn_content.append("# Materials")
    scn_content.append("scene.materials.ground.type = matte")
    scn_content.append("scene.materials.ground.kd = 0.7 0.7 0.7")
    scn_content.append("")
    scn_content.append("scene.materials.sphere.type = glossy2")
    scn_content.append("scene.materials.sphere.kd = 0.1 0.3 0.8")
    scn_content.append("scene.materials.sphere.ks = 0.6 0.6 0.6")
    scn_content.append("scene.materials.sphere.uroughness = 0.1")
    scn_content.append("scene.materials.sphere.vroughness = 0.1")
    scn_content.append("")
    scn_content.append("scene.materials.light.type = matte")
    scn_content.append("scene.materials.light.emission = 50 50 50")
    scn_content.append("scene.materials.light.kd = 1 1 1")
    scn_content.append("")

    # Objects
    scn_content.append("# Objects")
    scn_content.append("scene.objects.ground.material = ground")
    scn_content.append(f"scene.objects.ground.ply = stress_n{n_spheres}/ground.ply")
    scn_content.append("")
    scn_content.append("scene.objects.light.material = light")
    scn_content.append(f"scene.objects.light.ply = stress_n{n_spheres}/light.ply")
    scn_content.append("")

    for idx, x, y, z in sphere_positions:
        scn_content.append(f"scene.objects.sphere_{idx}.material = sphere")
        scn_content.append(f"scene.objects.sphere_{idx}.ply = stress_n{n_spheres}/sphere_{idx}.ply")

    scn_path = os.path.join(scene_dir, f"stress_n{n_spheres}.scn")
    with open(scn_path, 'w') as f:
        f.write('\n'.join(scn_content))

    # Generate .cfg file
    cfg_content = f"""# LuxCore GPU Benchmark - Stress N={n_spheres}
# Tier B: End-to-end wall clock

film.width = {resolution[0]}
film.height = {resolution[1]}
scene.file = stress_n{n_spheres}/stress_n{n_spheres}.scn

# GPU-only path tracing via OpenCL
renderengine.type = PATHOCL
opencl.gpu.use = 1
opencl.cpu.use = 0

# Sampler
sampler.type = SOBOL

# Halt conditions (SPP target with time fallback)
batch.haltspp = {spp}
batch.halttime = 120

# Path depth
path.maxdepth = 4

# Disable adaptive sampling convergence test
film.noiseestimation.warmup = 0
film.noiseestimation.step = 0

# Output
film.outputs.0.type = RGB_IMAGEPIPELINE
film.outputs.0.filename = stress_n{n_spheres}_output.png
film.imagepipelines.0.0.type = TONEMAP_LINEAR
film.imagepipelines.0.0.scale = 1.0
film.imagepipelines.0.1.type = GAMMA_CORRECTION
film.imagepipelines.0.1.value = 2.2
"""

    cfg_path = os.path.join(output_dir, f"stress_n{n_spheres}.cfg")
    with open(cfg_path, 'w') as f:
        f.write(cfg_content)

    print(f"Generated stress_n{n_spheres}:")
    print(f"  - {len(sphere_positions)} spheres")
    print(f"  - Scene: {scn_path}")
    print(f"  - Config: {cfg_path}")
    return cfg_path, scn_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scenes_dir = os.path.join(script_dir, "scenes")
    os.makedirs(scenes_dir, exist_ok=True)

    # Generate stress scenes for N=64, 128, 256
    for n in [64, 128, 256]:
        generate_stress_scene(scenes_dir, n, spp=64)

    print("\nAll stress scenes generated!")
    print(f"Output directory: {scenes_dir}")


if __name__ == "__main__":
    main()
