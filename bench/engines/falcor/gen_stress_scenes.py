#!/usr/bin/env python3
"""Generate stress sphere scenes for Falcor benchmarks.

Creates .pyscene files with N spheres in a grid pattern, matching
the stress scene used by other Tier B engines.
"""

import os
import math

SCENE_DIR = os.path.join(os.path.dirname(__file__), "scenes")


def generate_stress_scene(n_spheres: int) -> str:
    """Generate a pyscene file with n_spheres in a grid."""

    # Calculate grid dimensions (cube root for 3D grid)
    grid_size = max(1, int(math.ceil(n_spheres ** (1/3))))
    spacing = 2.5

    lines = [
        "# Stress scene with {} spheres".format(n_spheres),
        "# Auto-generated for Tier B benchmarks",
        "",
        "# Create materials",
        "",
        "# Ground plane material",
        "groundMaterial = Material('Ground')",
        "groundMaterial.baseColor = float4(0.5, 0.5, 0.5, 1.0)",
        "groundMaterial.roughness = 0.8",
        "",
        "# Light material",
        "lightMaterial = Material('Light')",
        "lightMaterial.emissiveColor = float3(17, 12, 4)",
        "lightMaterial.emissiveFactor = 3",
        "",
    ]

    # Create sphere materials with varying colors
    for i in range(n_spheres):
        r = 0.3 + 0.7 * ((i * 7) % 11) / 10.0
        g = 0.3 + 0.7 * ((i * 13) % 11) / 10.0
        b = 0.3 + 0.7 * ((i * 17) % 11) / 10.0
        metallic = 0.0 if i % 3 == 0 else 0.9
        roughness = 0.1 + 0.8 * ((i * 19) % 10) / 10.0

        lines.extend([
            f"sphereMat{i} = Material('Sphere{i}')",
            f"sphereMat{i}.baseColor = float4({r:.3f}, {g:.3f}, {b:.3f}, 1.0)",
            f"sphereMat{i}.roughness = {roughness:.2f}",
            f"sphereMat{i}.metallic = {metallic:.1f}",
            "",
        ])

    lines.extend([
        "# Create geometry",
        "",
        "quadMesh = TriangleMesh.createQuad()",
        "sphereMesh = TriangleMesh.createSphere()",
        "",
        "# Ground plane",
        "sceneBuilder.addMeshInstance(",
        f"    sceneBuilder.addNode('Ground', Transform(scaling=float3({grid_size * spacing * 2}, 1.0, {grid_size * spacing * 2}))),",
        "    sceneBuilder.addTriangleMesh(quadMesh, groundMaterial)",
        ")",
        "",
        "# Area light",
        "sceneBuilder.addMeshInstance(",
        f"    sceneBuilder.addNode('Light', Transform(scaling=float3({grid_size * spacing}, 1.0, {grid_size * spacing}), translation=float3(0, {grid_size * spacing + 5}, 0), rotationEulerDeg=float3(180, 0, 0))),",
        "    sceneBuilder.addTriangleMesh(quadMesh, lightMaterial)",
        ")",
        "",
        "# Spheres in grid",
    ])

    # Place spheres in a 3D grid
    sphere_idx = 0
    center_offset = (grid_size - 1) * spacing / 2

    for z in range(grid_size):
        for y in range(grid_size):
            for x in range(grid_size):
                if sphere_idx >= n_spheres:
                    break

                px = x * spacing - center_offset
                py = y * spacing + 1.0  # Raise above ground
                pz = z * spacing - center_offset

                lines.extend([
                    "sceneBuilder.addMeshInstance(",
                    f"    sceneBuilder.addNode('Sphere{sphere_idx}', Transform(translation=float3({px:.2f}, {py:.2f}, {pz:.2f}))),",
                    f"    sceneBuilder.addTriangleMesh(sphereMesh, sphereMat{sphere_idx})",
                    ")",
                ])

                sphere_idx += 1
            if sphere_idx >= n_spheres:
                break
        if sphere_idx >= n_spheres:
            break

    # Camera positioned to see all spheres
    cam_distance = grid_size * spacing * 1.5
    cam_height = grid_size * spacing * 0.8

    lines.extend([
        "",
        "# Camera",
        "camera = Camera()",
        f"camera.position = float3(0, {cam_height:.1f}, -{cam_distance:.1f})",
        "camera.target = float3(0, 1, 0)",
        "camera.up = float3(0, 1, 0)",
        "camera.focalLength = 35.0",
        "sceneBuilder.addCamera(camera)",
    ])

    return "\n".join(lines)


def main():
    os.makedirs(SCENE_DIR, exist_ok=True)

    for n in [64, 128, 256]:
        scene_content = generate_stress_scene(n)
        scene_path = os.path.join(SCENE_DIR, f"stress_n{n}.pyscene")

        with open(scene_path, "w") as f:
            f.write(scene_content)

        print(f"Generated: {scene_path}")


if __name__ == "__main__":
    main()
