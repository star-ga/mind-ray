"""
Python Reference Path Tracer
Simple CPU-based renderer for Tier B benchmarking
"""

import numpy as np
import time
import argparse

def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-10)

def ray_sphere_intersect(origin, direction, center, radius):
    oc = origin - center
    a = np.sum(direction * direction)
    b = 2.0 * np.sum(oc * direction)
    c = np.sum(oc * oc) - radius * radius
    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return np.inf

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    if t1 > 0.001:
        return t1
    if t2 > 0.001:
        return t2
    return np.inf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='stress')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=180)
    parser.add_argument('--spp', type=int, default=4)
    parser.add_argument('--bounces', type=int, default=2)
    parser.add_argument('--spheres', type=int, default=16)
    args = parser.parse_args()

    width = args.width
    height = args.height
    spp = args.spp
    max_bounces = args.bounces
    num_spheres = args.spheres

    # Create spheres
    np.random.seed(42)
    grid_size = int(np.ceil(np.sqrt(num_spheres)))
    spacing = 2.0
    offset = (grid_size - 1) * spacing / 2.0

    spheres = []
    colors = []
    for i in range(num_spheres):
        x = (i % grid_size) * spacing - offset
        z = (i // grid_size) * spacing - offset
        y = 0.5
        spheres.append([x, y, z, 0.5])
        r = abs(np.sin(i * 0.7)) * 0.8 + 0.2
        g = abs(np.sin(i * 1.3)) * 0.8 + 0.2
        b = abs(np.sin(i * 2.1)) * 0.8 + 0.2
        colors.append([r, g, b])

    # Add ground plane (large sphere)
    spheres.append([0, -1000, 0, 1000])
    colors.append([0.5, 0.5, 0.5])

    spheres = np.array(spheres)
    colors = np.array(colors)

    # Camera
    cam_origin = np.array([0.0, 3.0, 12.0])
    cam_target = np.array([0.0, 1.0, 0.0])
    cam_up = np.array([0.0, 1.0, 0.0])

    fov = 50.0
    aspect = width / height
    theta = np.radians(fov)
    h = np.tan(theta / 2)
    viewport_height = 2.0 * h
    viewport_width = aspect * viewport_height

    w = normalize(cam_origin - cam_target)
    u = normalize(np.cross(cam_up, w))
    v = np.cross(w, u)

    horizontal = viewport_width * u
    vertical = viewport_height * v
    lower_left = cam_origin - horizontal/2 - vertical/2 - w

    def trace_ray(origin, direction, depth):
        if depth >= max_bounces:
            return np.zeros(3)

        # Find closest intersection
        min_t = np.inf
        hit_idx = -1
        for i, sphere in enumerate(spheres):
            t = ray_sphere_intersect(origin, direction, sphere[:3], sphere[3])
            if 0.001 < t < min_t:
                min_t = t
                hit_idx = i

        if hit_idx < 0:
            # Sky color
            t = 0.5 * (direction[1] + 1.0)
            return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])

        # Hit point
        hit_point = origin + min_t * direction
        sphere = spheres[hit_idx]
        normal = normalize(hit_point - sphere[:3])

        # Simple diffuse reflection
        color = colors[hit_idx]

        # Random direction in hemisphere
        rand_dir = np.random.randn(3)
        rand_dir = normalize(rand_dir)
        if np.dot(rand_dir, normal) < 0:
            rand_dir = -rand_dir

        # Recursive trace
        bounced = trace_ray(hit_point + 0.001 * normal, rand_dir, depth + 1)
        return color * bounced

    # Render
    start_time = time.perf_counter()

    image = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            color = np.zeros(3)
            for s in range(spp):
                u_coord = (x + np.random.random()) / width
                v_coord = (y + np.random.random()) / height
                direction = normalize(lower_left + u_coord * horizontal + v_coord * vertical - cam_origin)
                color += trace_ray(cam_origin, direction, 0)
            image[height - 1 - y, x] = color / spp

    end_time = time.perf_counter()
    wall_ms = (end_time - start_time) * 1000.0
    total_samples = width * height * spp
    samples_per_sec = total_samples / (end_time - start_time) / 1e6

    # Output contract keys
    print(f"ENGINE=Python-Reference")
    print(f"ENGINE_VERSION=1.0")
    print(f"TIER=B")
    print(f"SCENE={args.scene}")
    print(f"WIDTH={width}")
    print(f"HEIGHT={height}")
    print(f"SPP={spp}")
    print(f"BOUNCES={max_bounces}")
    print(f"SPHERES={num_spheres}")
    print(f"SCENE_MATCH=approx")
    print(f"WALL_MS_TOTAL={wall_ms:.2f}")
    print(f"WALL_SAMPLES_PER_SEC={samples_per_sec:.4f}")
    print(f"STATUS=complete")

if __name__ == '__main__':
    main()
