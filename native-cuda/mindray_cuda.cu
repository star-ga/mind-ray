// mindray_cuda.cu - CUDA GPU Backend for Mind Ray
//
// This module provides a high-performance CUDA implementation of path tracing
// that can be called from Mind via FFI. It includes:
// - Deterministic per-pixel RNG (xorshift32)
// - Progressive frame accumulation
// - Sphere and plane intersection
// - Soft shadows and reflections
// - Russian roulette path termination
//
// Build: nvcc -O3 -shared -o mindray_cuda.dll mindray_cuda.cu -Xcompiler "/MD"
//
// Author: Mind Ray Contributors
// License: MIT

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>

// =============================================================================
// Export macros for DLL
// =============================================================================

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

// =============================================================================
// Vector Math (GPU compatible)
// =============================================================================

struct float3x {
    float x, y, z;
};

__host__ __device__ inline float3x make_f3(float x, float y, float z) {
    return {x, y, z};
}

__host__ __device__ inline float3x operator+(float3x a, float3x b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ inline float3x operator-(float3x a, float3x b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ inline float3x operator*(float3x a, float b) {
    return {a.x * b, a.y * b, a.z * b};
}

__host__ __device__ inline float3x operator*(float b, float3x a) {
    return a * b;
}

__host__ __device__ inline float3x operator/(float3x a, float b) {
    float inv = 1.0f / b;
    return {a.x * inv, a.y * inv, a.z * inv};
}

__host__ __device__ inline float dot(float3x a, float3x b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3x cross(float3x a, float3x b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__host__ __device__ inline float len(float3x a) {
    return sqrtf(dot(a, a));
}

__host__ __device__ inline float3x normalize(float3x a) {
    float l = len(a);
    return (l > 1e-8f) ? (a / l) : make_f3(0, 0, 0);
}

__host__ __device__ inline float3x clamp01(float3x a) {
    return {
        fminf(1.0f, fmaxf(0.0f, a.x)),
        fminf(1.0f, fmaxf(0.0f, a.y)),
        fminf(1.0f, fmaxf(0.0f, a.z))
    };
}

__host__ __device__ inline float3x hadamard(float3x a, float3x b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__host__ __device__ inline float3x reflect(float3x v, float3x n) {
    return v - n * (2.0f * dot(v, n));
}

// =============================================================================
// RNG (xorshift32 - deterministic per thread)
// =============================================================================

__device__ inline uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ inline float rand01(uint32_t& state) {
    uint32_t r = xorshift32(state);
    return (r & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

__device__ inline float3x random_in_unit_sphere(uint32_t& state) {
    // Rejection sampling for uniform distribution in unit sphere
    for (int i = 0; i < 100; ++i) {
        float3x p = make_f3(
            rand01(state) * 2.0f - 1.0f,
            rand01(state) * 2.0f - 1.0f,
            rand01(state) * 2.0f - 1.0f
        );
        if (dot(p, p) < 1.0f) return p;
    }
    return make_f3(0, 1, 0); // Fallback
}

// =============================================================================
// Scene Geometry
// =============================================================================

struct Sphere {
    float3x center;
    float radius;
    float3x albedo;
    float metallic;
    float roughness;
};

struct Hit {
    float t;
    float3x point;
    float3x normal;
    float3x albedo;
    float metallic;
    float roughness;
    int did_hit;
};

__device__ inline Hit miss() {
    Hit h;
    h.t = 1e30f;
    h.did_hit = 0;
    return h;
}

__device__ inline Hit hit_sphere(const Sphere& s, float3x ro, float3x rd, float t_min, float t_max) {
    float3x oc = ro - s.center;
    float a = dot(rd, rd);
    float half_b = dot(oc, rd);
    float c = dot(oc, oc) - s.radius * s.radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant < 0.0f) return miss();

    float sqrtd = sqrtf(discriminant);
    float root = (-half_b - sqrtd) / a;

    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) return miss();
    }

    Hit h;
    h.t = root;
    h.point = ro + rd * root;
    h.normal = normalize(h.point - s.center);
    h.albedo = s.albedo;
    h.metallic = s.metallic;
    h.roughness = s.roughness;
    h.did_hit = 1;
    return h;
}

__device__ inline Hit hit_plane(float3x ro, float3x rd, float3x normal, float d, float t_min, float t_max) {
    float denom = dot(normal, rd);
    if (fabsf(denom) < 1e-6f) return miss();

    float t = -(dot(normal, ro) + d) / denom;
    if (t < t_min || t > t_max) return miss();

    Hit h;
    h.t = t;
    h.point = ro + rd * t;
    h.normal = (denom < 0.0f) ? normal : (normal * -1.0f);

    // Checkerboard pattern
    float scale = 2.0f;
    int cx = (int)floorf(h.point.x * scale);
    int cz = (int)floorf(h.point.z * scale);
    int checker = (cx + cz) & 1;
    h.albedo = checker ? make_f3(0.9f, 0.9f, 0.9f) : make_f3(0.2f, 0.25f, 0.3f);
    h.metallic = 0.0f;
    h.roughness = 0.9f;
    h.did_hit = 1;
    return h;
}

// Scene constants for different presets
__constant__ int g_scene_type = 0;  // 0=spheres, 1=cornell, 2=stress

__device__ Hit scene_hit(float3x ro, float3x rd, int scene_type, int num_spheres) {
    Hit best = miss();
    Hit h;

    if (scene_type == 0) {
        // Spheres scene
        Sphere s0 = {make_f3(0.0f, 1.0f, -3.5f), 1.0f, make_f3(0.8f, 0.3f, 0.2f), 0.0f, 0.8f};
        Sphere s1 = {make_f3(-1.8f, 0.9f, -2.4f), 0.9f, make_f3(0.2f, 0.7f, 0.9f), 0.0f, 0.6f};
        Sphere s2 = {make_f3(1.7f, 0.8f, -2.0f), 0.8f, make_f3(0.9f, 0.85f, 0.7f), 1.0f, 0.15f};

        h = hit_sphere(s0, ro, rd, 0.001f, best.t); if (h.did_hit) best = h;
        h = hit_sphere(s1, ro, rd, 0.001f, best.t); if (h.did_hit) best = h;
        h = hit_sphere(s2, ro, rd, 0.001f, best.t); if (h.did_hit) best = h;

        // Ground plane
        h = hit_plane(ro, rd, make_f3(0, 1, 0), 0.0f, 0.001f, best.t);
        if (h.did_hit) best = h;
    }
    else if (scene_type == 1) {
        // Cornell box (simplified)
        Sphere s0 = {make_f3(-0.5f, 1.0f, -2.0f), 0.5f, make_f3(0.9f, 0.9f, 0.9f), 0.0f, 0.1f};
        Sphere s1 = {make_f3(0.5f, 0.5f, -1.5f), 0.5f, make_f3(0.9f, 0.9f, 0.9f), 1.0f, 0.0f};

        h = hit_sphere(s0, ro, rd, 0.001f, best.t); if (h.did_hit) best = h;
        h = hit_sphere(s1, ro, rd, 0.001f, best.t); if (h.did_hit) best = h;

        // Cornell box walls
        h = hit_plane(ro, rd, make_f3(0, 1, 0), 0.0f, 0.001f, best.t); // Floor (white)
        if (h.did_hit) { h.albedo = make_f3(0.73f, 0.73f, 0.73f); best = h; }

        h = hit_plane(ro, rd, make_f3(0, -1, 0), -5.0f, 0.001f, best.t); // Ceiling
        if (h.did_hit) { h.albedo = make_f3(0.73f, 0.73f, 0.73f); best = h; }

        h = hit_plane(ro, rd, make_f3(1, 0, 0), 2.5f, 0.001f, best.t); // Left (red)
        if (h.did_hit) { h.albedo = make_f3(0.65f, 0.05f, 0.05f); best = h; }

        h = hit_plane(ro, rd, make_f3(-1, 0, 0), 2.5f, 0.001f, best.t); // Right (green)
        if (h.did_hit) { h.albedo = make_f3(0.12f, 0.45f, 0.15f); best = h; }

        h = hit_plane(ro, rd, make_f3(0, 0, 1), 5.0f, 0.001f, best.t); // Back
        if (h.did_hit) { h.albedo = make_f3(0.73f, 0.73f, 0.73f); best = h; }
    }
    else {
        // Stress test (many spheres)
        for (int i = 0; i < num_spheres; ++i) {
            float x = (float)(i % 10) - 4.5f;
            float z = (float)(i / 10) * -1.5f - 2.0f;
            float r = 0.3f + 0.1f * ((i * 7) % 5);
            float3x color = make_f3(
                0.3f + 0.7f * ((i * 13) % 7) / 6.0f,
                0.3f + 0.7f * ((i * 17) % 7) / 6.0f,
                0.3f + 0.7f * ((i * 19) % 7) / 6.0f
            );
            float metal = ((i * 23) % 3 == 0) ? 1.0f : 0.0f;
            Sphere s = {make_f3(x, r, z), r, color, metal, 0.3f};
            h = hit_sphere(s, ro, rd, 0.001f, best.t);
            if (h.did_hit) best = h;
        }

        h = hit_plane(ro, rd, make_f3(0, 1, 0), 0.0f, 0.001f, best.t);
        if (h.did_hit) best = h;
    }

    return best;
}

// =============================================================================
// Lighting and Shading
// =============================================================================

__device__ float3x sky_color(float3x rd) {
    float t = 0.5f * (rd.y + 1.0f);
    return make_f3(0.6f, 0.75f, 0.95f) * t + make_f3(0.95f, 0.98f, 1.0f) * (1.0f - t);
}

__device__ float soft_shadow(float3x p, float3x light_dir, uint32_t& rng, int scene_type, int num_spheres) {
    float3x jitter = random_in_unit_sphere(rng) * 0.02f;
    float3x d = normalize(light_dir + jitter);
    Hit h = scene_hit(p + d * 0.01f, d, scene_type, num_spheres);
    return h.did_hit ? 0.0f : 1.0f;
}

__device__ float3x trace_path(float3x ro, float3x rd, uint32_t& rng, int scene_type, int max_bounces, int num_spheres) {
    float3x throughput = make_f3(1, 1, 1);
    float3x radiance = make_f3(0, 0, 0);

    float3x sun_dir = normalize(make_f3(-0.4f, 0.9f, -0.2f));
    float3x sun_col = make_f3(3.0f, 2.8f, 2.5f);

    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        Hit h = scene_hit(ro, rd, scene_type, num_spheres);

        if (!h.did_hit) {
            radiance = radiance + hadamard(throughput, sky_color(rd));
            break;
        }

        // Direct lighting with soft shadows
        float ndl = fmaxf(0.0f, dot(h.normal, sun_dir));
        float shadow = soft_shadow(h.point, sun_dir, rng, scene_type, num_spheres);
        float3x direct = sun_col * (ndl * shadow);
        radiance = radiance + hadamard(throughput, hadamard(h.albedo, direct));

        // Scatter ray
        float3x new_dir;
        if (h.metallic > 0.5f) {
            // Metal reflection
            float3x reflected = reflect(rd, h.normal);
            float3x fuzz = random_in_unit_sphere(rng) * h.roughness;
            new_dir = normalize(reflected + fuzz);
        } else {
            // Diffuse (cosine-weighted hemisphere)
            float3x random_vec = normalize(random_in_unit_sphere(rng));
            new_dir = normalize(h.normal + random_vec);
        }

        ro = h.point + h.normal * 0.002f;
        rd = new_dir;
        throughput = hadamard(throughput, h.albedo);

        // Russian roulette after 2 bounces
        if (bounce >= 2) {
            float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            p = fminf(p, 0.95f);
            if (rand01(rng) > p) break;
            throughput = throughput / p;
        }
    }

    return radiance;
}

// =============================================================================
// Main Render Kernel
// =============================================================================

__global__ void render_kernel(
    float3* accum_buffer,
    uint8_t* output_rgb,
    int width,
    int height,
    int samples_per_pixel,
    int frame_index,
    int scene_type,
    int max_bounces,
    uint32_t base_seed,
    int num_spheres,
    // Camera parameters
    float cam_pos_x, float cam_pos_y, float cam_pos_z,
    float cam_look_x, float cam_look_y, float cam_look_z,
    float fov_degrees
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel_index = y * width + x;

    // Deterministic RNG seed: combines base_seed, pixel position, and frame
    uint32_t rng = base_seed;
    rng ^= (uint32_t)(pixel_index * 747796405u);
    rng ^= (uint32_t)(frame_index * 2891336453u);
    xorshift32(rng); // Warm up

    // Camera setup
    float aspect = (float)width / (float)height;
    float3x cam_pos = make_f3(cam_pos_x, cam_pos_y, cam_pos_z);
    float3x cam_look = make_f3(cam_look_x, cam_look_y, cam_look_z);
    float3x cam_up = make_f3(0.0f, 1.0f, 0.0f);
    float fov = fov_degrees * 3.14159265f / 180.0f;

    float3x forward = normalize(cam_look - cam_pos);
    float3x right = normalize(cross(forward, cam_up));
    float3x up = cross(right, forward);

    // Accumulate samples
    float3x color = make_f3(0, 0, 0);

    for (int s = 0; s < samples_per_pixel; ++s) {
        float u = ((float)x + rand01(rng)) / (float)width;
        float v = ((float)y + rand01(rng)) / (float)height;

        float px = (2.0f * u - 1.0f) * tanf(fov * 0.5f) * aspect;
        float py = (1.0f - 2.0f * v) * tanf(fov * 0.5f);

        float3x ray_dir = normalize(forward + right * px + up * py);
        color = color + trace_path(cam_pos, ray_dir, rng, scene_type, max_bounces, num_spheres);
    }

    color = color / (float)samples_per_pixel;

    // Progressive accumulation (EMA)
    float3 prev = accum_buffer[pixel_index];
    float alpha = 1.0f / (float)(frame_index + 1);
    float3 current;
    current.x = prev.x + (color.x - prev.x) * alpha;
    current.y = prev.y + (color.y - prev.y) * alpha;
    current.z = prev.z + (color.z - prev.z) * alpha;
    accum_buffer[pixel_index] = current;

    // Tonemap (Reinhard) and gamma correction
    auto tonemap = [](float v) {
        v = fmaxf(0.0f, v);
        v = v / (1.0f + v);
        return powf(v, 1.0f / 2.2f);
    };

    float3x final_color = make_f3(tonemap(current.x), tonemap(current.y), tonemap(current.z));
    final_color = clamp01(final_color);

    output_rgb[pixel_index * 3 + 0] = (uint8_t)lrintf(final_color.x * 255.0f);
    output_rgb[pixel_index * 3 + 1] = (uint8_t)lrintf(final_color.y * 255.0f);
    output_rgb[pixel_index * 3 + 2] = (uint8_t)lrintf(final_color.z * 255.0f);
}

// =============================================================================
// C ABI Exports for Mind FFI
// =============================================================================

struct MindRayCudaContext {
    int device_id;
    int width;
    int height;
    float3* d_accum;
    uint8_t* d_output;
    int current_frame;
    int num_spheres;  // For stress scene variable sphere count
    char error_msg[256];
};

static MindRayCudaContext* g_ctx = nullptr;

EXPORT int mindray_cuda_init(int device_id, int width, int height) {
    if (g_ctx) {
        return -1; // Already initialized
    }

    g_ctx = new MindRayCudaContext();
    g_ctx->device_id = device_id;
    g_ctx->width = width;
    g_ctx->height = height;
    g_ctx->current_frame = 0;
    g_ctx->num_spheres = 50;  // Default stress scene sphere count
    g_ctx->error_msg[0] = '\0';

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        snprintf(g_ctx->error_msg, 256, "cudaSetDevice failed: %s", cudaGetErrorString(err));
        delete g_ctx;
        g_ctx = nullptr;
        return -2;
    }

    size_t pixels = (size_t)width * (size_t)height;

    err = cudaMalloc(&g_ctx->d_accum, pixels * sizeof(float3));
    if (err != cudaSuccess) {
        snprintf(g_ctx->error_msg, 256, "cudaMalloc accum failed: %s", cudaGetErrorString(err));
        delete g_ctx;
        g_ctx = nullptr;
        return -3;
    }

    err = cudaMalloc(&g_ctx->d_output, pixels * 3);
    if (err != cudaSuccess) {
        cudaFree(g_ctx->d_accum);
        snprintf(g_ctx->error_msg, 256, "cudaMalloc output failed: %s", cudaGetErrorString(err));
        delete g_ctx;
        g_ctx = nullptr;
        return -4;
    }

    err = cudaMemset(g_ctx->d_accum, 0, pixels * sizeof(float3));
    if (err != cudaSuccess) {
        cudaFree(g_ctx->d_accum);
        cudaFree(g_ctx->d_output);
        delete g_ctx;
        g_ctx = nullptr;
        return -5;
    }

    return 0; // Success
}

EXPORT int mindray_cuda_render_frame(
    int spp,
    int scene_type,
    int max_bounces,
    uint32_t seed,
    float cam_x, float cam_y, float cam_z,
    float look_x, float look_y, float look_z,
    float fov
) {
    if (!g_ctx) return -1;

    dim3 block(16, 16);
    dim3 grid(
        (g_ctx->width + block.x - 1) / block.x,
        (g_ctx->height + block.y - 1) / block.y
    );

    render_kernel<<<grid, block>>>(
        g_ctx->d_accum,
        g_ctx->d_output,
        g_ctx->width,
        g_ctx->height,
        spp,
        g_ctx->current_frame,
        scene_type,
        max_bounces,
        seed,
        g_ctx->num_spheres,
        cam_x, cam_y, cam_z,
        look_x, look_y, look_z,
        fov
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        snprintf(g_ctx->error_msg, 256, "Kernel launch failed: %s", cudaGetErrorString(err));
        return -2;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        snprintf(g_ctx->error_msg, 256, "cudaDeviceSynchronize failed: %s", cudaGetErrorString(err));
        return -3;
    }

    g_ctx->current_frame++;
    return 0;
}

EXPORT int mindray_cuda_copy_output(uint8_t* host_buffer) {
    if (!g_ctx) return -1;

    size_t bytes = (size_t)g_ctx->width * (size_t)g_ctx->height * 3;
    cudaError_t err = cudaMemcpy(host_buffer, g_ctx->d_output, bytes, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        snprintf(g_ctx->error_msg, 256, "cudaMemcpy failed: %s", cudaGetErrorString(err));
        return -2;
    }

    return 0;
}

EXPORT int mindray_cuda_reset_accumulator() {
    if (!g_ctx) return -1;

    size_t bytes = (size_t)g_ctx->width * (size_t)g_ctx->height * sizeof(float3);
    cudaError_t err = cudaMemset(g_ctx->d_accum, 0, bytes);

    if (err != cudaSuccess) {
        return -2;
    }

    g_ctx->current_frame = 0;
    return 0;
}

EXPORT void mindray_cuda_free() {
    if (!g_ctx) return;

    if (g_ctx->d_accum) cudaFree(g_ctx->d_accum);
    if (g_ctx->d_output) cudaFree(g_ctx->d_output);

    delete g_ctx;
    g_ctx = nullptr;
}

EXPORT const char* mindray_cuda_get_error() {
    if (!g_ctx) return "Not initialized";
    return g_ctx->error_msg;
}

EXPORT int mindray_cuda_get_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

EXPORT int mindray_cuda_get_device_name(int device_id, char* name_buffer, int buffer_size) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) return -1;

    strncpy(name_buffer, prop.name, buffer_size - 1);
    name_buffer[buffer_size - 1] = '\0';
    return 0;
}

EXPORT int mindray_cuda_get_frame_count() {
    if (!g_ctx) return 0;
    return g_ctx->current_frame;
}

EXPORT void mindray_cuda_set_num_spheres(int n) {
    if (g_ctx) {
        g_ctx->num_spheres = (n > 0) ? n : 50;
    }
}

EXPORT int mindray_cuda_get_num_spheres() {
    if (!g_ctx) return 50;
    return g_ctx->num_spheres;
}
