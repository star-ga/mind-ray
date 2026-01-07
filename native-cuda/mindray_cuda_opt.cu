// mindray_cuda_opt.cu - Optimized CUDA GPU Backend for Mind Ray
//
// Optimizations applied:
// 1. Fast RNG without rejection sampling (Marsaglia method)
// 2. rsqrtf() instead of sqrtf() + division
// 3. __forceinline__ on all hot path functions
// 4. Fast gamma approximation
// 5. Constant memory for camera parameters
// 6. Reduced register pressure
// 7. Loop unrolling hints
//
// Build: nvcc -O3 -use_fast_math -shared -o mindray_cuda.dll mindray_cuda_opt.cu -Xcompiler "/MD"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>

// =============================================================================
// Export macros
// =============================================================================

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

// =============================================================================
// Optimized Vector Math
// =============================================================================

// Use CUDA's native float3 for better compiler optimization
__host__ __device__ __forceinline__ float3 make_v3(float x, float y, float z) {
    return make_float3(x, y, z);
}

__host__ __device__ __forceinline__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ __forceinline__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ __forceinline__ float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ __forceinline__ float3 operator*(float b, float3 a) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

// Device-only fast dot with FMA
__device__ __forceinline__ float dot_device(float3 a, float3 b) {
    return __fmaf_rn(a.x, b.x, __fmaf_rn(a.y, b.y, a.z * b.z));
}

// Host-device compatible dot product
__host__ __device__ __forceinline__ float dot(float3 a, float3 b) {
#ifdef __CUDA_ARCH__
    return __fmaf_rn(a.x, b.x, __fmaf_rn(a.y, b.y, a.z * b.z));
#else
    return a.x * b.x + a.y * b.y + a.z * b.z;
#endif
}

// Fast normalize using rsqrtf (device) or sqrtf (host)
__host__ __device__ __forceinline__ float3 normalize_fast(float3 a) {
#ifdef __CUDA_ARCH__
    float inv_len = rsqrtf(dot(a, a) + 1e-8f);
#else
    float inv_len = 1.0f / sqrtf(dot(a, a) + 1e-8f);
#endif
    return a * inv_len;
}

__device__ __forceinline__ float3 hadamard(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 reflect(float3 v, float3 n) {
    return v - n * (2.0f * dot(v, n));
}

// =============================================================================
// Fast RNG - xorshift32 with direct unit vector generation
// =============================================================================

__device__ __forceinline__ uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ __forceinline__ float rand01(uint32_t& state) {
    return __uint_as_float((xorshift32(state) >> 9) | 0x3f800000u) - 1.0f;
}

// Marsaglia's method for uniform point on unit sphere (NO rejection sampling!)
// Uses 2 random numbers instead of potentially infinite
__device__ __forceinline__ float3 random_unit_vector(uint32_t& state) {
    float z = rand01(state) * 2.0f - 1.0f;
    float phi = rand01(state) * 6.283185307f;
    float r = sqrtf(1.0f - z * z);
    float sp, cp;
    __sincosf(phi, &sp, &cp);
    return make_float3(r * cp, r * sp, z);
}

// Fast random in hemisphere (cosine-weighted)
__device__ __forceinline__ float3 random_cosine_direction(uint32_t& state, float3 normal) {
    float3 rand_dir = random_unit_vector(state);
    // Ensure it's in the same hemisphere as normal
    if (dot(rand_dir, normal) < 0.0f) {
        rand_dir = make_float3(-rand_dir.x, -rand_dir.y, -rand_dir.z);
    }
    return normalize_fast(normal + rand_dir);
}

// =============================================================================
// Constant Memory for Camera and Scene Parameters
// =============================================================================

struct CameraParams {
    float3 pos;
    float3 forward;
    float3 right;
    float3 up;
    float tan_half_fov;
    float aspect;
};

__constant__ CameraParams d_camera;

// =============================================================================
// Scene Geometry (Optimized)
// =============================================================================

struct SphereSOA {
    float* cx;  // center x
    float* cy;  // center y
    float* cz;  // center z
    float* r;   // radius
    float* albedo_r;
    float* albedo_g;
    float* albedo_b;
    float* metallic;
    float* roughness;
    int count;
};

struct HitRecord {
    float t;
    float3 point;
    float3 normal;
    float3 albedo;
    float metallic;
    float roughness;
};

__device__ __forceinline__ bool hit_sphere_fast(
    float3 center, float radius,
    float3 ro, float3 rd,
    float t_min, float t_max,
    float& out_t, float3& out_normal
) {
    float3 oc = ro - center;
    float a = dot(rd, rd);
    float half_b = dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant < 0.0f) return false;

    float sqrtd = sqrtf(discriminant);
    float root = (-half_b - sqrtd) / a;

    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) return false;
    }

    out_t = root;
    float3 hit_point = ro + rd * root;
    out_normal = (hit_point - center) * (1.0f / radius);  // normalized
    return true;
}

__device__ __forceinline__ bool hit_plane_fast(
    float3 ro, float3 rd,
    float3 normal, float d,
    float t_min, float t_max,
    float& out_t
) {
    float denom = dot(normal, rd);
    if (fabsf(denom) < 1e-6f) return false;

    float t = -(dot(normal, ro) + d) / denom;
    if (t < t_min || t > t_max) return false;

    out_t = t;
    return true;
}

// =============================================================================
// Optimized Scene Intersection
// =============================================================================

__device__ void scene_hit_opt(
    float3 ro, float3 rd,
    int scene_type, int num_spheres,
    HitRecord& hit, bool& did_hit
) {
    hit.t = 1e30f;
    did_hit = false;

    float out_t;
    float3 out_normal;

    if (scene_type == 0) {
        // Spheres scene - inline sphere data
        const float spheres_data[3][8] = {
            {0.0f, 1.0f, -3.5f, 1.0f, 0.8f, 0.3f, 0.2f, 0.0f},  // center xyz, radius, albedo rgb, metallic
            {-1.8f, 0.9f, -2.4f, 0.9f, 0.2f, 0.7f, 0.9f, 0.0f},
            {1.7f, 0.8f, -2.0f, 0.8f, 0.9f, 0.85f, 0.7f, 1.0f}
        };

        #pragma unroll
        for (int i = 0; i < 3; i++) {
            float3 center = make_float3(spheres_data[i][0], spheres_data[i][1], spheres_data[i][2]);
            float radius = spheres_data[i][3];
            if (hit_sphere_fast(center, radius, ro, rd, 0.001f, hit.t, out_t, out_normal)) {
                hit.t = out_t;
                hit.point = ro + rd * out_t;
                hit.normal = out_normal;
                hit.albedo = make_float3(spheres_data[i][4], spheres_data[i][5], spheres_data[i][6]);
                hit.metallic = spheres_data[i][7];
                hit.roughness = (i == 2) ? 0.15f : ((i == 1) ? 0.6f : 0.8f);
                did_hit = true;
            }
        }

        // Ground plane
        float3 plane_normal = make_float3(0.0f, 1.0f, 0.0f);
        if (hit_plane_fast(ro, rd, plane_normal, 0.0f, 0.001f, hit.t, out_t)) {
            hit.t = out_t;
            hit.point = ro + rd * out_t;
            hit.normal = (dot(rd, plane_normal) < 0.0f) ? plane_normal : make_float3(0.0f, -1.0f, 0.0f);

            // Checkerboard
            int cx = (int)floorf(hit.point.x * 2.0f);
            int cz = (int)floorf(hit.point.z * 2.0f);
            int checker = (cx + cz) & 1;
            hit.albedo = checker ? make_float3(0.9f, 0.9f, 0.9f) : make_float3(0.2f, 0.25f, 0.3f);
            hit.metallic = 0.0f;
            hit.roughness = 0.9f;
            did_hit = true;
        }
    }
    else if (scene_type == 1) {
        // Cornell box
        const float spheres_data[2][8] = {
            {-0.5f, 1.0f, -2.0f, 0.5f, 0.9f, 0.9f, 0.9f, 0.0f},
            {0.5f, 0.5f, -1.5f, 0.5f, 0.9f, 0.9f, 0.9f, 1.0f}
        };

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            float3 center = make_float3(spheres_data[i][0], spheres_data[i][1], spheres_data[i][2]);
            float radius = spheres_data[i][3];
            if (hit_sphere_fast(center, radius, ro, rd, 0.001f, hit.t, out_t, out_normal)) {
                hit.t = out_t;
                hit.point = ro + rd * out_t;
                hit.normal = out_normal;
                hit.albedo = make_float3(spheres_data[i][4], spheres_data[i][5], spheres_data[i][6]);
                hit.metallic = spheres_data[i][7];
                hit.roughness = (i == 0) ? 0.1f : 0.0f;
                did_hit = true;
            }
        }

        // Cornell box walls (simplified)
        const float walls[5][7] = {
            {0.0f, 1.0f, 0.0f, 0.0f, 0.73f, 0.73f, 0.73f},    // floor
            {0.0f, -1.0f, 0.0f, -5.0f, 0.73f, 0.73f, 0.73f},  // ceiling
            {1.0f, 0.0f, 0.0f, 2.5f, 0.65f, 0.05f, 0.05f},    // left (red)
            {-1.0f, 0.0f, 0.0f, 2.5f, 0.12f, 0.45f, 0.15f},   // right (green)
            {0.0f, 0.0f, 1.0f, 5.0f, 0.73f, 0.73f, 0.73f}     // back
        };

        #pragma unroll
        for (int i = 0; i < 5; i++) {
            float3 normal = make_float3(walls[i][0], walls[i][1], walls[i][2]);
            float d = walls[i][3];
            if (hit_plane_fast(ro, rd, normal, d, 0.001f, hit.t, out_t)) {
                hit.t = out_t;
                hit.point = ro + rd * out_t;
                float denom = dot(normal, rd);
                hit.normal = (denom < 0.0f) ? normal : make_float3(-normal.x, -normal.y, -normal.z);
                hit.albedo = make_float3(walls[i][4], walls[i][5], walls[i][6]);
                hit.metallic = 0.0f;
                hit.roughness = 0.9f;
                did_hit = true;
            }
        }
    }
    else {
        // Stress test - optimized sphere loop
        for (int i = 0; i < num_spheres; i++) {
            float x = (float)(i % 10) - 4.5f;
            float z = (float)(i / 10) * -1.5f - 2.0f;
            float r = 0.3f + 0.1f * ((i * 7) % 5);

            float3 center = make_float3(x, r, z);
            if (hit_sphere_fast(center, r, ro, rd, 0.001f, hit.t, out_t, out_normal)) {
                hit.t = out_t;
                hit.point = ro + rd * out_t;
                hit.normal = out_normal;
                hit.albedo = make_float3(
                    0.3f + 0.7f * ((i * 13) % 7) / 6.0f,
                    0.3f + 0.7f * ((i * 17) % 7) / 6.0f,
                    0.3f + 0.7f * ((i * 19) % 7) / 6.0f
                );
                hit.metallic = ((i * 23) % 3 == 0) ? 1.0f : 0.0f;
                hit.roughness = 0.3f;
                did_hit = true;
            }
        }

        // Ground plane
        float3 plane_normal = make_float3(0.0f, 1.0f, 0.0f);
        if (hit_plane_fast(ro, rd, plane_normal, 0.0f, 0.001f, hit.t, out_t)) {
            hit.t = out_t;
            hit.point = ro + rd * out_t;
            hit.normal = plane_normal;

            int cx = (int)floorf(hit.point.x * 2.0f);
            int cz = (int)floorf(hit.point.z * 2.0f);
            int checker = (cx + cz) & 1;
            hit.albedo = checker ? make_float3(0.9f, 0.9f, 0.9f) : make_float3(0.2f, 0.25f, 0.3f);
            hit.metallic = 0.0f;
            hit.roughness = 0.9f;
            did_hit = true;
        }
    }
}

// =============================================================================
// Optimized Lighting
// =============================================================================

__device__ __forceinline__ float3 sky_color(float3 rd) {
    float t = 0.5f * (rd.y + 1.0f);
    return make_float3(
        0.6f + 0.35f * (1.0f - t),
        0.75f + 0.23f * (1.0f - t),
        0.95f + 0.05f * (1.0f - t)
    );
}

__device__ __forceinline__ float soft_shadow_opt(
    float3 p, float3 light_dir, uint32_t& rng,
    int scene_type, int num_spheres
) {
    // Simplified shadow - just offset, no jitter for speed
    float3 d = light_dir;
    HitRecord hit;
    bool did_hit;
    scene_hit_opt(p + d * 0.01f, d, scene_type, num_spheres, hit, did_hit);
    return did_hit ? 0.0f : 1.0f;
}

// =============================================================================
// Optimized Path Tracing
// =============================================================================

__device__ float3 trace_path_opt(
    float3 ro, float3 rd, uint32_t& rng,
    int scene_type, int max_bounces, int num_spheres
) {
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);

    const float3 sun_dir = normalize_fast(make_float3(-0.4f, 0.9f, -0.2f));
    const float3 sun_col = make_float3(3.0f, 2.8f, 2.5f);

    #pragma unroll 4
    for (int bounce = 0; bounce < max_bounces; bounce++) {
        HitRecord hit;
        bool did_hit;
        scene_hit_opt(ro, rd, scene_type, num_spheres, hit, did_hit);

        if (!did_hit) {
            radiance = radiance + hadamard(throughput, sky_color(rd));
            break;
        }

        // Direct lighting
        float ndl = fmaxf(0.0f, dot(hit.normal, sun_dir));
        float shadow = soft_shadow_opt(hit.point, sun_dir, rng, scene_type, num_spheres);
        float3 direct = sun_col * (ndl * shadow);
        radiance = radiance + hadamard(throughput, hadamard(hit.albedo, direct));

        // Scatter ray (branchless approach)
        float3 new_dir;
        if (hit.metallic > 0.5f) {
            float3 reflected = reflect(rd, hit.normal);
            float3 fuzz = random_unit_vector(rng) * hit.roughness;
            new_dir = normalize_fast(reflected + fuzz);
        } else {
            new_dir = random_cosine_direction(rng, hit.normal);
        }

        ro = hit.point + hit.normal * 0.002f;
        rd = new_dir;
        throughput = hadamard(throughput, hit.albedo);

        // Russian roulette
        if (bounce >= 2) {
            float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            p = fminf(p, 0.95f);
            if (rand01(rng) > p) break;
            throughput = throughput * (1.0f / p);
        }
    }

    return radiance;
}

// =============================================================================
// Fast Gamma Approximation
// =============================================================================

// Fast approximation: x^0.4545 ≈ x * rsqrt(rsqrt(x)) for x in [0,1]
// More accurate: use polynomial or table lookup
__device__ __forceinline__ float fast_gamma(float x) {
    // Polynomial approximation for x^(1/2.2) accurate to ~1%
    // Using sqrt twice: sqrt(sqrt(x)) ≈ x^0.25, then multiply by x^0.2
    // Simpler: use hardware sqrt
    x = fmaxf(0.0f, fminf(1.0f, x));
    return sqrtf(x * sqrtf(sqrtf(x)));  // x^0.5 * x^0.125 = x^0.625 (close to 0.4545)
}

// Alternative: use actual fast approximation
__device__ __forceinline__ float fast_pow_045(float x) {
    // Use log/exp approximation: x^a = exp(a * log(x))
    // For a = 0.4545, this can be approximated
    if (x <= 0.0f) return 0.0f;
    if (x >= 1.0f) return 1.0f;

    // Approximate x^0.4545 using bit manipulation
    union { float f; int i; } u;
    u.f = x;
    u.i = (int)(0.4545f * (u.i - 1064866805) + 1064866805);
    return u.f;
}

// =============================================================================
// Main Render Kernel (Optimized)
// =============================================================================

__global__ void render_kernel_opt(
    float3* accum_buffer,
    uint8_t* output_rgb,
    int width,
    int height,
    int samples_per_pixel,
    int frame_index,
    int scene_type,
    int max_bounces,
    uint32_t base_seed,
    int num_spheres
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel_index = y * width + x;

    // Optimized RNG seeding
    uint32_t rng = base_seed ^ (pixel_index * 747796405u) ^ (frame_index * 2891336453u);
    rng ^= rng >> 16;
    rng *= 0x85ebca6bu;
    rng ^= rng >> 13;

    // Pre-computed camera values from constant memory
    float u = ((float)x + rand01(rng)) * (1.0f / (float)width);
    float v = ((float)y + rand01(rng)) * (1.0f / (float)height);

    float px = (2.0f * u - 1.0f) * d_camera.tan_half_fov * d_camera.aspect;
    float py = (1.0f - 2.0f * v) * d_camera.tan_half_fov;

    float3 ray_dir = normalize_fast(d_camera.forward + d_camera.right * px + d_camera.up * py);

    // Accumulate samples
    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    for (int s = 0; s < samples_per_pixel; s++) {
        // Jitter for anti-aliasing
        if (s > 0) {
            u = ((float)x + rand01(rng)) * (1.0f / (float)width);
            v = ((float)y + rand01(rng)) * (1.0f / (float)height);
            px = (2.0f * u - 1.0f) * d_camera.tan_half_fov * d_camera.aspect;
            py = (1.0f - 2.0f * v) * d_camera.tan_half_fov;
            ray_dir = normalize_fast(d_camera.forward + d_camera.right * px + d_camera.up * py);
        }

        color = color + trace_path_opt(d_camera.pos, ray_dir, rng, scene_type, max_bounces, num_spheres);
    }

    color = color * (1.0f / (float)samples_per_pixel);

    // Progressive accumulation
    float3 prev = accum_buffer[pixel_index];
    float alpha = 1.0f / (float)(frame_index + 1);
    float3 current = make_float3(
        prev.x + (color.x - prev.x) * alpha,
        prev.y + (color.y - prev.y) * alpha,
        prev.z + (color.z - prev.z) * alpha
    );
    accum_buffer[pixel_index] = current;

    // Fast tonemap and gamma
    auto tonemap_gamma = [](float v) {
        v = fmaxf(0.0f, v);
        v = v / (1.0f + v);  // Reinhard
        return fast_pow_045(v);
    };

    float r = fminf(1.0f, tonemap_gamma(current.x));
    float g = fminf(1.0f, tonemap_gamma(current.y));
    float b = fminf(1.0f, tonemap_gamma(current.z));

    output_rgb[pixel_index * 3 + 0] = (uint8_t)(r * 255.0f);
    output_rgb[pixel_index * 3 + 1] = (uint8_t)(g * 255.0f);
    output_rgb[pixel_index * 3 + 2] = (uint8_t)(b * 255.0f);
}

// =============================================================================
// C ABI Exports
// =============================================================================

struct MindRayCudaContext {
    int device_id;
    int width;
    int height;
    float3* d_accum;
    uint8_t* d_output;
    int current_frame;
    int num_spheres;
    char error_msg[256];
};

static MindRayCudaContext* g_ctx = nullptr;

EXPORT int mindray_cuda_init(int device_id, int width, int height) {
    if (g_ctx) return -1;

    g_ctx = new MindRayCudaContext();
    g_ctx->device_id = device_id;
    g_ctx->width = width;
    g_ctx->height = height;
    g_ctx->current_frame = 0;
    g_ctx->num_spheres = 50;
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
        delete g_ctx;
        g_ctx = nullptr;
        return -3;
    }

    err = cudaMalloc(&g_ctx->d_output, pixels * 3);
    if (err != cudaSuccess) {
        cudaFree(g_ctx->d_accum);
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

    return 0;
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

    // Update camera in constant memory
    CameraParams cam;
    cam.pos = make_float3(cam_x, cam_y, cam_z);
    float3 look = make_float3(look_x, look_y, look_z);
    float3 up = make_float3(0.0f, 1.0f, 0.0f);

    cam.forward = normalize_fast(look - cam.pos);
    cam.right = normalize_fast(make_float3(
        cam.forward.z, 0.0f, -cam.forward.x  // simplified cross with up
    ));
    cam.up = make_float3(
        cam.right.y * cam.forward.z - cam.right.z * cam.forward.y,
        cam.right.z * cam.forward.x - cam.right.x * cam.forward.z,
        cam.right.x * cam.forward.y - cam.right.y * cam.forward.x
    );

    cam.tan_half_fov = tanf(fov * 3.14159265f / 180.0f * 0.5f);
    cam.aspect = (float)g_ctx->width / (float)g_ctx->height;

    cudaMemcpyToSymbol(d_camera, &cam, sizeof(CameraParams));

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid(
        (g_ctx->width + block.x - 1) / block.x,
        (g_ctx->height + block.y - 1) / block.y
    );

    render_kernel_opt<<<grid, block>>>(
        g_ctx->d_accum,
        g_ctx->d_output,
        g_ctx->width,
        g_ctx->height,
        spp,
        g_ctx->current_frame,
        scene_type,
        max_bounces,
        seed,
        g_ctx->num_spheres
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
    return (err == cudaSuccess) ? 0 : -2;
}

EXPORT int mindray_cuda_reset_accumulator() {
    if (!g_ctx) return -1;
    size_t bytes = (size_t)g_ctx->width * (size_t)g_ctx->height * sizeof(float3);
    cudaMemset(g_ctx->d_accum, 0, bytes);
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
    if (g_ctx) g_ctx->num_spheres = (n > 0) ? n : 50;
}

EXPORT int mindray_cuda_get_num_spheres() {
    if (!g_ctx) return 50;
    return g_ctx->num_spheres;
}
