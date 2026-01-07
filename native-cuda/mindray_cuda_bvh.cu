// mindray_cuda_bvh.cu - Optimized CUDA GPU Backend with BVH for Mind Ray
//
// Optimizations applied:
// 1. BVH acceleration for O(log N) intersection scaling
// 2. Fast RNG without rejection sampling (Marsaglia method)
// 3. rsqrtf() instead of sqrtf() + division
// 4. __forceinline__ on all hot path functions
// 5. Fast gamma approximation
// 6. Constant memory for camera parameters
//
// Build: nvcc -O3 -use_fast_math -arch=sm_89 -shared -o mindray_cuda.dll mindray_cuda_bvh.cu -Xcompiler "/MD"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

// =============================================================================
// Export macros
// =============================================================================

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

// =============================================================================
// Vector Math (host + device compatible)
// =============================================================================

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

__host__ __device__ __forceinline__ float dot(float3 a, float3 b) {
#ifdef __CUDA_ARCH__
    return __fmaf_rn(a.x, b.x, __fmaf_rn(a.y, b.y, a.z * b.z));
#else
    return a.x * b.x + a.y * b.y + a.z * b.z;
#endif
}

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
// Fast RNG
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

__device__ __forceinline__ float3 random_unit_vector(uint32_t& state) {
    float z = rand01(state) * 2.0f - 1.0f;
    float phi = rand01(state) * 6.283185307f;
    float r = sqrtf(1.0f - z * z);
    float sp, cp;
    __sincosf(phi, &sp, &cp);
    return make_float3(r * cp, r * sp, z);
}

__device__ __forceinline__ float3 random_cosine_direction(uint32_t& state, float3 normal) {
    float3 rand_dir = random_unit_vector(state);
    if (dot(rand_dir, normal) < 0.0f) {
        rand_dir = make_float3(-rand_dir.x, -rand_dir.y, -rand_dir.z);
    }
    return normalize_fast(normal + rand_dir);
}

// =============================================================================
// Camera (constant memory)
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
// BVH Structures
// =============================================================================

struct AABB {
    float3 min_pt;
    float3 max_pt;
};

struct BVHNode {
    AABB bounds;
    int left_or_first;
    int count;
};

struct BVHSphere {
    float3 center;
    float radius;
    float3 albedo;
    float metallic;
    float roughness;
};

// =============================================================================
// BVH Traversal
// =============================================================================

#define BVH_STACK_SIZE 32

__device__ __forceinline__ bool ray_aabb_intersect(
    float3 ray_origin, float3 ray_dir_inv,
    const AABB& bounds, float t_max
) {
    float tx1 = (bounds.min_pt.x - ray_origin.x) * ray_dir_inv.x;
    float tx2 = (bounds.max_pt.x - ray_origin.x) * ray_dir_inv.x;
    float tmin = fminf(tx1, tx2);
    float tmax = fmaxf(tx1, tx2);

    float ty1 = (bounds.min_pt.y - ray_origin.y) * ray_dir_inv.y;
    float ty2 = (bounds.max_pt.y - ray_origin.y) * ray_dir_inv.y;
    tmin = fmaxf(tmin, fminf(ty1, ty2));
    tmax = fminf(tmax, fmaxf(ty1, ty2));

    float tz1 = (bounds.min_pt.z - ray_origin.z) * ray_dir_inv.z;
    float tz2 = (bounds.max_pt.z - ray_origin.z) * ray_dir_inv.z;
    tmin = fmaxf(tmin, fminf(tz1, tz2));
    tmax = fminf(tmax, fmaxf(tz1, tz2));

    return tmax >= fmaxf(0.0f, tmin) && tmin < t_max;
}

__device__ __forceinline__ bool ray_sphere_intersect(
    float3 ro, float3 rd, float3 center, float radius,
    float t_min, float t_max, float& out_t
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
    return true;
}

// =============================================================================
// Hit Record
// =============================================================================

struct HitRecord {
    float t;
    float3 point;
    float3 normal;
    float3 albedo;
    float metallic;
    float roughness;
};

// =============================================================================
// Scene Intersection with BVH
// =============================================================================

__device__ bool bvh_scene_hit(
    float3 ro, float3 rd,
    const BVHNode* __restrict__ nodes,
    const BVHSphere* __restrict__ spheres,
    int node_count,
    HitRecord& hit
) {
    if (node_count == 0) return false;

    float3 rd_inv = make_float3(1.0f / rd.x, 1.0f / rd.y, 1.0f / rd.z);

    int stack[BVH_STACK_SIZE];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    bool hit_anything = false;
    float closest_t = 1e30f;
    int closest_idx = -1;

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        const BVHNode& node = nodes[node_idx];

        if (!ray_aabb_intersect(ro, rd_inv, node.bounds, closest_t)) {
            continue;
        }

        if (node.count > 0) {  // Leaf node
            for (int i = 0; i < node.count; i++) {
                int prim_idx = node.left_or_first + i;
                const BVHSphere& sphere = spheres[prim_idx];

                float t;
                if (ray_sphere_intersect(ro, rd, sphere.center, sphere.radius, 0.001f, closest_t, t)) {
                    closest_t = t;
                    closest_idx = prim_idx;
                    hit_anything = true;
                }
            }
        } else {  // Inner node
            if (stack_ptr < BVH_STACK_SIZE - 1) {
                stack[stack_ptr++] = node.left_or_first;
                stack[stack_ptr++] = node.left_or_first + 1;
            }
        }
    }

    if (hit_anything) {
        const BVHSphere& sphere = spheres[closest_idx];
        hit.t = closest_t;
        hit.point = ro + rd * closest_t;
        hit.normal = (hit.point - sphere.center) * (1.0f / sphere.radius);
        hit.albedo = sphere.albedo;
        hit.metallic = sphere.metallic;
        hit.roughness = sphere.roughness;
    }

    return hit_anything;
}

// Ground plane intersection
__device__ __forceinline__ bool hit_ground(
    float3 ro, float3 rd, float t_max, HitRecord& hit
) {
    if (fabsf(rd.y) < 1e-6f) return false;
    float t = -ro.y / rd.y;
    if (t < 0.001f || t > t_max) return false;

    hit.t = t;
    hit.point = ro + rd * t;
    hit.normal = make_float3(0.0f, 1.0f, 0.0f);

    int cx = (int)floorf(hit.point.x * 2.0f);
    int cz = (int)floorf(hit.point.z * 2.0f);
    int checker = (cx + cz) & 1;
    hit.albedo = checker ? make_float3(0.9f, 0.9f, 0.9f) : make_float3(0.2f, 0.25f, 0.3f);
    hit.metallic = 0.0f;
    hit.roughness = 0.9f;

    return true;
}

// =============================================================================
// Full Scene Hit (BVH + Ground)
// =============================================================================

__device__ bool scene_hit_bvh(
    float3 ro, float3 rd,
    const BVHNode* __restrict__ nodes,
    const BVHSphere* __restrict__ spheres,
    int node_count,
    HitRecord& hit
) {
    bool hit_anything = false;
    hit.t = 1e30f;

    // Check BVH
    HitRecord bvh_hit;
    if (bvh_scene_hit(ro, rd, nodes, spheres, node_count, bvh_hit)) {
        if (bvh_hit.t < hit.t) {
            hit = bvh_hit;
            hit_anything = true;
        }
    }

    // Check ground
    HitRecord ground_hit;
    if (hit_ground(ro, rd, hit.t, ground_hit)) {
        hit = ground_hit;
        hit_anything = true;
    }

    return hit_anything;
}

// =============================================================================
// Shadow Ray (simplified, no jitter)
// =============================================================================

__device__ __forceinline__ float soft_shadow_bvh(
    float3 p, float3 light_dir,
    const BVHNode* __restrict__ nodes,
    const BVHSphere* __restrict__ spheres,
    int node_count
) {
    HitRecord hit;
    return scene_hit_bvh(p + light_dir * 0.01f, light_dir, nodes, spheres, node_count, hit) ? 0.0f : 1.0f;
}

// =============================================================================
// Sky
// =============================================================================

__device__ __forceinline__ float3 sky_color(float3 rd) {
    float t = 0.5f * (rd.y + 1.0f);
    return make_float3(
        0.6f + 0.35f * (1.0f - t),
        0.75f + 0.23f * (1.0f - t),
        0.95f + 0.05f * (1.0f - t)
    );
}

// =============================================================================
// Path Tracing with BVH
// =============================================================================

__device__ float3 trace_path_bvh(
    float3 ro, float3 rd, uint32_t& rng,
    const BVHNode* __restrict__ nodes,
    const BVHSphere* __restrict__ spheres,
    int node_count, int max_bounces
) {
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);

    const float3 sun_dir = normalize_fast(make_float3(-0.4f, 0.9f, -0.2f));
    const float3 sun_col = make_float3(3.0f, 2.8f, 2.5f);

    for (int bounce = 0; bounce < max_bounces; bounce++) {
        HitRecord hit;
        if (!scene_hit_bvh(ro, rd, nodes, spheres, node_count, hit)) {
            radiance = radiance + hadamard(throughput, sky_color(rd));
            break;
        }

        // Direct lighting
        float ndl = fmaxf(0.0f, dot(hit.normal, sun_dir));
        float shadow = soft_shadow_bvh(hit.point, sun_dir, nodes, spheres, node_count);
        float3 direct = sun_col * (ndl * shadow);
        radiance = radiance + hadamard(throughput, hadamard(hit.albedo, direct));

        // Scatter
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
// Fast Gamma
// =============================================================================

__device__ __forceinline__ float fast_pow_045(float x) {
    if (x <= 0.0f) return 0.0f;
    if (x >= 1.0f) return 1.0f;
    union { float f; int i; } u;
    u.f = x;
    u.i = (int)(0.4545f * (u.i - 1064866805) + 1064866805);
    return u.f;
}

// =============================================================================
// Render Kernel with BVH
// =============================================================================

__global__ void render_kernel_bvh(
    float3* accum_buffer,
    uint8_t* output_rgb,
    const BVHNode* __restrict__ nodes,
    const BVHSphere* __restrict__ spheres,
    int node_count,
    int width, int height,
    int samples_per_pixel,
    int frame_index,
    int max_bounces,
    uint32_t base_seed
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel_index = y * width + x;

    uint32_t rng = base_seed ^ (pixel_index * 747796405u) ^ (frame_index * 2891336453u);
    rng ^= rng >> 16;
    rng *= 0x85ebca6bu;
    rng ^= rng >> 13;

    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    for (int s = 0; s < samples_per_pixel; s++) {
        float u = ((float)x + rand01(rng)) * (1.0f / (float)width);
        float v = ((float)y + rand01(rng)) * (1.0f / (float)height);

        float px = (2.0f * u - 1.0f) * d_camera.tan_half_fov * d_camera.aspect;
        float py = (1.0f - 2.0f * v) * d_camera.tan_half_fov;

        float3 ray_dir = normalize_fast(d_camera.forward + d_camera.right * px + d_camera.up * py);
        color = color + trace_path_bvh(d_camera.pos, ray_dir, rng, nodes, spheres, node_count, max_bounces);
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

    // Tonemap + gamma
    auto tonemap_gamma = [](float v) {
        v = fmaxf(0.0f, v);
        v = v / (1.0f + v);
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
// CPU-side BVH Builder
// =============================================================================

class BVHBuilder {
public:
    std::vector<BVHNode> nodes;
    std::vector<BVHSphere> primitives;

    void build(const std::vector<BVHSphere>& spheres) {
        primitives = spheres;
        nodes.clear();
        nodes.reserve(2 * spheres.size());

        BVHNode root;
        root.left_or_first = 0;
        root.count = (int)primitives.size();
        nodes.push_back(root);
        update_bounds(0);
        subdivide(0);
    }

private:
    void update_bounds(int node_idx) {
        BVHNode& node = nodes[node_idx];
        node.bounds.min_pt = make_float3(1e30f, 1e30f, 1e30f);
        node.bounds.max_pt = make_float3(-1e30f, -1e30f, -1e30f);

        for (int i = 0; i < node.count; i++) {
            const BVHSphere& s = primitives[node.left_or_first + i];
            node.bounds.min_pt.x = fminf(node.bounds.min_pt.x, s.center.x - s.radius);
            node.bounds.min_pt.y = fminf(node.bounds.min_pt.y, s.center.y - s.radius);
            node.bounds.min_pt.z = fminf(node.bounds.min_pt.z, s.center.z - s.radius);
            node.bounds.max_pt.x = fmaxf(node.bounds.max_pt.x, s.center.x + s.radius);
            node.bounds.max_pt.y = fmaxf(node.bounds.max_pt.y, s.center.y + s.radius);
            node.bounds.max_pt.z = fmaxf(node.bounds.max_pt.z, s.center.z + s.radius);
        }
    }

    void subdivide(int node_idx) {
        BVHNode& node = nodes[node_idx];
        if (node.count <= 2) return;

        // Find longest axis
        float3 extent = make_float3(
            node.bounds.max_pt.x - node.bounds.min_pt.x,
            node.bounds.max_pt.y - node.bounds.min_pt.y,
            node.bounds.max_pt.z - node.bounds.min_pt.z
        );

        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > (axis == 0 ? extent.x : extent.y)) axis = 2;

        float split_pos = (axis == 0 ? node.bounds.min_pt.x + node.bounds.max_pt.x :
                          axis == 1 ? node.bounds.min_pt.y + node.bounds.max_pt.y :
                                      node.bounds.min_pt.z + node.bounds.max_pt.z) * 0.5f;

        // Partition
        int first = node.left_or_first;
        int last = first + node.count - 1;
        int i = first;

        while (i <= last) {
            float center = axis == 0 ? primitives[i].center.x :
                          axis == 1 ? primitives[i].center.y :
                                      primitives[i].center.z;
            if (center < split_pos) {
                i++;
            } else {
                std::swap(primitives[i], primitives[last]);
                last--;
            }
        }

        int left_count = i - first;
        if (left_count == 0 || left_count == node.count) return;

        // Create children
        int left_idx = (int)nodes.size();
        BVHNode left_node, right_node;
        left_node.left_or_first = first;
        left_node.count = left_count;
        right_node.left_or_first = i;
        right_node.count = node.count - left_count;

        nodes.push_back(left_node);
        nodes.push_back(right_node);

        node.left_or_first = left_idx;
        node.count = 0;

        update_bounds(left_idx);
        update_bounds(left_idx + 1);
        subdivide(left_idx);
        subdivide(left_idx + 1);
    }
};

// =============================================================================
// Context with BVH
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

    // BVH data
    BVHNode* d_nodes;
    BVHSphere* d_spheres;
    int node_count;
    int sphere_count;
    bool bvh_dirty;

    BVHBuilder builder;
};

static MindRayCudaContext* g_ctx = nullptr;

// =============================================================================
// Build BVH for stress scene
// =============================================================================

static void build_stress_bvh(int num_spheres) {
    if (!g_ctx) return;

    std::vector<BVHSphere> spheres;
    spheres.reserve(num_spheres);

    for (int i = 0; i < num_spheres; i++) {
        BVHSphere s;
        s.center.x = (float)(i % 10) - 4.5f;
        s.center.z = (float)(i / 10) * -1.5f - 2.0f;
        s.radius = 0.3f + 0.1f * ((i * 7) % 5);
        s.center.y = s.radius;

        s.albedo = make_float3(
            0.3f + 0.7f * ((i * 13) % 7) / 6.0f,
            0.3f + 0.7f * ((i * 17) % 7) / 6.0f,
            0.3f + 0.7f * ((i * 19) % 7) / 6.0f
        );
        s.metallic = ((i * 23) % 3 == 0) ? 1.0f : 0.0f;
        s.roughness = 0.3f;

        spheres.push_back(s);
    }

    // Build BVH on CPU
    g_ctx->builder.build(spheres);

    // Free old GPU data
    if (g_ctx->d_nodes) { cudaFree(g_ctx->d_nodes); g_ctx->d_nodes = nullptr; }
    if (g_ctx->d_spheres) { cudaFree(g_ctx->d_spheres); g_ctx->d_spheres = nullptr; }

    // Upload to GPU
    g_ctx->node_count = (int)g_ctx->builder.nodes.size();
    g_ctx->sphere_count = (int)g_ctx->builder.primitives.size();

    cudaMalloc(&g_ctx->d_nodes, g_ctx->node_count * sizeof(BVHNode));
    cudaMalloc(&g_ctx->d_spheres, g_ctx->sphere_count * sizeof(BVHSphere));

    cudaMemcpy(g_ctx->d_nodes, g_ctx->builder.nodes.data(),
               g_ctx->node_count * sizeof(BVHNode), cudaMemcpyHostToDevice);
    cudaMemcpy(g_ctx->d_spheres, g_ctx->builder.primitives.data(),
               g_ctx->sphere_count * sizeof(BVHSphere), cudaMemcpyHostToDevice);

    g_ctx->bvh_dirty = false;
}

// =============================================================================
// C ABI Exports
// =============================================================================

EXPORT int mindray_cuda_init(int device_id, int width, int height) {
    if (g_ctx) return -1;

    g_ctx = new MindRayCudaContext();
    g_ctx->device_id = device_id;
    g_ctx->width = width;
    g_ctx->height = height;
    g_ctx->current_frame = 0;
    g_ctx->num_spheres = 50;
    g_ctx->error_msg[0] = '\0';
    g_ctx->d_nodes = nullptr;
    g_ctx->d_spheres = nullptr;
    g_ctx->node_count = 0;
    g_ctx->sphere_count = 0;
    g_ctx->bvh_dirty = true;

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

    cudaMemset(g_ctx->d_accum, 0, pixels * sizeof(float3));
    return 0;
}

EXPORT int mindray_cuda_render_frame(
    int spp, int scene_type, int max_bounces, uint32_t seed,
    float cam_x, float cam_y, float cam_z,
    float look_x, float look_y, float look_z, float fov
) {
    if (!g_ctx) return -1;

    // Rebuild BVH if needed (stress scene only)
    if (scene_type == 2 && g_ctx->bvh_dirty) {
        build_stress_bvh(g_ctx->num_spheres);
    }

    // Update camera
    CameraParams cam;
    cam.pos = make_float3(cam_x, cam_y, cam_z);
    float3 look = make_float3(look_x, look_y, look_z);
    cam.forward = normalize_fast(look - cam.pos);
    cam.right = normalize_fast(make_float3(cam.forward.z, 0.0f, -cam.forward.x));
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

    render_kernel_bvh<<<grid, block>>>(
        g_ctx->d_accum,
        g_ctx->d_output,
        g_ctx->d_nodes,
        g_ctx->d_spheres,
        g_ctx->node_count,
        g_ctx->width,
        g_ctx->height,
        spp,
        g_ctx->current_frame,
        max_bounces,
        seed
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        snprintf(g_ctx->error_msg, 256, "Kernel launch failed: %s", cudaGetErrorString(err));
        return -2;
    }

    cudaDeviceSynchronize();
    g_ctx->current_frame++;
    return 0;
}

EXPORT int mindray_cuda_copy_output(uint8_t* host_buffer) {
    if (!g_ctx) return -1;
    size_t bytes = (size_t)g_ctx->width * (size_t)g_ctx->height * 3;
    cudaMemcpy(host_buffer, g_ctx->d_output, bytes, cudaMemcpyDeviceToHost);
    return 0;
}

// Batch render: renders N frames without CPU-GPU sync between frames
// Syncs only once at the end for maximum throughput
EXPORT int mindray_cuda_render_batch(
    int num_frames, int spp, int scene_type, int max_bounces, uint32_t seed,
    float cam_x, float cam_y, float cam_z,
    float look_x, float look_y, float look_z, float fov
) {
    if (!g_ctx) return -1;

    // Rebuild BVH if needed (stress scene only)
    if (scene_type == 2 && g_ctx->bvh_dirty) {
        build_stress_bvh(g_ctx->num_spheres);
    }

    // Update camera once (static camera for batch)
    CameraParams cam;
    cam.pos = make_float3(cam_x, cam_y, cam_z);
    float3 look = make_float3(look_x, look_y, look_z);
    cam.forward = normalize_fast(look - cam.pos);
    cam.right = normalize_fast(make_float3(cam.forward.z, 0.0f, -cam.forward.x));
    cam.up = make_float3(
        cam.right.y * cam.forward.z - cam.right.z * cam.forward.y,
        cam.right.z * cam.forward.x - cam.right.x * cam.forward.z,
        cam.right.x * cam.forward.y - cam.right.y * cam.forward.x
    );
    cam.tan_half_fov = tanf(fov * 3.14159265f / 180.0f * 0.5f);
    cam.aspect = (float)g_ctx->width / (float)g_ctx->height;

    cudaMemcpyToSymbol(d_camera, &cam, sizeof(CameraParams));

    dim3 block(16, 16);
    dim3 grid(
        (g_ctx->width + block.x - 1) / block.x,
        (g_ctx->height + block.y - 1) / block.y
    );

    // Launch all frames without sync between them
    for (int i = 0; i < num_frames; i++) {
        render_kernel_bvh<<<grid, block>>>(
            g_ctx->d_accum,
            g_ctx->d_output,
            g_ctx->d_nodes,
            g_ctx->d_spheres,
            g_ctx->node_count,
            g_ctx->width,
            g_ctx->height,
            spp,
            g_ctx->current_frame,
            max_bounces,
            seed
        );
        g_ctx->current_frame++;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        snprintf(g_ctx->error_msg, 256, "Batch kernel launch failed: %s", cudaGetErrorString(err));
        return -2;
    }

    // Single sync at end of batch
    cudaDeviceSynchronize();
    return 0;
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
    if (g_ctx->d_nodes) cudaFree(g_ctx->d_nodes);
    if (g_ctx->d_spheres) cudaFree(g_ctx->d_spheres);
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
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) return -1;
    strncpy(name_buffer, prop.name, buffer_size - 1);
    name_buffer[buffer_size - 1] = '\0';
    return 0;
}

EXPORT int mindray_cuda_get_frame_count() {
    return g_ctx ? g_ctx->current_frame : 0;
}

EXPORT void mindray_cuda_set_num_spheres(int n) {
    if (g_ctx) {
        int new_count = (n > 0) ? n : 50;
        if (new_count != g_ctx->num_spheres) {
            g_ctx->num_spheres = new_count;
            g_ctx->bvh_dirty = true;
        }
    }
}

EXPORT int mindray_cuda_get_num_spheres() {
    return g_ctx ? g_ctx->num_spheres : 50;
}
