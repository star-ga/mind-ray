/*
 * OptiX Benchmark - Device Code
 * Path tracing kernels matching Mind-Ray benchmark contract
 */

#include <optix.h>
#include <cuda_runtime.h>
#include "optix_benchmark.h"

extern "C" {
__constant__ Params params;
}

// PCG random number generator (matches Mind-Ray)
struct PCGState {
    uint64_t state;
    uint64_t inc;
};

__device__ __forceinline__ uint32_t pcg32(PCGState* rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

__device__ __forceinline__ float rnd(PCGState* rng) {
    return (float)pcg32(rng) / 4294967296.0f;
}

__device__ __forceinline__ void pcg_init(PCGState* rng, uint64_t seed, uint64_t seq) {
    rng->state = 0U;
    rng->inc = (seq << 1u) | 1u;
    pcg32(rng);
    rng->state += seed;
    pcg32(rng);
}

// Vector utilities
__device__ __forceinline__ float3 operator*(float t, const float3& v) {
    return make_float3(t * v.x, t * v.y, t * v.z);
}

__device__ __forceinline__ float3 operator*(const float3& v, float t) {
    return make_float3(v.x * t, v.y * t, v.z * t);
}

__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 normalize(const float3& v) {
    float inv_len = rsqrtf(dot(v, v));
    return v * inv_len;
}

__device__ __forceinline__ float length(const float3& v) {
    return sqrtf(dot(v, v));
}

__device__ __forceinline__ float3 reflect(const float3& v, const float3& n) {
    return v - 2.0f * dot(v, n) * n;
}

__device__ __forceinline__ float3 random_in_unit_sphere(PCGState* rng) {
    float3 p;
    do {
        p = make_float3(2.0f * rnd(rng) - 1.0f, 2.0f * rnd(rng) - 1.0f, 2.0f * rnd(rng) - 1.0f);
    } while (dot(p, p) >= 1.0f);
    return p;
}

// Payload for ray tracing
struct Payload {
    float3 color;
    float3 origin;
    float3 direction;
    PCGState* rng;
    int depth;
    bool hit;
    float3 attenuation;
};

// Get payload pointers
__device__ __forceinline__ Payload* getPayload() {
    return reinterpret_cast<Payload*>(optixGetPayload_0());
}

// Ray generation program
extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const uint32_t pixel_idx = idx.y * params.width + idx.x;

    // Initialize RNG (same seed strategy as Mind-Ray)
    PCGState rng;
    uint64_t seed = (uint64_t)params.seed ^ ((uint64_t)pixel_idx << 16) ^ ((uint64_t)params.frame_number << 32);
    pcg_init(&rng, seed, pixel_idx);

    float3 result = make_float3(0.0f, 0.0f, 0.0f);

    for (uint32_t s = 0; s < params.spp; s++) {
        // Jittered sample
        float u = ((float)idx.x + rnd(&rng)) / (float)params.width;
        float v = ((float)idx.y + rnd(&rng)) / (float)params.height;

        // Camera ray
        float3 direction = normalize(params.cam_w + (2.0f * u - 1.0f) * params.cam_u + (2.0f * v - 1.0f) * params.cam_v);
        float3 origin = params.cam_eye;

        // Path tracing
        float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
        float3 color = make_float3(0.0f, 0.0f, 0.0f);

        for (uint32_t bounce = 0; bounce <= params.max_bounces; bounce++) {
            // Trace ray
            uint32_t p0 = 0, p1 = 0, p2 = 0, p3 = 0;
            float3 payload_origin = origin;
            float3 payload_dir = direction;
            float3 payload_atten = make_float3(1.0f, 1.0f, 1.0f);
            int payload_hit = 0;
            PCGState* payload_rng = &rng;

            optixTrace(
                params.handle,
                origin,
                direction,
                0.001f,          // tmin
                1e16f,           // tmax
                0.0f,            // ray time
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0,               // SBT offset
                1,               // SBT stride
                0,               // miss SBT index
                p0, p1, p2, p3
            );

            // Unpack payload
            float3 hit_color;
            hit_color.x = __uint_as_float(p0);
            hit_color.y = __uint_as_float(p1);
            hit_color.z = __uint_as_float(p2);
            int hit = (int)p3;

            if (hit == 0) {
                // Miss - sky gradient
                float t = 0.5f * (direction.y + 1.0f);
                float3 sky = (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
                color = color + throughput * sky;
                break;
            } else if (hit == 1) {
                // Emissive hit
                color = color + throughput * hit_color;
                break;
            } else {
                // Continue path
                throughput = throughput * hit_color;
                // Direction is stored in p0-p2 on scatter hits
            }
        }

        result = result + color;
    }

    result = result * (1.0f / (float)params.spp);

    // Accumulate
    if (params.frame_number > 0) {
        float4 prev = params.accum_buffer[pixel_idx];
        float t = 1.0f / (float)(params.frame_number + 1);
        result = (1.0f - t) * make_float3(prev.x, prev.y, prev.z) + t * result;
    }

    params.accum_buffer[pixel_idx] = make_float4(result.x, result.y, result.z, 1.0f);

    // Gamma correction and output
    result.x = sqrtf(fmaxf(0.0f, fminf(1.0f, result.x)));
    result.y = sqrtf(fmaxf(0.0f, fminf(1.0f, result.y)));
    result.z = sqrtf(fmaxf(0.0f, fminf(1.0f, result.z)));

    params.frame_buffer[pixel_idx] = make_uchar4(
        (unsigned char)(result.x * 255.0f),
        (unsigned char)(result.y * 255.0f),
        (unsigned char)(result.z * 255.0f),
        255
    );
}

// Miss program - sky gradient
extern "C" __global__ void __miss__ms() {
    // Return sky color via payload
    const float3 dir = optixGetWorldRayDirection();
    float t = 0.5f * (dir.y + 1.0f);
    float3 sky = (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);

    optixSetPayload_0(__float_as_uint(sky.x));
    optixSetPayload_1(__float_as_uint(sky.y));
    optixSetPayload_2(__float_as_uint(sky.z));
    optixSetPayload_3(0);  // miss
}

// Closest hit program - sphere intersection
extern "C" __global__ void __closesthit__ch() {
    const HitGroupData* data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const int prim_idx = optixGetPrimitiveIndex();

    // Get sphere data
    const SphereData& sphere = data->spheres[prim_idx];

    // Compute hit point and normal
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float t = optixGetRayTmax();
    const float3 hit_point = ray_orig + t * ray_dir;
    float3 normal = normalize(hit_point - sphere.center);

    // Front face check
    bool front_face = dot(ray_dir, normal) < 0.0f;
    if (!front_face) normal = make_float3(-normal.x, -normal.y, -normal.z);

    // Material response
    float3 attenuation = sphere.albedo;

    if (sphere.material == MAT_DIFFUSE) {
        // Lambertian - return attenuation, signal scatter
        optixSetPayload_0(__float_as_uint(attenuation.x));
        optixSetPayload_1(__float_as_uint(attenuation.y));
        optixSetPayload_2(__float_as_uint(attenuation.z));
        optixSetPayload_3(2);  // scatter hit
    } else if (sphere.material == MAT_METAL) {
        // Metal reflection
        float3 reflected = reflect(normalize(ray_dir), normal);
        optixSetPayload_0(__float_as_uint(attenuation.x));
        optixSetPayload_1(__float_as_uint(attenuation.y));
        optixSetPayload_2(__float_as_uint(attenuation.z));
        optixSetPayload_3(2);  // scatter hit
    } else {
        // Default diffuse
        optixSetPayload_0(__float_as_uint(attenuation.x));
        optixSetPayload_1(__float_as_uint(attenuation.y));
        optixSetPayload_2(__float_as_uint(attenuation.z));
        optixSetPayload_3(2);
    }
}
