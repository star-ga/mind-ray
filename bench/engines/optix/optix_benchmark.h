/*
 * OptiX Benchmark - Header
 * Matches Mind-Ray benchmark contract v2
 */

#ifndef OPTIX_BENCHMARK_H
#define OPTIX_BENCHMARK_H

#include <cuda_runtime.h>
#include <stdint.h>

// Scene types matching contract
enum SceneType {
    SCENE_SPHERES = 0,
    SCENE_CORNELL = 1,
    SCENE_STRESS = 2
};

// Material types
enum MaterialType {
    MAT_DIFFUSE = 0,
    MAT_METAL = 1,
    MAT_DIELECTRIC = 2
};

// Sphere data for SBT
struct SphereData {
    float3 center;
    float radius;
    float3 albedo;
    int material;
    float fuzz;  // for metal
};

// Launch parameters
struct Params {
    float4* accum_buffer;
    uchar4* frame_buffer;
    uint32_t width;
    uint32_t height;
    uint32_t spp;
    uint32_t max_bounces;
    uint32_t frame_number;
    uint32_t seed;

    // Camera
    float3 cam_eye;
    float3 cam_u;
    float3 cam_v;
    float3 cam_w;

    // Scene
    OptixTraversableHandle handle;
};

// Ray gen data
struct RayGenData {
    // empty
};

// Miss data
struct MissData {
    float3 bg_color_top;
    float3 bg_color_bottom;
};

// Hit group data
struct HitGroupData {
    SphereData* spheres;
    int num_spheres;
};

#endif // OPTIX_BENCHMARK_H
