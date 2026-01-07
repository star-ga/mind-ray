/*
 * OptiX Benchmark - Main
 * Matches Mind-Ray benchmark contract v2
 *
 * CLI: --scene <spheres|stress> --spheres N --width W --height H --spp N --bounces N
 * Output: ENGINE, SCENE_HASH, KERNEL_MS_TOTAL, etc.
 */

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "optix_benchmark.h"

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t rc = call;                                                 \
        if (rc != cudaSuccess) {                                               \
            std::cerr << "CUDA Error: " << cudaGetErrorString(rc)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Macro for OptiX error checking
#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            std::cerr << "OptiX Error: " << optixGetErrorName(res)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static char log_buffer[2048];
static size_t log_size = sizeof(log_buffer);

#define OPTIX_CHECK_LOG(call)                                                  \
    do {                                                                       \
        log_size = sizeof(log_buffer);                                         \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            std::cerr << "OptiX Error: " << optixGetErrorName(res)             \
                      << "\nLog: " << log_buffer << std::endl;                 \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// SBT record template
template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

// FNV-1a hash (matches contract exactly)
static uint32_t compute_scene_hash(int scene, int w, int h, int spp, int bounces, int spheres) {
    uint32_t hash = 0x811c9dc5u;
#define MIX(v) hash ^= (uint32_t)(v); hash *= 0x01000193u
    MIX(scene);
    MIX(w);
    MIX(h);
    MIX(spp);
    MIX(bounces);
    MIX(spheres);
    MIX(0xA341316Cu);  // fixed seed constant
#undef MIX
    return hash;
}

// Scene configuration
struct SceneConfig {
    std::string name;
    SceneType type;
    int num_spheres;
    float3 cam_eye;
    float3 cam_lookat;
    float cam_fov;
};

// Generate spheres for scenes
std::vector<SphereData> generate_spheres(SceneType scene, int num_spheres) {
    std::vector<SphereData> spheres;

    if (scene == SCENE_SPHERES) {
        // Ground
        SphereData ground;
        ground.center = make_float3(0.0f, -100.5f, -1.0f);
        ground.radius = 100.0f;
        ground.albedo = make_float3(0.5f, 0.5f, 0.5f);
        ground.material = MAT_DIFFUSE;
        ground.fuzz = 0.0f;
        spheres.push_back(ground);

        // Center sphere (diffuse)
        SphereData center;
        center.center = make_float3(0.0f, 0.0f, -1.2f);
        center.radius = 0.5f;
        center.albedo = make_float3(0.7f, 0.3f, 0.3f);
        center.material = MAT_DIFFUSE;
        center.fuzz = 0.0f;
        spheres.push_back(center);

        // Left sphere (metal)
        SphereData left;
        left.center = make_float3(-1.0f, 0.0f, -1.2f);
        left.radius = 0.5f;
        left.albedo = make_float3(0.8f, 0.8f, 0.8f);
        left.material = MAT_METAL;
        left.fuzz = 0.1f;
        spheres.push_back(left);

        // Right sphere (metal gold)
        SphereData right;
        right.center = make_float3(1.0f, 0.0f, -1.2f);
        right.radius = 0.5f;
        right.albedo = make_float3(0.8f, 0.6f, 0.2f);
        right.material = MAT_METAL;
        right.fuzz = 0.0f;
        spheres.push_back(right);

    } else if (scene == SCENE_STRESS) {
        // Ground
        SphereData ground;
        ground.center = make_float3(0.0f, -100.5f, 0.0f);
        ground.radius = 100.0f;
        ground.albedo = make_float3(0.4f, 0.4f, 0.4f);
        ground.material = MAT_DIFFUSE;
        ground.fuzz = 0.0f;
        spheres.push_back(ground);

        // Grid of spheres (same layout as Mind-Ray)
        int grid_size = (int)ceil(sqrt((float)num_spheres));
        float spacing = 1.2f;
        float start_x = -spacing * (grid_size - 1) / 2.0f;
        float start_z = -spacing * (grid_size - 1) / 2.0f;

        for (int i = 0; i < num_spheres; i++) {
            int gx = i % grid_size;
            int gz = i / grid_size;

            SphereData s;
            s.center = make_float3(
                start_x + gx * spacing,
                0.0f,
                start_z + gz * spacing
            );
            s.radius = 0.4f;

            // Deterministic color based on index (matches Mind-Ray)
            float hue = (float)(i * 137 % 360) / 360.0f;
            // HSV to RGB (simplified)
            float h = hue * 6.0f;
            int hi = (int)h;
            float f = h - hi;
            float q = 1.0f - f;
            float t = f;
            switch (hi % 6) {
                case 0: s.albedo = make_float3(1.0f, t, 0.0f); break;
                case 1: s.albedo = make_float3(q, 1.0f, 0.0f); break;
                case 2: s.albedo = make_float3(0.0f, 1.0f, t); break;
                case 3: s.albedo = make_float3(0.0f, q, 1.0f); break;
                case 4: s.albedo = make_float3(t, 0.0f, 1.0f); break;
                case 5: s.albedo = make_float3(1.0f, 0.0f, q); break;
            }

            // Alternate materials
            s.material = (i % 3 == 0) ? MAT_METAL : MAT_DIFFUSE;
            s.fuzz = (s.material == MAT_METAL) ? 0.1f : 0.0f;

            spheres.push_back(s);
        }
    }

    return spheres;
}

// Get directory of current executable
std::string get_exe_dir() {
#ifdef _WIN32
    char path[MAX_PATH];
    GetModuleFileNameA(NULL, path, MAX_PATH);
    std::string s(path);
    size_t pos = s.find_last_of("\\/");
    return (pos != std::string::npos) ? s.substr(0, pos + 1) : "";
#else
    return "./";
#endif
}

// Load PTX from file (searches in exe directory first)
std::string load_ptx(const char* filename) {
    // Try exe directory first
    std::string exe_path = get_exe_dir() + filename;
    std::ifstream file(exe_path, std::ios::binary);
    if (!file) {
        // Try current directory
        file.open(filename, std::ios::binary);
    }
    if (!file) {
        std::cerr << "ERROR: Could not open PTX file: " << filename << std::endl;
        std::cerr << "Searched: " << exe_path << " and ./" << filename << std::endl;
        exit(1);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [options]\n"
              << "Options:\n"
              << "  --scene <spheres|stress>  Scene to render (default: spheres)\n"
              << "  --spheres N               Number of spheres for stress scene (default: 50)\n"
              << "  --width W                 Image width (default: 640)\n"
              << "  --height H                Image height (default: 360)\n"
              << "  --spp N                   Samples per pixel (default: 64)\n"
              << "  --bounces N               Max bounces (default: 4)\n"
              << "  --frames N                Number of frames (default: 5)\n"
              << "  --output FILE             Output image file (optional)\n"
              << "  --help                    Print this message\n";
    exit(1);
}

int main(int argc, char* argv[]) {
    // Default parameters (match contract)
    std::string scene_name = "spheres";
    int width = 640;
    int height = 360;
    int spp = 64;
    int bounces = 4;
    int num_spheres = 50;
    int frames = 5;
    std::string output_file;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--scene" && i + 1 < argc) {
            scene_name = argv[++i];
        } else if (arg == "--spheres" && i + 1 < argc) {
            num_spheres = atoi(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            width = atoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            height = atoi(argv[++i]);
        } else if (arg == "--spp" && i + 1 < argc) {
            spp = atoi(argv[++i]);
        } else if (arg == "--bounces" && i + 1 < argc) {
            bounces = atoi(argv[++i]);
        } else if (arg == "--frames" && i + 1 < argc) {
            frames = atoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
        }
    }

    // Determine scene type
    SceneType scene_type;
    if (scene_name == "spheres") {
        scene_type = SCENE_SPHERES;
        num_spheres = 3;  // Fixed for spheres scene
    } else if (scene_name == "stress") {
        scene_type = SCENE_STRESS;
    } else if (scene_name == "cornell") {
        scene_type = SCENE_CORNELL;
        num_spheres = 0;  // Cornell box uses triangles
        std::cerr << "ERROR: Cornell scene not implemented in OptiX adapter\n";
        return 1;
    } else {
        std::cerr << "ERROR: Unknown scene: " << scene_name << std::endl;
        return 1;
    }

    // Compute scene hash
    uint32_t scene_hash = compute_scene_hash(scene_type, width, height, spp, bounces, num_spheres);

    // Print engine info
    std::cout << "ENGINE=OptiX SDK Path Tracer" << std::endl;
    std::cout << "SCENE=" << scene_name << std::endl;
    std::cout << "WIDTH=" << width << std::endl;
    std::cout << "HEIGHT=" << height << std::endl;
    std::cout << "SPP=" << spp << std::endl;
    std::cout << "BOUNCES=" << bounces << std::endl;
    std::cout << "SPHERES=" << num_spheres << std::endl;
    std::cout << "SEED=0xDEADBEEF" << std::endl;
    std::cout << "SCENE_HASH=0x" << std::hex << std::uppercase << scene_hash << std::dec << std::endl;
    std::cout << "TIMING_TIER=A" << std::endl;

    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "ERROR: No CUDA devices found" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cerr << "Using GPU: " << prop.name << std::endl;

    // Initialize OptiX
    OPTIX_CHECK(optixInit());

    OptixDeviceContext context = nullptr;
    {
        CUcontext cuCtx = 0;
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = nullptr;
        options.logCallbackLevel = 0;
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
    }

    // Generate scene geometry
    std::vector<SphereData> spheres = generate_spheres(scene_type, num_spheres);
    int total_spheres = (int)spheres.size();

    // Build acceleration structure for spheres
    OptixTraversableHandle gas_handle;
    CUdeviceptr d_gas_output;
    {
        // Upload sphere centers and radii
        std::vector<float3> centers(total_spheres);
        std::vector<float> radii(total_spheres);
        for (int i = 0; i < total_spheres; i++) {
            centers[i] = spheres[i].center;
            radii[i] = spheres[i].radius;
        }

        CUdeviceptr d_centers, d_radii;
        CUDA_CHECK(cudaMalloc((void**)&d_centers, total_spheres * sizeof(float3)));
        CUDA_CHECK(cudaMalloc((void**)&d_radii, total_spheres * sizeof(float)));
        CUDA_CHECK(cudaMemcpy((void*)d_centers, centers.data(), total_spheres * sizeof(float3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)d_radii, radii.data(), total_spheres * sizeof(float), cudaMemcpyHostToDevice));

        OptixBuildInput sphere_input = {};
        sphere_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        sphere_input.sphereArray.vertexBuffers = &d_centers;
        sphere_input.sphereArray.numVertices = total_spheres;
        sphere_input.sphereArray.radiusBuffers = &d_radii;

        uint32_t sphere_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
        sphere_input.sphereArray.flags = sphere_flags;
        sphere_input.sphereArray.numSbtRecords = 1;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &sphere_input, 1, &gas_buffer_sizes));

        CUdeviceptr d_temp_buffer;
        CUDA_CHECK(cudaMalloc((void**)&d_temp_buffer, gas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc((void**)&d_gas_output, gas_buffer_sizes.outputSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(
            context, 0, &accel_options, &sphere_input, 1,
            d_temp_buffer, gas_buffer_sizes.tempSizeInBytes,
            d_gas_output, gas_buffer_sizes.outputSizeInBytes,
            &gas_handle, nullptr, 0
        ));

        CUDA_CHECK(cudaFree((void*)d_temp_buffer));
        CUDA_CHECK(cudaFree((void*)d_centers));
        CUDA_CHECK(cudaFree((void*)d_radii));
    }

    // Create OptiX module from PTX
    OptixModule module = nullptr;
    OptixModule sphere_module = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    {
        OptixModuleCompileOptions module_compile_options = {};

        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues = 4;
        pipeline_compile_options.numAttributeValues = 2;
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

        std::string ptx = load_ptx("optix_benchmark.ptx");
        OPTIX_CHECK_LOG(optixModuleCreate(
            context, &module_compile_options, &pipeline_compile_options,
            ptx.c_str(), ptx.size(), log_buffer, &log_size, &module
        ));

        OptixBuiltinISOptions builtin_is_options = {};
        builtin_is_options.usesMotionBlur = false;
        builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
        OPTIX_CHECK_LOG(optixBuiltinISModuleGet(
            context, &module_compile_options, &pipeline_compile_options,
            &builtin_is_options, &sphere_module
        ));
    }

    // Create program groups
    OptixProgramGroup raygen_pg = nullptr;
    OptixProgramGroup miss_pg = nullptr;
    OptixProgramGroup hitgroup_pg = nullptr;
    {
        OptixProgramGroupOptions pg_options = {};

        OptixProgramGroupDesc raygen_desc = {};
        raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_desc.raygen.module = module;
        raygen_desc.raygen.entryFunctionName = "__raygen__rg";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygen_desc, 1, &pg_options, log_buffer, &log_size, &raygen_pg));

        OptixProgramGroupDesc miss_desc = {};
        miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_desc.miss.module = module;
        miss_desc.miss.entryFunctionName = "__miss__ms";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &miss_desc, 1, &pg_options, log_buffer, &log_size, &miss_pg));

        OptixProgramGroupDesc hitgroup_desc = {};
        hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_desc.hitgroup.moduleCH = module;
        hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        hitgroup_desc.hitgroup.moduleIS = sphere_module;
        hitgroup_desc.hitgroup.entryFunctionNameIS = nullptr;
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hitgroup_desc, 1, &pg_options, log_buffer, &log_size, &hitgroup_pg));
    }

    // Create pipeline
    OptixPipeline pipeline = nullptr;
    {
        OptixProgramGroup pgs[] = {raygen_pg, miss_pg, hitgroup_pg};
        OptixPipelineLinkOptions link_options = {};
        link_options.maxTraceDepth = 1;

        OPTIX_CHECK_LOG(optixPipelineCreate(
            context, &pipeline_compile_options, &link_options,
            pgs, 3, log_buffer, &log_size, &pipeline
        ));

        OptixStackSizes stack_sizes = {};
        for (auto& pg : pgs) {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes, pipeline));
        }

        uint32_t dc_from_trav, dc_from_state, cont_stack;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, 1, 0, 0, &dc_from_trav, &dc_from_state, &cont_stack));
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, dc_from_trav, dc_from_state, cont_stack, 1));
    }

    // Set up shader binding table
    OptixShaderBindingTable sbt = {};
    CUdeviceptr d_raygen_record, d_miss_record, d_hitgroup_record;
    {
        // Raygen record
        RayGenSbtRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg, &rg_sbt));
        CUDA_CHECK(cudaMalloc((void**)&d_raygen_record, sizeof(RayGenSbtRecord)));
        CUDA_CHECK(cudaMemcpy((void*)d_raygen_record, &rg_sbt, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

        // Miss record
        MissSbtRecord ms_sbt;
        ms_sbt.data.bg_color_top = make_float3(0.5f, 0.7f, 1.0f);
        ms_sbt.data.bg_color_bottom = make_float3(1.0f, 1.0f, 1.0f);
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg, &ms_sbt));
        CUDA_CHECK(cudaMalloc((void**)&d_miss_record, sizeof(MissSbtRecord)));
        CUDA_CHECK(cudaMemcpy((void*)d_miss_record, &ms_sbt, sizeof(MissSbtRecord), cudaMemcpyHostToDevice));

        // Hitgroup record
        HitGroupSbtRecord hg_sbt;
        // Upload sphere data
        CUdeviceptr d_spheres;
        CUDA_CHECK(cudaMalloc((void**)&d_spheres, spheres.size() * sizeof(SphereData)));
        CUDA_CHECK(cudaMemcpy((void*)d_spheres, spheres.data(), spheres.size() * sizeof(SphereData), cudaMemcpyHostToDevice));
        hg_sbt.data.spheres = (SphereData*)d_spheres;
        hg_sbt.data.num_spheres = total_spheres;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg, &hg_sbt));
        CUDA_CHECK(cudaMalloc((void**)&d_hitgroup_record, sizeof(HitGroupSbtRecord)));
        CUDA_CHECK(cudaMemcpy((void*)d_hitgroup_record, &hg_sbt, sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice));

        sbt.raygenRecord = d_raygen_record;
        sbt.missRecordBase = d_miss_record;
        sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = d_hitgroup_record;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount = 1;
    }

    // Allocate buffers
    float4* d_accum_buffer;
    uchar4* d_frame_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_accum_buffer, width * height * sizeof(float4)));
    CUDA_CHECK(cudaMalloc((void**)&d_frame_buffer, width * height * sizeof(uchar4)));
    CUDA_CHECK(cudaMemset(d_accum_buffer, 0, width * height * sizeof(float4)));

    // Setup camera (match contract)
    float3 cam_eye, cam_lookat;
    float cam_fov;
    if (scene_type == SCENE_SPHERES) {
        cam_eye = make_float3(0.0f, 1.2f, 1.2f);
        cam_lookat = make_float3(0.0f, 0.9f, -2.8f);
        cam_fov = 55.0f;
    } else {  // STRESS
        cam_eye = make_float3(0.0f, 3.0f, 12.0f);
        cam_lookat = make_float3(0.0f, 1.0f, 0.0f);
        cam_fov = 50.0f;
    }

    // Compute camera basis
    float3 cam_up = make_float3(0.0f, 1.0f, 0.0f);
    float3 cam_w = make_float3(
        cam_lookat.x - cam_eye.x,
        cam_lookat.y - cam_eye.y,
        cam_lookat.z - cam_eye.z
    );
    float w_len = sqrtf(cam_w.x * cam_w.x + cam_w.y * cam_w.y + cam_w.z * cam_w.z);
    cam_w.x /= w_len; cam_w.y /= w_len; cam_w.z /= w_len;

    float3 cam_u = make_float3(
        cam_up.y * cam_w.z - cam_up.z * cam_w.y,
        cam_up.z * cam_w.x - cam_up.x * cam_w.z,
        cam_up.x * cam_w.y - cam_up.y * cam_w.x
    );
    float u_len = sqrtf(cam_u.x * cam_u.x + cam_u.y * cam_u.y + cam_u.z * cam_u.z);
    cam_u.x /= u_len; cam_u.y /= u_len; cam_u.z /= u_len;

    float3 cam_v = make_float3(
        cam_w.y * cam_u.z - cam_w.z * cam_u.y,
        cam_w.z * cam_u.x - cam_w.x * cam_u.z,
        cam_w.x * cam_u.y - cam_w.y * cam_u.x
    );

    float aspect = (float)width / (float)height;
    float theta = cam_fov * 3.14159265f / 180.0f;
    float half_height = tanf(theta / 2.0f);
    float half_width = aspect * half_height;

    cam_u.x *= half_width; cam_u.y *= half_width; cam_u.z *= half_width;
    cam_v.x *= half_height; cam_v.y *= half_height; cam_v.z *= half_height;

    // Setup launch params
    Params params;
    params.accum_buffer = d_accum_buffer;
    params.frame_buffer = d_frame_buffer;
    params.width = width;
    params.height = height;
    params.spp = spp;
    params.max_bounces = bounces;
    params.frame_number = 0;
    params.seed = 0xDEADBEEF;
    params.cam_eye = cam_eye;
    params.cam_u = cam_u;
    params.cam_v = cam_v;
    params.cam_w = cam_w;
    params.handle = gas_handle;

    CUdeviceptr d_params;
    CUDA_CHECK(cudaMalloc((void**)&d_params, sizeof(Params)));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Create stream
    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warmup
    params.frame_number = 0;
    CUDA_CHECK(cudaMemcpy((void*)d_params, &params, sizeof(Params), cudaMemcpyHostToDevice));
    OPTIX_CHECK(optixLaunch(pipeline, stream, d_params, sizeof(Params), &sbt, width, height, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed rendering
    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int f = 0; f < frames; f++) {
        params.frame_number = f;
        CUDA_CHECK(cudaMemcpy((void*)d_params, &params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(pipeline, stream, d_params, sizeof(Params), &sbt, width, height, 1));
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    // Output timing
    float ms_per_frame = kernel_ms / frames;
    double total_samples = (double)width * height * spp * frames;
    double samples_per_sec = total_samples / (kernel_ms / 1000.0) / 1e6;

    std::cout << "KERNEL_MS_TOTAL=" << std::fixed << std::setprecision(2) << kernel_ms << std::endl;
    std::cout << "KERNEL_MS_PER_FRAME=" << std::fixed << std::setprecision(2) << ms_per_frame << std::endl;
    std::cout << "KERNEL_SAMPLES_PER_SEC=" << std::fixed << std::setprecision(1) << samples_per_sec << std::endl;
    std::cout << "Samples/sec: " << std::fixed << std::setprecision(1) << samples_per_sec << "M" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree((void*)d_params));
    CUDA_CHECK(cudaFree(d_accum_buffer));
    CUDA_CHECK(cudaFree(d_frame_buffer));
    CUDA_CHECK(cudaFree((void*)d_raygen_record));
    CUDA_CHECK(cudaFree((void*)d_miss_record));
    CUDA_CHECK(cudaFree((void*)d_hitgroup_record));
    CUDA_CHECK(cudaFree((void*)d_gas_output));

    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_pg));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_pg));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_pg));
    OPTIX_CHECK(optixModuleDestroy(module));
    OPTIX_CHECK(optixModuleDestroy(sphere_module));
    OPTIX_CHECK(optixDeviceContextDestroy(context));

    return 0;
}
