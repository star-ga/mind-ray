// mind_gpu_raytrace_demo - standalone CUDA ray tracer demo
//
// What it demonstrates:
// - A real GPU kernel doing ray tracing (sphere + plane, soft shadow, reflections)
// - Deterministic RNG per pixel for reproducibility
// - Simple progressive sampling (spp)
//
// Output:
// - Writes a binary PPM (P6) image to a file.
//
// Build (Windows PowerShell example):
//   $env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
//   & "$env:CUDA_PATH\bin\nvcc.exe" -O3 -std=c++17 -lineinfo -o mind_raytrace.exe cuda\main.cu
//
// Build (Linux):
//   nvcc -O3 -std=c++17 -lineinfo -o mind_raytrace cuda/main.cu
//
// Run:
//   ./mind_raytrace --w 1280 --h 720 --spp 64 --out out.ppm

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>

// ----------------------
// Small math utils
// ----------------------

struct float3x {
    float x, y, z;
};

__host__ __device__ static inline float3x make_f3(float x, float y, float z) { return {x,y,z}; }

__host__ __device__ static inline float3x operator+(float3x a, float3x b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
__host__ __device__ static inline float3x operator-(float3x a, float3x b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__host__ __device__ static inline float3x operator*(float3x a, float b) { return {a.x*b, a.y*b, a.z*b}; }
__host__ __device__ static inline float3x operator*(float b, float3x a) { return a*b; }
__host__ __device__ static inline float3x operator/(float3x a, float b) { return {a.x/b, a.y/b, a.z/b}; }

__host__ __device__ static inline float dot(float3x a, float3x b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

__host__ __device__ static inline float3x cross(float3x a, float3x b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}

__host__ __device__ static inline float len(float3x a) { return sqrtf(dot(a,a)); }

__host__ __device__ static inline float3x normalize(float3x a) {
    float l = len(a);
    return (l > 0.0f) ? (a / l) : make_f3(0,0,0);
}

__host__ __device__ static inline float3x clamp01(float3x a) {
    auto c = [](float v){ return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); };
    return {c(a.x), c(a.y), c(a.z)};
}

__host__ __device__ static inline float3x hadamard(float3x a, float3x b) { return {a.x*b.x, a.y*b.y, a.z*b.z}; }

__host__ __device__ static inline float3x reflect(float3x v, float3x n) {
    return v - n * (2.0f * dot(v, n));
}

// ----------------------
// RNG (xorshift32)
// ----------------------

__device__ static inline uint32_t xorshift32(uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ static inline float rand01(uint32_t &state) {
    // 24-bit mantissa
    uint32_t r = xorshift32(state);
    return (r & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

__device__ static inline float3x random_in_unit_sphere(uint32_t &state) {
    while (true) {
        float3x p = make_f3(rand01(state)*2.f-1.f, rand01(state)*2.f-1.f, rand01(state)*2.f-1.f);
        if (dot(p,p) < 1.0f) return p;
    }
}

// ----------------------
// Scene: spheres + plane
// ----------------------

struct Sphere {
    float3x c;
    float r;
    float3x albedo;
    float metal;   // 0..1
    float rough;   // 0..1
};

struct Hit {
    float t;
    float3x p;
    float3x n;
    float3x albedo;
    float metal;
    float rough;
    int hit;
};

__device__ static inline Hit miss() {
    Hit h{};
    h.t = 1e30f;
    h.hit = 0;
    return h;
}

__device__ static inline Hit hit_sphere(const Sphere &s, float3x ro, float3x rd, float tmin, float tmax) {
    float3x oc = ro - s.c;
    float a = dot(rd, rd);
    float b = dot(oc, rd);
    float c = dot(oc, oc) - s.r*s.r;
    float disc = b*b - a*c;
    if (disc < 0.0f) return miss();
    float sq = sqrtf(disc);
    float t = (-b - sq) / a;
    if (t < tmin || t > tmax) {
        t = (-b + sq) / a;
        if (t < tmin || t > tmax) return miss();
    }
    Hit h{};
    h.t = t;
    h.p = ro + rd * t;
    h.n = normalize(h.p - s.c);
    h.albedo = s.albedo;
    h.metal = s.metal;
    h.rough = s.rough;
    h.hit = 1;
    return h;
}

__device__ static inline Hit hit_plane(float3x ro, float3x rd, float3x n, float d, float tmin, float tmax) {
    // plane: dot(n, p) + d = 0
    float denom = dot(n, rd);
    if (fabsf(denom) < 1e-6f) return miss();
    float t = -(dot(n, ro) + d) / denom;
    if (t < tmin || t > tmax) return miss();
    Hit h{};
    h.t = t;
    h.p = ro + rd * t;
    h.n = (denom < 0.0f) ? n : (n * -1.0f);
    // checker
    float scale = 2.0f;
    int cx = (int)floorf(h.p.x * scale);
    int cz = (int)floorf(h.p.z * scale);
    int checker = (cx + cz) & 1;
    h.albedo = checker ? make_f3(0.9f, 0.9f, 0.9f) : make_f3(0.2f, 0.25f, 0.3f);
    h.metal = 0.0f;
    h.rough = 0.9f;
    h.hit = 1;
    return h;
}

// Scene types: 0=spheres, 1=cornell, 2=stress (matching Mind-Ray)
__device__ static inline Hit scene_hit(float3x ro, float3x rd, int scene_type, int num_spheres) {
    Hit best = miss();
    Hit h;

    if (scene_type == 0) {
        // Spheres scene (matches Mind-Ray exactly)
        Sphere s0{ make_f3(0.0f, 1.0f, -3.5f), 1.0f, make_f3(0.8f, 0.3f, 0.2f), 0.0f, 0.8f };
        Sphere s1{ make_f3(-1.8f, 0.9f, -2.4f), 0.9f, make_f3(0.2f, 0.7f, 0.9f), 0.0f, 0.6f };
        Sphere s2{ make_f3(1.7f, 0.8f, -2.0f), 0.8f, make_f3(0.9f, 0.85f, 0.7f), 1.0f, 0.15f };

        h = hit_sphere(s0, ro, rd, 0.001f, best.t); if (h.hit) best = h;
        h = hit_sphere(s1, ro, rd, 0.001f, best.t); if (h.hit) best = h;
        h = hit_sphere(s2, ro, rd, 0.001f, best.t); if (h.hit) best = h;

        // Ground plane
        h = hit_plane(ro, rd, make_f3(0,1,0), 0.0f, 0.001f, best.t);
        if (h.hit) best = h;
    }
    else if (scene_type == 1) {
        // Cornell box (matches Mind-Ray exactly)
        Sphere s0{ make_f3(-0.5f, 1.0f, -2.0f), 0.5f, make_f3(0.9f, 0.9f, 0.9f), 0.0f, 0.1f };
        Sphere s1{ make_f3(0.5f, 0.5f, -1.5f), 0.5f, make_f3(0.9f, 0.9f, 0.9f), 1.0f, 0.0f };

        h = hit_sphere(s0, ro, rd, 0.001f, best.t); if (h.hit) best = h;
        h = hit_sphere(s1, ro, rd, 0.001f, best.t); if (h.hit) best = h;

        // Cornell box walls (floor, ceiling, left red, right green, back)
        h = hit_plane(ro, rd, make_f3(0,1,0), 0.0f, 0.001f, best.t);
        if (h.hit) { h.albedo = make_f3(0.73f, 0.73f, 0.73f); h.metal = 0; h.rough = 0.9f; best = h; }

        h = hit_plane(ro, rd, make_f3(0,-1,0), -5.0f, 0.001f, best.t);
        if (h.hit) { h.albedo = make_f3(0.73f, 0.73f, 0.73f); h.metal = 0; h.rough = 0.9f; best = h; }

        h = hit_plane(ro, rd, make_f3(1,0,0), 2.5f, 0.001f, best.t);
        if (h.hit) { h.albedo = make_f3(0.65f, 0.05f, 0.05f); h.metal = 0; h.rough = 0.9f; best = h; }

        h = hit_plane(ro, rd, make_f3(-1,0,0), 2.5f, 0.001f, best.t);
        if (h.hit) { h.albedo = make_f3(0.12f, 0.45f, 0.15f); h.metal = 0; h.rough = 0.9f; best = h; }

        h = hit_plane(ro, rd, make_f3(0,0,1), 5.0f, 0.001f, best.t);
        if (h.hit) { h.albedo = make_f3(0.73f, 0.73f, 0.73f); h.metal = 0; h.rough = 0.9f; best = h; }
    }
    else {
        // Stress test: N spheres (configurable via --spheres)
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
            Sphere s{ make_f3(x, r, z), r, color, metal, 0.3f };
            h = hit_sphere(s, ro, rd, 0.001f, best.t);
            if (h.hit) best = h;
        }

        h = hit_plane(ro, rd, make_f3(0,1,0), 0.0f, 0.001f, best.t);
        if (h.hit) best = h;
    }

    return best;
}

// ----------------------
// Lighting
// ----------------------

__device__ static inline float soft_shadow(float3x p, float3x ldir, uint32_t &rng, int scene_type, int num_spheres) {
    // jitter the light direction slightly for soft penumbra
    float3x jitter = random_in_unit_sphere(rng) * 0.02f;
    float3x d = normalize(ldir + jitter);
    Hit h = scene_hit(p + d * 0.01f, d, scene_type, num_spheres);
    return h.hit ? 0.0f : 1.0f;
}

__device__ static inline float3x sky(float3x rd) {
    float t = 0.5f * (rd.y + 1.0f);
    return make_f3(0.6f, 0.75f, 0.95f) * t + make_f3(0.95f, 0.98f, 1.0f) * (1.0f - t);
}

// ----------------------
// Path-ish tracer (few bounces)
// ----------------------

__device__ static inline float3x trace(float3x ro, float3x rd, uint32_t &rng, int scene_type, int max_bounces, int num_spheres) {
    float3x throughput = make_f3(1,1,1);
    float3x radiance = make_f3(0,0,0);

    // simple sun light
    float3x sun_dir = normalize(make_f3(-0.4f, 0.9f, -0.2f));
    float3x sun_col = make_f3(3.0f, 2.8f, 2.5f);

    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        Hit h = scene_hit(ro, rd, scene_type, num_spheres);
        if (!h.hit) {
            radiance = radiance + hadamard(throughput, sky(rd));
            break;
        }

        // direct light (shadowed)
        float ndl = fmaxf(0.0f, dot(h.n, sun_dir));
        float sh = soft_shadow(h.p, sun_dir, rng, scene_type, num_spheres);
        float3x direct = sun_col * (ndl * sh);
        radiance = radiance + hadamard(throughput, hadamard(h.albedo, direct));

        // scatter / reflect
        float3x target;
        if (h.metal > 0.5f) {
            float3x refl = reflect(rd, h.n);
            float3x fuzz = random_in_unit_sphere(rng) * h.rough;
            target = normalize(refl + fuzz);
        } else {
            // cosine-ish hemisphere sampling
            float3x randv = normalize(random_in_unit_sphere(rng));
            target = normalize(h.n + randv);
        }

        ro = h.p + h.n * 0.002f;
        rd = target;

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

// ----------------------
// Kernel
// ----------------------

__global__ void render_kernel(float3 *accum, uint8_t *out_rgb, int w, int h, int spp, int frame, int scene_type, int bounces, int num_spheres) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = (y * w + x);

    // deterministic seed per pixel + frame
    uint32_t rng = 0xA341316Cu ^ (uint32_t)(idx * 747796405u) ^ (uint32_t)(frame * 2891336453u);

    // camera
    float aspect = (float)w / (float)h;
    float3x cam_pos = make_f3(0.0f, 1.2f, 1.2f);
    float3x cam_look = make_f3(0.0f, 0.9f, -2.8f);
    float3x cam_up   = make_f3(0.0f, 1.0f, 0.0f);
    float fov = 55.0f * 3.14159265f / 180.0f;

    float3x forward = normalize(cam_look - cam_pos);
    float3x right   = normalize(cross(forward, cam_up));
    float3x up      = cross(right, forward);

    float3x col = make_f3(0,0,0);

    for (int s = 0; s < spp; ++s) {
        float u = ((float)x + rand01(rng)) / (float)w;
        float v = ((float)y + rand01(rng)) / (float)h;
        float px = (2.0f * u - 1.0f) * tanf(fov * 0.5f) * aspect;
        float py = (1.0f - 2.0f * v) * tanf(fov * 0.5f);
        float3x rd = normalize(forward + right * px + up * py);
        col = col + trace(cam_pos, rd, rng, scene_type, bounces, num_spheres);
    }

    col = col / (float)spp;

    // accumulate (optional) - simple EMA for progressive refinement
    float3 prev = accum[idx];
    float a = 1.0f / (float)(frame + 1);
    float3 cur;
    cur.x = prev.x + (col.x - prev.x) * a;
    cur.y = prev.y + (col.y - prev.y) * a;
    cur.z = prev.z + (col.z - prev.z) * a;
    accum[idx] = cur;

    // tonemap + gamma
    auto tonemap = [](float v) {
        v = fmaxf(0.0f, v);
        v = v / (1.0f + v);
        return powf(v, 1.0f / 2.2f);
    };

    float3x t = make_f3(tonemap(cur.x), tonemap(cur.y), tonemap(cur.z));
    t = clamp01(t);

    out_rgb[idx*3 + 0] = (uint8_t)lrintf(t.x * 255.0f);
    out_rgb[idx*3 + 1] = (uint8_t)lrintf(t.y * 255.0f);
    out_rgb[idx*3 + 2] = (uint8_t)lrintf(t.z * 255.0f);
}

// ----------------------
// Host utilities
// ----------------------

static void die(const char *msg, cudaError_t err = cudaSuccess) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    } else {
        std::fprintf(stderr, "%s\n", msg);
    }
    std::exit(1);
}

static bool str_eq(const char* a, const char* b) {
    return std::strcmp(a,b) == 0;
}

static int parse_scene(const char* name) {
    if (str_eq(name, "spheres")) return 0;
    if (str_eq(name, "cornell")) return 1;
    if (str_eq(name, "stress")) return 2;
    return 0;  // default to spheres
}

static const char* scene_name(int scene_type) {
    switch (scene_type) {
        case 0: return "spheres";
        case 1: return "cornell";
        case 2: return "stress";
        default: return "spheres";
    }
}

// Compute scene hash for verification
static uint32_t compute_scene_hash(int scene_type, int w, int h, int spp, int bounces, int num_spheres) {
    uint32_t hash = 0x811c9dc5u;  // FNV-1a
    auto mix = [&hash](uint32_t v) { hash ^= v; hash *= 0x01000193u; };
    mix(scene_type);
    mix(w);
    mix(h);
    mix(spp);
    mix(bounces);
    mix(num_spheres);
    mix(0xA341316Cu);  // seed
    return hash;
}

int main(int argc, char** argv) {
    int w = 1280;
    int h = 720;
    int spp = 32;
    int frames = 1;
    int bounces = 4;
    int scene_type = 0;
    int num_spheres = 50;  // for stress scene
    const char* out_path = "out.ppm";

    for (int i = 1; i < argc; ++i) {
        if (i + 1 < argc && str_eq(argv[i], "--w")) { w = std::atoi(argv[++i]); continue; }
        if (i + 1 < argc && str_eq(argv[i], "--h")) { h = std::atoi(argv[++i]); continue; }
        if (i + 1 < argc && str_eq(argv[i], "--spp")) { spp = std::atoi(argv[++i]); continue; }
        if (i + 1 < argc && str_eq(argv[i], "--frames")) { frames = std::atoi(argv[++i]); continue; }
        if (i + 1 < argc && str_eq(argv[i], "--bounces")) { bounces = std::atoi(argv[++i]); continue; }
        if (i + 1 < argc && str_eq(argv[i], "--scene")) { scene_type = parse_scene(argv[++i]); continue; }
        if (i + 1 < argc && str_eq(argv[i], "--spheres")) { num_spheres = std::atoi(argv[++i]); continue; }
        if (i + 1 < argc && str_eq(argv[i], "--out")) { out_path = argv[++i]; continue; }
        if (str_eq(argv[i], "--help")) {
            std::printf("cuda_reference options:\n");
            std::printf("  --w N        width (default 1280)\n");
            std::printf("  --h N        height (default 720)\n");
            std::printf("  --spp N      samples per pixel per frame (default 32)\n");
            std::printf("  --frames N   progressive frames (default 1)\n");
            std::printf("  --bounces N  max bounces (default 4)\n");
            std::printf("  --scene S    scene: spheres, cornell, stress (default spheres)\n");
            std::printf("  --spheres N  number of spheres for stress scene (default 50)\n");
            std::printf("  --out FILE   output PPM path (default out.ppm)\n");
            return 0;
        }
        std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
        return 1;
    }

    if (w <= 0 || h <= 0 || spp <= 0 || frames <= 0 || bounces <= 0 || num_spheres <= 0) {
        die("Invalid args: w/h/spp/frames/bounces/spheres must be > 0");
    }

    // pick device 0
    int dev = 0;
    cudaError_t err = cudaSetDevice(dev);
    if (err != cudaSuccess) die("cudaSetDevice failed", err);

    size_t pixels = (size_t)w * (size_t)h;
    size_t accum_bytes = pixels * sizeof(float3);
    size_t out_bytes = pixels * 3;

    float3 *d_accum = nullptr;
    uint8_t *d_out = nullptr;

    err = cudaMalloc(&d_accum, accum_bytes);
    if (err != cudaSuccess) die("cudaMalloc accum failed", err);
    err = cudaMalloc(&d_out, out_bytes);
    if (err != cudaSuccess) die("cudaMalloc out failed", err);

    err = cudaMemset(d_accum, 0, accum_bytes);
    if (err != cudaSuccess) die("cudaMemset accum failed", err);

    dim3 block(16, 16, 1);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);

    // CUDA event timing for kernel-only measurement
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start);

    for (int frame = 0; frame < frames; ++frame) {
        render_kernel<<<grid, block>>>(d_accum, d_out, w, h, spp, frame, scene_type, bounces, num_spheres);
        err = cudaGetLastError();
        if (err != cudaSuccess) die("Kernel launch failed", err);
    }

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop);

    double ms_per_frame = (frames > 0) ? (double)kernel_ms / frames : 0.0;
    double total_samples = (double)w * h * spp * frames;
    double samples_per_sec = (kernel_ms > 0) ? (total_samples / (kernel_ms / 1000.0)) / 1e6 : 0.0;

    uint32_t scene_hash = compute_scene_hash(scene_type, w, h, spp, bounces, num_spheres);

    std::printf("KERNEL_MS_TOTAL=%.3f\n", kernel_ms);
    std::printf("KERNEL_MS_PER_FRAME=%.3f\n", ms_per_frame);
    std::printf("KERNEL_SAMPLES_PER_SEC=%.3f\n", samples_per_sec);
    std::printf("KERNEL_CONFIG width=%d height=%d spp=%d frames=%d bounces=%d spheres=%d seed=0xA341316C scene=%s\n", w, h, spp, frames, bounces, num_spheres, scene_name(scene_type));
    std::printf("SCENE_HASH=0x%08X\n", scene_hash);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    // copy back
    uint8_t *h_out = (uint8_t*)std::malloc(out_bytes);
    if (!h_out) die("malloc failed");

    err = cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) die("cudaMemcpy failed", err);

    // write PPM
    std::FILE* f = std::fopen(out_path, "wb");
    if (!f) die("Failed to open output file");
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);
    std::fwrite(h_out, 1, out_bytes, f);
    std::fclose(f);

    std::free(h_out);
    cudaFree(d_out);
    cudaFree(d_accum);

    std::printf("Wrote %s (%dx%d, spp=%d, frames=%d)\n", out_path, w, h, spp, frames);
    return 0;
}
