/**
 * cuda_benchmark.c - Direct CUDA backend benchmark with detailed timing
 *
 * Calls mindray_cuda.dll directly to measure GPU path tracing performance.
 * This bypasses the Mind compiler and tests the raw CUDA kernel speed.
 *
 * Build (from bench/ directory with VS Developer Command Prompt):
 *   cl /O2 cuda_benchmark.c /link /LIBPATH:..\native-cuda mindray_cuda.lib
 *
 * Run:
 *   cuda_benchmark.exe [--width 640] [--height 360] [--spp 64] [--bounces 4]
 *                      [--frames N] [--warmup W] [--no-output] [--timing]
 *                      [--tier-bp]
 *
 * Tier modes:
 *   Tier B (default): Process wall clock timing
 *   Tier BP (--tier-bp): Persistent mode with cold start + steady state
 *
 * Tier BP outputs (--tier-bp):
 *   COLD_START_MS      - Time from process start to first frame complete
 *   STEADY_MS_PER_FRAME - Median per-frame time after warmup
 *   STEADY_P95_MS       - 95th percentile per-frame time
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

/* DLL function declarations */
typedef int (*mindray_cuda_init_fn)(int device_id, int width, int height);
typedef int (*mindray_cuda_render_frame_fn)(int spp, int scene_type, int max_bounces,
    unsigned int seed, float cam_x, float cam_y, float cam_z,
    float look_x, float look_y, float look_z, float fov);
typedef int (*mindray_cuda_copy_output_fn)(unsigned char* buffer);
typedef int (*mindray_cuda_reset_accumulator_fn)(void);
typedef void (*mindray_cuda_free_fn)(void);
typedef int (*mindray_cuda_get_device_count_fn)(void);
typedef int (*mindray_cuda_get_device_name_fn)(int id, char* buf, int size);
typedef void (*mindray_cuda_set_num_spheres_fn)(int n);
typedef int (*mindray_cuda_get_num_spheres_fn)(void);
typedef int (*mindray_cuda_render_batch_fn)(int num_frames, int spp, int scene_type, int max_bounces,
    unsigned int seed, float cam_x, float cam_y, float cam_z,
    float look_x, float look_y, float look_z, float fov);

/* Scene types */
#define SCENE_SPHERES 0
#define SCENE_CORNELL 1
#define SCENE_STRESS  2

/* FNV-1a hash for SCENE_HASH (matches CUDA Reference) */
static unsigned int compute_scene_hash(int scene_type, int w, int h, int spp, int bounces, int num_spheres) {
    unsigned int hash = 0x811c9dc5u;
    #define MIX(v) hash ^= (unsigned int)(v); hash *= 0x01000193u
    MIX(scene_type);
    MIX(w);
    MIX(h);
    MIX(spp);
    MIX(bounces);
    MIX(num_spheres);
    MIX(0xA341316Cu);  /* Fixed seed for hash consistency with CUDA Reference */
    #undef MIX
    return hash;
}

/* High-resolution timer */
static double get_time_seconds(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
}

int main(int argc, char* argv[]) {
    /* Default config */
    int width = 640;
    int height = 360;
    int spp = 64;
    int bounces = 4;
    int scene = SCENE_CORNELL;
    int warmup = 10;
    int frames = 60;  /* Total frames including warmup */
    unsigned int seed = 42;
    int num_spheres = 50;
    int no_output = 0;   /* Skip file write */
    int show_timing = 0; /* Show detailed timing breakdown */
    int tier_bp = 0;     /* Tier BP (persistent) mode */

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) width = atoi(argv[++i]);
        else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) height = atoi(argv[++i]);
        else if (strcmp(argv[i], "--spp") == 0 && i + 1 < argc) spp = atoi(argv[++i]);
        else if (strcmp(argv[i], "--bounces") == 0 && i + 1 < argc) bounces = atoi(argv[++i]);
        else if (strcmp(argv[i], "--scene") == 0 && i + 1 < argc) {
            if (strcmp(argv[i+1], "spheres") == 0) scene = SCENE_SPHERES;
            else if (strcmp(argv[i+1], "cornell") == 0) scene = SCENE_CORNELL;
            else if (strcmp(argv[i+1], "stress") == 0) scene = SCENE_STRESS;
            i++;
        }
        else if (strcmp(argv[i], "--spheres") == 0 && i + 1 < argc) num_spheres = atoi(argv[++i]);
        else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-output") == 0) no_output = 1;
        else if (strcmp(argv[i], "--timing") == 0) show_timing = 1;
        else if (strcmp(argv[i], "--tier-bp") == 0) tier_bp = 1;
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: cuda_benchmark.exe [options]\n");
            printf("  --width N       Resolution width (default: 640)\n");
            printf("  --height N      Resolution height (default: 360)\n");
            printf("  --spp N         Samples per pixel (default: 64)\n");
            printf("  --bounces N     Max ray bounces (default: 4)\n");
            printf("  --scene TYPE    spheres|cornell|stress (default: cornell)\n");
            printf("  --spheres N     Number of spheres for stress scene (default: 50)\n");
            printf("  --frames N      Total frames (default: 60)\n");
            printf("  --warmup N      Warmup frames (default: 10)\n");
            printf("  --no-output     Skip PPM file write\n");
            printf("  --timing        Show detailed timing breakdown\n");
            printf("  --tier-bp       Enable Tier BP (persistent) mode\n");
            printf("  --help          Show this help\n");
            return 0;
        }
    }

    /* Timing variables */
    double t_start_total = get_time_seconds();
    double t_dll_start, t_dll_end;
    double t_init_start, t_init_end;
    double t_warmup_start, t_warmup_end;
    double t_render_start, t_render_end;
    double t_read_start, t_read_end;
    double t_write_start, t_write_end;
    double t_end_total;

    printf("=== Mind Ray CUDA Benchmark ===\n\n");

    /* Load DLL (timed) */
    t_dll_start = get_time_seconds();
    HMODULE dll = LoadLibraryA("mindray_cuda.dll");
    if (!dll) {
        dll = LoadLibraryA("../native-cuda/mindray_cuda.dll");
    }
    t_dll_end = get_time_seconds();
    if (!dll) {
        printf("ERROR: Could not load mindray_cuda.dll\n");
        return 1;
    }

    /* Get function pointers */
    mindray_cuda_init_fn cuda_init = (mindray_cuda_init_fn)GetProcAddress(dll, "mindray_cuda_init");
    mindray_cuda_render_frame_fn cuda_render = (mindray_cuda_render_frame_fn)GetProcAddress(dll, "mindray_cuda_render_frame");
    mindray_cuda_copy_output_fn cuda_copy = (mindray_cuda_copy_output_fn)GetProcAddress(dll, "mindray_cuda_copy_output");
    mindray_cuda_reset_accumulator_fn cuda_reset = (mindray_cuda_reset_accumulator_fn)GetProcAddress(dll, "mindray_cuda_reset_accumulator");
    mindray_cuda_free_fn cuda_free = (mindray_cuda_free_fn)GetProcAddress(dll, "mindray_cuda_free");
    mindray_cuda_get_device_count_fn cuda_device_count = (mindray_cuda_get_device_count_fn)GetProcAddress(dll, "mindray_cuda_get_device_count");
    mindray_cuda_get_device_name_fn cuda_device_name = (mindray_cuda_get_device_name_fn)GetProcAddress(dll, "mindray_cuda_get_device_name");
    mindray_cuda_set_num_spheres_fn cuda_set_spheres = (mindray_cuda_set_num_spheres_fn)GetProcAddress(dll, "mindray_cuda_set_num_spheres");
    mindray_cuda_render_batch_fn cuda_render_batch = (mindray_cuda_render_batch_fn)GetProcAddress(dll, "mindray_cuda_render_batch");

    if (!cuda_init || !cuda_render || !cuda_copy || !cuda_reset || !cuda_free) {
        printf("ERROR: Missing DLL exports\n");
        FreeLibrary(dll);
        return 1;
    }

    /* Print device info */
    int device_count = cuda_device_count ? cuda_device_count() : 0;
    printf("CUDA devices: %d\n", device_count);
    if (device_count > 0 && cuda_device_name) {
        char name[256];
        if (cuda_device_name(0, name, 256) == 0) {
            printf("Device 0: %s\n", name);
        }
    }
    printf("\n");

    /* Print config */
    const char* scene_names[] = {"spheres", "cornell", "stress"};
    printf("Config:\n");
    printf("  Resolution: %dx%d\n", width, height);
    printf("  SPP: %d\n", spp);
    printf("  Bounces: %d\n", bounces);
    printf("  Scene: %s\n", scene_names[scene]);
    printf("  Seed: %u\n", seed);
    printf("  Warmup frames: %d\n", warmup);
    printf("  Benchmark frames: %d\n", frames);
    printf("  No output: %s\n", no_output ? "yes" : "no");
    if (scene == SCENE_STRESS) {
        printf("  Spheres: %d\n", num_spheres);
    }

    /* Compute and print SCENE_HASH */
    unsigned int scene_hash = compute_scene_hash(scene, width, height, spp, bounces,
        (scene == SCENE_STRESS) ? num_spheres : 0);
    printf("SCENE_HASH=0x%08X\n\n", scene_hash);

    /* Initialize (timed - includes CUDA context creation, buffer allocation) */
    t_init_start = get_time_seconds();
    int err = cuda_init(0, width, height);
    if (err != 0) {
        printf("ERROR: cuda_init failed with code %d\n", err);
        FreeLibrary(dll);
        return 1;
    }

    /* Set sphere count for stress scene (part of init) */
    if (cuda_set_spheres && scene == SCENE_STRESS) {
        cuda_set_spheres(num_spheres);
    }
    t_init_end = get_time_seconds();

    /* Allocate output buffer */
    unsigned char* output = (unsigned char*)malloc(width * height * 3);
    if (!output) {
        printf("ERROR: malloc failed\n");
        cuda_free();
        FreeLibrary(dll);
        return 1;
    }

    /* Camera params (cornell box default) */
    float cam_x = 0.0f, cam_y = 2.5f, cam_z = 10.0f;
    float look_x = 0.0f, look_y = 2.5f, look_z = 0.0f;
    float fov = 40.0f;

    if (scene == SCENE_SPHERES) {
        cam_x = 0.0f; cam_y = 1.2f; cam_z = 1.2f;
        look_x = 0.0f; look_y = 0.9f; look_z = -2.8f;
        fov = 55.0f;
    } else if (scene == SCENE_STRESS) {
        cam_x = 0.0f; cam_y = 3.0f; cam_z = 12.0f;
        look_x = 0.0f; look_y = 1.0f; look_z = 0.0f;
        fov = 50.0f;
    }

    /* Allocate per-frame timing array for Tier BP */
    int measure_frames = frames - warmup;
    if (measure_frames < 1) measure_frames = 1;
    double* frame_times = (double*)malloc(sizeof(double) * measure_frames);
    double cold_start_ms = 0.0;
    double t_first_frame_end = 0.0;

    /* Warmup + Benchmark in single loop for Tier BP */
    printf("Running %d frames (%d warmup + %d measured)...\n", frames, warmup, measure_frames);
    t_warmup_start = get_time_seconds();

    for (int i = 0; i < frames; i++) {
        double t_frame_start = get_time_seconds();
        cuda_reset();
        cuda_render(spp, scene, bounces, seed + i, cam_x, cam_y, cam_z, look_x, look_y, look_z, fov);
        double t_frame_end = get_time_seconds();

        /* Record cold start (first frame from process start) */
        if (i == 0) {
            t_first_frame_end = t_frame_end;
            cold_start_ms = (t_frame_end - t_start_total) * 1000.0;
        }

        /* Record measured frame times (after warmup) */
        if (i >= warmup) {
            frame_times[i - warmup] = (t_frame_end - t_frame_start) * 1000.0;
        }
    }
    t_warmup_end = get_time_seconds();

    /* Sort frame times for median/P95 calculation */
    for (int i = 0; i < measure_frames - 1; i++) {
        for (int j = i + 1; j < measure_frames; j++) {
            if (frame_times[j] < frame_times[i]) {
                double tmp = frame_times[i];
                frame_times[i] = frame_times[j];
                frame_times[j] = tmp;
            }
        }
    }

    double steady_median_ms = frame_times[measure_frames / 2];
    int p95_idx = (int)(measure_frames * 0.95);
    if (p95_idx >= measure_frames) p95_idx = measure_frames - 1;
    double steady_p95_ms = frame_times[p95_idx];

    t_render_start = t_warmup_start;
    t_render_end = t_warmup_end;
    double render_time = t_render_end - t_render_start;
    double time_per_frame = render_time / frames;

    /* Copy output (timed - readback from GPU) */
    t_read_start = get_time_seconds();
    cuda_copy(output);
    t_read_end = get_time_seconds();

    /* Calculate metrics */
    long long pixels = (long long)width * height;
    long long total_samples = pixels * spp * frames;
    long long total_rays = total_samples * bounces;
    double samples_per_sec = total_samples / render_time;
    double rays_per_sec = total_rays / render_time;

    printf("\n=== Results ===\n");
    printf("Render time: %.3f s\n", render_time);
    printf("Time per frame: %.3f s (%.2f ms)\n", time_per_frame, time_per_frame * 1000.0);
    printf("Samples/sec: %.2f M\n", samples_per_sec / 1e6);
    printf("Rays/sec: %.2f M\n", rays_per_sec / 1e6);

    /* Write PPM output (timed, skipped if --no-output) */
    t_write_start = get_time_seconds();
    if (!no_output) {
        const char* out_path = "out/cuda_bench.ppm";
        FILE* f = fopen(out_path, "wb");
        if (f) {
            fprintf(f, "P6\n%d %d\n255\n", width, height);
            fwrite(output, 1, width * height * 3, f);
            fclose(f);
            printf("\nOutput saved: %s\n", out_path);
        }
    }
    t_write_end = get_time_seconds();
    t_end_total = get_time_seconds();

    /* Calculate timing breakdown */
    double time_dll = (t_dll_end - t_dll_start) * 1000.0;
    double time_init = (t_init_end - t_init_start) * 1000.0;
    double time_warmup = (t_warmup_end - t_warmup_start) * 1000.0;
    double time_render = render_time * 1000.0;
    double time_kernel = time_per_frame * 1000.0;
    double time_read = (t_read_end - t_read_start) * 1000.0;
    double time_write = (t_write_end - t_write_start) * 1000.0;
    double time_total = (t_end_total - t_start_total) * 1000.0;

    /* B1 = full end-to-end (init + warmup + render + readback + write) */
    double time_b1 = time_init + time_warmup + time_render + time_read + time_write;
    /* B2 = steady-state per-frame (render / N + readback / N) - amortized */
    double time_b2_per_frame = time_kernel + (time_read / frames);

    if (show_timing) {
        printf("\n=== Timing Breakdown (ms) ===\n");
        printf("TIME_DLL_MS=%.2f\n", time_dll);
        printf("TIME_INIT_MS=%.2f\n", time_init);
        printf("TIME_WARMUP_MS=%.2f\n", time_warmup);
        printf("TIME_RENDER_MS=%.2f\n", time_render);
        printf("TIME_KERNEL_MS=%.2f (per frame)\n", time_kernel);
        printf("TIME_READ_MS=%.2f\n", time_read);
        printf("TIME_WRITE_MS=%.2f\n", time_write);
        printf("TIME_TOTAL_MS=%.2f\n", time_total);
        printf("\n=== Tier B Metrics ===\n");
        printf("B1_TOTAL_MS=%.2f (init+warmup+render+read+write)\n", time_b1);
        printf("B2_PER_FRAME_MS=%.2f (kernel + read/N, steady-state)\n", time_b2_per_frame);
    }

    /* Always output parseable summary line */
    printf("\nSUMMARY: frames=%d kernel_ms=%.2f b1_ms=%.2f b2_per_frame_ms=%.2f\n",
           frames, time_kernel, time_b1, time_b2_per_frame);

    /* Tier BP contract output */
    if (tier_bp) {
        printf("\n=== Tier BP Contract ===\n");
        printf("ENGINE=Mind-Ray\n");
        printf("TIER=BP\n");
        printf("DEVICE=GPU\n");
        printf("DEVICE_NAME=%s\n", device_count > 0 && cuda_device_name ? "NVIDIA GPU" : "Unknown");
        printf("WIDTH=%d\n", width);
        printf("HEIGHT=%d\n", height);
        printf("SPP=%d\n", spp);
        printf("BOUNCES=%d\n", bounces);
        printf("SPHERES=%d\n", num_spheres);
        printf("SEED=%u\n", seed);
        printf("WARMUP_FRAMES=%d\n", warmup);
        printf("MEASURE_FRAMES=%d\n", measure_frames);
        printf("FRAMES_TOTAL=%d\n", frames);
        printf("COLD_START_MS=%.2f\n", cold_start_ms);
        printf("STEADY_MS_PER_FRAME=%.2f\n", steady_median_ms);
        printf("STEADY_P95_MS=%.2f\n", steady_p95_ms);
        printf("STATUS=complete\n");
    }

    /* Cleanup */
    free(frame_times);
    free(output);
    cuda_free();
    FreeLibrary(dll);

    printf("\nBenchmark complete.\n");
    return 0;
}
