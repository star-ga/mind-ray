// bvh.cuh - GPU-friendly BVH for Mind-Ray
//
// Structure:
// - Flat array layout (no pointers)
// - 32-byte aligned nodes for coalesced access
// - Iterative traversal with explicit stack
//
// Build: CPU-side (outside Tier A timing)
// Traverse: GPU kernel (inside Tier A timing)

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>

// =============================================================================
// AABB (Axis-Aligned Bounding Box)
// =============================================================================

struct AABB {
    float3 min_pt;
    float3 max_pt;

    __host__ __device__ AABB() {
        min_pt = make_float3(1e30f, 1e30f, 1e30f);
        max_pt = make_float3(-1e30f, -1e30f, -1e30f);
    }

    __host__ __device__ AABB(float3 mi, float3 ma) : min_pt(mi), max_pt(ma) {}

    __host__ __device__ void expand(float3 p) {
        min_pt.x = fminf(min_pt.x, p.x);
        min_pt.y = fminf(min_pt.y, p.y);
        min_pt.z = fminf(min_pt.z, p.z);
        max_pt.x = fmaxf(max_pt.x, p.x);
        max_pt.y = fmaxf(max_pt.y, p.y);
        max_pt.z = fmaxf(max_pt.z, p.z);
    }

    __host__ __device__ void expand(const AABB& other) {
        min_pt.x = fminf(min_pt.x, other.min_pt.x);
        min_pt.y = fminf(min_pt.y, other.min_pt.y);
        min_pt.z = fminf(min_pt.z, other.min_pt.z);
        max_pt.x = fmaxf(max_pt.x, other.max_pt.x);
        max_pt.y = fmaxf(max_pt.y, other.max_pt.y);
        max_pt.z = fmaxf(max_pt.z, other.max_pt.z);
    }

    __host__ __device__ float3 center() const {
        return make_float3(
            (min_pt.x + max_pt.x) * 0.5f,
            (min_pt.y + max_pt.y) * 0.5f,
            (min_pt.z + max_pt.z) * 0.5f
        );
    }

    __host__ __device__ float surface_area() const {
        float dx = max_pt.x - min_pt.x;
        float dy = max_pt.y - min_pt.y;
        float dz = max_pt.z - min_pt.z;
        return 2.0f * (dx * dy + dy * dz + dz * dx);
    }
};

// =============================================================================
// BVH Node (32 bytes, GPU-friendly layout)
// =============================================================================

struct BVHNode {
    AABB bounds;           // 24 bytes
    int left_or_first;     // 4 bytes: if leaf: first primitive index; else: left child index
    int count;             // 4 bytes: if leaf: primitive count (>0); else: 0 (inner node)

    __host__ __device__ bool is_leaf() const { return count > 0; }
};

// =============================================================================
// Sphere primitive for BVH
// =============================================================================

struct BVHSphere {
    float3 center;
    float radius;
    float3 albedo;
    float metallic;
    float roughness;
    int original_idx;

    __host__ __device__ AABB get_bounds() const {
        return AABB(
            make_float3(center.x - radius, center.y - radius, center.z - radius),
            make_float3(center.x + radius, center.y + radius, center.z + radius)
        );
    }
};

// =============================================================================
// CPU-side BVH Builder (median split for simplicity, can upgrade to SAH)
// =============================================================================

class BVHBuilder {
public:
    std::vector<BVHNode> nodes;
    std::vector<BVHSphere> primitives;  // Reordered primitives

    void build(const std::vector<BVHSphere>& spheres) {
        primitives = spheres;
        nodes.clear();
        nodes.reserve(2 * spheres.size());  // Upper bound

        // Create root node
        nodes.push_back(BVHNode());
        nodes[0].left_or_first = 0;
        nodes[0].count = (int)primitives.size();
        update_bounds(0);

        // Recursively subdivide
        subdivide(0);
    }

private:
    void update_bounds(int node_idx) {
        BVHNode& node = nodes[node_idx];
        node.bounds = AABB();
        for (int i = 0; i < node.count; i++) {
            node.bounds.expand(primitives[node.left_or_first + i].get_bounds());
        }
    }

    void subdivide(int node_idx) {
        BVHNode& node = nodes[node_idx];

        // Stop if too few primitives
        if (node.count <= 2) return;

        // Find best split axis (longest extent)
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

        // Partition primitives
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
        if (left_count == 0 || left_count == node.count) return;  // Can't split

        // Create child nodes
        int left_idx = (int)nodes.size();
        nodes.push_back(BVHNode());
        nodes.push_back(BVHNode());

        nodes[left_idx].left_or_first = first;
        nodes[left_idx].count = left_count;
        nodes[left_idx + 1].left_or_first = i;
        nodes[left_idx + 1].count = node.count - left_count;

        // Convert current node to inner node
        node.left_or_first = left_idx;
        node.count = 0;

        // Update bounds and recurse
        update_bounds(left_idx);
        update_bounds(left_idx + 1);
        subdivide(left_idx);
        subdivide(left_idx + 1);
    }
};

// =============================================================================
// GPU BVH Traversal (iterative with explicit stack)
// =============================================================================

#define BVH_STACK_SIZE 32

__device__ __forceinline__ bool ray_aabb_intersect(
    float3 ray_origin, float3 ray_dir_inv,
    const AABB& bounds,
    float t_max
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
    float3 ray_origin, float3 ray_dir,
    float3 center, float radius,
    float t_min, float t_max,
    float& out_t
) {
    float3 oc = make_float3(
        ray_origin.x - center.x,
        ray_origin.y - center.y,
        ray_origin.z - center.z
    );

    float a = ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y + ray_dir.z * ray_dir.z;
    float half_b = oc.x * ray_dir.x + oc.y * ray_dir.y + oc.z * ray_dir.z;
    float c = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z - radius * radius;
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

// Main BVH traversal function
__device__ bool bvh_intersect(
    float3 ray_origin, float3 ray_dir,
    const BVHNode* __restrict__ nodes,
    const BVHSphere* __restrict__ primitives,
    float t_min, float t_max,
    int& out_prim_idx, float& out_t
) {
    float3 ray_dir_inv = make_float3(1.0f / ray_dir.x, 1.0f / ray_dir.y, 1.0f / ray_dir.z);

    int stack[BVH_STACK_SIZE];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;  // Start with root

    bool hit_anything = false;
    float closest_t = t_max;
    int closest_prim = -1;

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        const BVHNode& node = nodes[node_idx];

        if (!ray_aabb_intersect(ray_origin, ray_dir_inv, node.bounds, closest_t)) {
            continue;
        }

        if (node.is_leaf()) {
            // Test all primitives in leaf
            for (int i = 0; i < node.count; i++) {
                int prim_idx = node.left_or_first + i;
                const BVHSphere& sphere = primitives[prim_idx];

                float t;
                if (ray_sphere_intersect(ray_origin, ray_dir, sphere.center, sphere.radius, t_min, closest_t, t)) {
                    closest_t = t;
                    closest_prim = prim_idx;
                    hit_anything = true;
                }
            }
        } else {
            // Push children (could order by distance for better performance)
            if (stack_ptr < BVH_STACK_SIZE - 1) {
                stack[stack_ptr++] = node.left_or_first;      // Left child
                stack[stack_ptr++] = node.left_or_first + 1;  // Right child
            }
        }
    }

    out_prim_idx = closest_prim;
    out_t = closest_t;
    return hit_anything;
}

#endif // BVH_CUH
