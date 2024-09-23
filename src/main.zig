// 3D Voxel Physics Engine
//
// This physics engine handles rigid body dynamics for voxel-based objects in a 3D environment.
// It includes collision detection, body representation, integration, constraint solving, and rendering support.
//
//
//
// Collision Detection
//
// Broad Phase:
// - Utilize Axis-Aligned Bounding Boxes (AABB) or Oriented Bounding Boxes (OBB) to quickly identify potential collision pairs.
// - Implement spatial partitioning techniques such as spatial hashing, octrees, or sweep and prune to optimize collision pair identification.
// - Resolve collisions between OBBs to filter out non-colliding pairs before proceeding to the narrow phase.
//
// Narrow Phase:
// - Perform precise collision detection between voxelized objects.
// - Use the heuristic inspired by Teardown (Dennis Gustaffson):
//   - Transform the corner voxels of one object into the local coordinate space of the other object.
//   - Check if any transformed corner voxel intersects with a voxel of the second object.
//   - If a collision is detected, record the collision point and normal, then stop further checks for this pair.
//   - If no collision is found at the corner, inspect the surrounding voxels to account for possible partial overlaps or missed detections due to voxel discretization.
//
//
//
// Body Representation
//
// Rigid Bodies:
// - Each rigid body is represented by a collection of voxels forming its shape.
// - Properties include:
//   - Mass: Calculated by summing the mass of individual voxels, considering their density or material properties if applicable.
//   - Inertia Tensor: Computed by aggregating the inertia tensors of all constituent voxels relative to the body's center of mass.
//   - Velocity: Both linear and angular velocities to capture movement and rotation.
//   - Position and Orientation: Defined by a transformation matrix or position vector and quaternion for rotation.
//
// Voxel Properties:
// - Each voxel can have additional properties such as material type, friction coefficients, and restitution values to influence physical interactions.
//
//
//
// Integration
//
// Time Integration:
// - Use a numerical integration method to update the state of each rigid body over time.
// - Common methods include:
//   - Explicit Euler: Simple but may suffer from instability in some scenarios.
//   - Semi-Implicit Euler (Symplectic Euler): More stable than explicit Euler and widely used in real-time simulations.
//   - Verlet Integration: Suitable for certain types of simulations but may require additional constraints handling.
//
// Force and Torque Application:
// - Accumulate forces (e.g., gravity, user-applied forces) and torques acting on each rigid body.
// - Apply these to update linear and angular velocities during the integration step.
//
// Collision Response:
// - After integration, detect and resolve collisions to prevent interpenetration.
// - Apply impulse-based responses to adjust velocities based on collision normals and restitution coefficients.
//
//
//
// Constraint Solving
//
// Constraints:
// - Implement constraints to simulate joints, limits, or other physical restrictions between bodies or within a single body.
// - Examples include fixed joints, hinge joints, or distance constraints.
//
// Solver:
// - Sequential Impulse to solve the constraint equations.
// - Ensure stability and convergence by limiting the number of iterations and using warm-starting techniques.
//
//
//
// Continuous Collision Detection (Optional)
//
// Prevent Tunneling:
// - Implement continuous collision detection (CCD) to handle fast-moving voxels that might otherwise tunnel through other objects between frames.
// - Techniques include swept volumes, ray casting, or time-of-impact calculations.
//
//

const std = @import("std");
const c = @cImport({
    @cInclude("raylib.h");
});
const ga = @import("generational-arena");
const m = @import("math.zig");
const Allocator = std.mem.Allocator;

pub const RigidBody = struct {
    mass: f32,
    inverse_mass: f32,
    center_of_mass: m.Vec3,
    inertia_tensor: m.Mat3,
    inverse_inertia_tensor: m.Mat3,
    velocity: m.Vec3,
    angular_velocity: m.Vec3,
    position: m.Vec3,
    rotation: m.Quat,
    voxels: []u8,
    voxel_grid_size: m.UVec3,
    is_static: bool,
};
const Bodies = ga.MultiArena(RigidBody, u16, u16);
pub const World = struct {
    const Self = @This();
    bodies: Bodies,
    gravity: m.Vec3,

    pub fn addBody(self: *Self, position: m.Vec3, rotation: m.Quat, voxels: []u8, voxel_grid_size: m.UVec3, is_static: bool) !Bodies.Index {
        const mass_per_voxel: f32 = 1.0;
        var total_mass: f32 = 0.0;
        var center_of_mass = m.Vec3{ 0.0, 0.0, 0.0 };
        var inertia_tensor = m.Mat3.zero();
        var num_solid_voxels: usize = 0;

        for (0..voxels.len) |index| {
            if (voxels[index] != 0) {
                const x = index % voxel_grid_size[0];
                const y = (index / voxel_grid_size[0]) % voxel_grid_size[1];
                const z = index / (voxel_grid_size[0] * voxel_grid_size[1]);
                const pos = m.Vec3{
                    @floatFromInt(x),
                    @floatFromInt(y),
                    @floatFromInt(z),
                };
                total_mass += mass_per_voxel;
                center_of_mass += pos;
                num_solid_voxels += 1;
            }
        }

        if (num_solid_voxels > 0) {
            center_of_mass /= @as(m.Vec3, @splat(@floatFromInt(num_solid_voxels)));
        } else {
            center_of_mass = @splat(0.0);
        }

        const I3 = m.Mat3.identity();

        // Compute inertia tensor
        for (0..voxels.len) |index| {
            if (voxels[index] != 0) {
                const x = index % voxel_grid_size[0];
                const y = (index / voxel_grid_size[0]) % voxel_grid_size[1];
                const z = index / (voxel_grid_size[0] * voxel_grid_size[1]);
                const pos = m.Vec3{
                    @floatFromInt(x),
                    @floatFromInt(y),
                    @floatFromInt(z),
                };
                const r_i = pos - center_of_mass;
                const r_i_dot_r_i = m.vec.dot(r_i, r_i);
                var outer: m.Mat3 = m.Mat3.zero();
                outer.data = .{
                    r_i[0] * r_i[0], r_i[0] * r_i[1], r_i[0] * r_i[2],
                    r_i[1] * r_i[0], r_i[1] * r_i[1], r_i[1] * r_i[2],
                    r_i[2] * r_i[0], r_i[2] * r_i[1], r_i[2] * r_i[2],
                };

                const contrib = I3.scale(r_i_dot_r_i).add(outer.neg());
                inertia_tensor = inertia_tensor.add(contrib.scale(mass_per_voxel));
            }
        }

        var inv_mass: f32 = 0.0;
        var inv_inertia_tensor = m.Mat3.zero();

        if (!is_static and total_mass != 0.0) {
            inv_mass = 1.0 / total_mass;
            inv_inertia_tensor = m.mat3Inverse(inertia_tensor);
        }

        return try self.bodies.append(.{
            .mass = total_mass,
            .inverse_mass = inv_mass,
            .center_of_mass = center_of_mass,
            .inertia_tensor = inertia_tensor,
            .inverse_inertia_tensor = inv_inertia_tensor,
            .velocity = .{ 0.0, 0.0, 0.0 },
            .angular_velocity = .{ 0.0, 0.0, 0.0 },
            .position = position,
            .rotation = rotation,
            .voxels = voxels,
            .voxel_grid_size = voxel_grid_size,
            .is_static = is_static,
        });
    }

    pub fn removeBody(self: *Self, index: Bodies.Index) ?Bodies.Entry {
        return self.bodies.remove(index);
    }

    pub fn update(self: *Self, dt: f32) !void {
        try self.integrateBodies(dt);
        const pairs = try self.getPairs();

        std.debug.print("{d} pairs\n", .{pairs.len});
    }

    fn integrateBodies(self: *Self, dt: f32) !void {
        var iter = self.bodies.denseIterator();
        while (iter.next()) |handle| {
            var body = self.bodies.getUnchecked(handle);
            if (body.is_static) continue;

            // apply gravity
            const gravity_force = self.gravity * @as(m.Vec3, @splat(body.mass));

            // linear integration
            const acceleration = gravity_force * @as(m.Vec3, @splat(body.inverse_mass));
            body.velocity += acceleration * @as(m.Vec3, @splat(dt));
            body.position += body.velocity * @as(m.Vec3, @splat(dt));

            // Angular integration
            const external_torque = m.Vec3{ 0.0, 0.0, 0.0 };

            // Angular acceleration: α = I⁻¹ * τ
            const angular_acceleration = body.inverse_inertia_tensor.multiplyVec3(external_torque);
            // Update angular velocity: ω(t + Δt) = ω(t) + α * Δt
            body.angular_velocity += angular_acceleration * @as(m.Vec3, @splat(dt));

            // Update rotation
            const omega = body.angular_velocity;
            const omega_mag = m.vec.len(omega);

            if (omega_mag > 0.0) {
                // Calculate the rotation angle for this time step
                const theta = omega_mag * dt;
                // Normalize the angular velocity vector to get the rotation axis
                const axis = omega / @as(m.Vec3, @splat(omega_mag));

                // Create a quaternion representing the rotation over Δt
                const half_theta = theta * 0.5;
                const sin_half_theta = @sin(half_theta);
                const cos_half_theta = @cos(half_theta);

                var delta_rotation = m.Quat{
                    .w = cos_half_theta,
                    .x = axis[0] * sin_half_theta,
                    .y = axis[1] * sin_half_theta,
                    .z = axis[2] * sin_half_theta,
                };
                delta_rotation = delta_rotation.multiply(body.rotation);
                delta_rotation.normalize();

                // Update the body's rotation
                body.rotation = delta_rotation;
            }

            try self.bodies.set(handle, body);
        }
    }

    fn getPairs(self: *Self) ![][2]Bodies.Index {
        // for now, every body collides with every other body
        var list = std.ArrayList([2]Bodies.Index).init(std.heap.c_allocator);
        var iter = self.bodies.denseIterator();
        while (iter.next()) |handle1| {
            var iter2 = self.bodies.denseIterator();
            while (iter2.next()) |handle2| {
                if (handle1.index != handle2.index) {
                    try list.append(.{
                        handle1,
                        handle2,
                    });
                }
            }
        }

        return list.items;
    }
};

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    var world = World{
        .bodies = Bodies.init(allocator),
        .gravity = m.Vec3{ 0.0, -9.81, 0.0 },
    };

    _ = try world.addBody(
        .{ 0, 30, 0 },
        m.Quat.identity(),
        try voxelCube(allocator, .{ 8, 8, 8 }),
        .{ 8, 8, 8 },
        false,
    );
    _ = try world.addBody(
        .{ 0, 0, 0 },
        m.Quat.identity(),
        try voxelCube(allocator, .{ 60, 4, 60 }),
        .{ 60, 4, 60 },
        true,
    );

    c.InitWindow(800, 600, "Gustaf");
    c.SetTargetFPS(60);

    var camera = c.Camera{
        .position = .{ .x = 0.0, .y = 100.0, .z = 100.0 },
        .target = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .projection = c.CAMERA_PERSPECTIVE,
        .fovy = 45.0,
        .up = .{ .x = 0.0, .y = 1.0, .z = 0.0 },
    };

    const cube_model = c.LoadModelFromMesh(c.GenMeshCube(1.0, 1.0, 1.0));

    while (!c.WindowShouldClose()) {
        c.UpdateCamera(&camera, c.CAMERA_ORBITAL);

        try world.update(c.GetFrameTime());

        c.BeginDrawing();
        c.ClearBackground(c.RAYWHITE);

        c.BeginMode3D(camera);
        c.SetRandomSeed(0);

        var iter = world.bodies.denseIterator();
        while (iter.next()) |index| {
            const body = world.bodies.getUnchecked(index);

            const axis, const angle = m.quatToAxisAngle(body.rotation);

            c.DrawModelEx(
                cube_model,
                .{ .x = body.position[0], .y = body.position[1], .z = body.position[2] },
                .{ .x = axis[0], .y = axis[1], .z = axis[2] },
                angle,
                .{
                    .x = @floatFromInt(body.voxel_grid_size[0]),
                    .y = @floatFromInt(body.voxel_grid_size[1]),
                    .z = @floatFromInt(body.voxel_grid_size[2]),
                },
                c.ColorFromNormalized(.{ .x = @as(f32, @floatFromInt(c.GetRandomValue(0, 255))) / 255.0, .y = @as(f32, @floatFromInt(c.GetRandomValue(0, 255))) / 255.0, .z = @as(f32, @floatFromInt(c.GetRandomValue(0, 255))) / 255.0, .w = 1.0 }),
            );
        }
        c.EndMode3D();
        c.EndDrawing();
    }
}

// helpers
fn voxelCube(allocator: Allocator, size: m.UVec3) ![]u8 {
    const volume = size[0] * size[1] * size[2];
    const voxels = try allocator.alloc(u8, volume);
    // 0 is empty, 1 is solid
    @memset(voxels, 1);
    return voxels;
}

fn voxelSphere(allocator: Allocator, radius: f32) ![]u8 {
    const diameter = 2.0 * radius;
    const size: m.UVec3 = @splat(diameter);
    const center = diameter / 2.0;
    const voxels = try voxelCube(allocator, size);

    for (0..voxels.len) |index| {
        const pos: m.Vec3 = @floatFromInt(m.UVec3{
            index % size.x,
            (index / size.x) % size.y,
            index / (size.x * size.y),
        });
        const distance = m.vec.len(pos - center);
        if (distance > radius) {
            voxels[index] = 0;
        }
    }

    return voxels;
}
