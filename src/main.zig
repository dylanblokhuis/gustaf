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

    fn modelMatrix(self: RigidBody) m.Mat4 {
        const mat1 = m.Mat4.translationVec3(self.position);
        const mat2 = m.Mat4.fromQuaternion(self.rotation);
        // const mat3 = m.Mat4.scaling(
        //     @floatFromInt(self.voxel_grid_size[0]),
        //     @floatFromInt(self.voxel_grid_size[1]),
        //     @floatFromInt(self.voxel_grid_size[2]),
        // );
        return mat1.multiply(mat2);
    }

    fn occupancy(self: RigidBody, x: i32, y: i32, z: i32) bool {
        if (x < 0 or x >= @as(i32, @intCast(self.voxel_grid_size[0])) or
            y < 0 or y >= @as(i32, @intCast(self.voxel_grid_size[1])) or
            z < 0 or z >= @as(i32, @intCast(self.voxel_grid_size[2])))
        {
            return false;
        }

        const index = x + y * @as(i32, @intCast(self.voxel_grid_size[0])) + z * @as(i32, @intCast(self.voxel_grid_size[0])) * @as(i32, @intCast(self.voxel_grid_size[1]));
        return self.voxels[@intCast(index)] != 0;
    }

    pub fn collidesWith(a: RigidBody, b: RigidBody, a_handle: Bodies.Index, b_handle: Bodies.Index, contacts: *std.ArrayList(ContactPoint)) !void {
        const a_transform = a.modelMatrix();
        const b_transform = b.modelMatrix();
        const a_to_b = b_transform.inverse().multiply(a_transform);

        for (0..a.voxels.len) |index| {
            if (a.voxels[index] == 0) continue;

            const x = index % a.voxel_grid_size[0];
            const y = (index / a.voxel_grid_size[0]) % a.voxel_grid_size[1];
            const z = index / (a.voxel_grid_size[0] * a.voxel_grid_size[1]);
            const pos_a = m.Vec3{
                @floatFromInt(x),
                @floatFromInt(y),
                @floatFromInt(z),
                // } + m.Vec3{ 0.5, 0.5, 0.5 }
            } + m.Vec3{ 0.5, 0.5, 0.5 } - (m.Vec3{ @floatFromInt(a.voxel_grid_size[0]), @floatFromInt(a.voxel_grid_size[1]), @floatFromInt(a.voxel_grid_size[2]) } / m.Vec3{ 2.0, 2.0, 2.0 });

            const pos_a_in_b: m.Vec3 = m.vec.xyz(a_to_b.multiplyVec4(m.Vec4{ pos_a[0], pos_a[1], pos_a[2], 1.0 }));
            // std.debug.print("a_voxel_pos: {d} {d} {d}\n", .{ x, y, z });
            // std.debug.print("pos_in_a: {d} {d} {d}\n", .{ pos_a[0], pos_a[1], pos_a[2] });

            const a_voxel_pos_in_b = pos_a_in_b - m.Vec3{ 0.5, 0.5, 0.5 } + (m.Vec3{ @floatFromInt(b.voxel_grid_size[0]), @floatFromInt(b.voxel_grid_size[1]), @floatFromInt(b.voxel_grid_size[2]) } / m.Vec3{ 2.0, 2.0, 2.0 });
            // std.debug.print("pos_in_b: {d} {d} {d} ({d} {d} {d})\n", .{ pos_a_in_b[0], pos_a_in_b[1], pos_a_in_b[2], b.position[0], b.position[1], b.position[2] });
            // std.debug.print("voxel_pos_in_b: {d} {d} {d}\n", .{ a_voxel_pos_in_b[0], a_voxel_pos_in_b[1], a_voxel_pos_in_b[2] });
            if (a_voxel_pos_in_b[0] < 0 or a_voxel_pos_in_b[0] >= @as(f32, @floatFromInt(b.voxel_grid_size[0])) or
                a_voxel_pos_in_b[1] < 0 or a_voxel_pos_in_b[1] >= @as(f32, @floatFromInt(b.voxel_grid_size[1])) or
                a_voxel_pos_in_b[2] < 0 or a_voxel_pos_in_b[2] >= @as(f32, @floatFromInt(b.voxel_grid_size[2])))
            {
                continue;
            }
            // std.debug.print("passed!\n\n", .{});

            const voxel_coord_in_b: m.UVec3 = @intFromFloat(a_voxel_pos_in_b);
            std.debug.print("voxel_coord_in_b: {}\n", .{voxel_coord_in_b});
            // if (voxel_coord_in_b[0] < 0 or voxel_coord_in_b[0] >= b.voxel_grid_size[0] or
            //     voxel_coord_in_b[1] < 0 or voxel_coord_in_b[1] >= b.voxel_grid_size[1] or
            //     voxel_coord_in_b[2] < 0 or voxel_coord_in_b[2] >= b.voxel_grid_size[2])
            // {
            //     continue;
            // }

            const index_in_b = voxel_coord_in_b[0] + voxel_coord_in_b[1] * b.voxel_grid_size[0] + voxel_coord_in_b[2] * b.voxel_grid_size[0] * b.voxel_grid_size[1];
            if (b.voxels[index_in_b] == 0) {
                continue;
            }

            const contact_point_world: m.Vec3 = m.vec.xyz(a_transform.multiplyVec4(m.Vec4{ pos_a[0], pos_a[1], pos_a[2], 1.0 }));

            const i: i32 = @intCast(voxel_coord_in_b[0]);
            const j: i32 = @intCast(voxel_coord_in_b[1]);
            const k: i32 = @intCast(voxel_coord_in_b[2]);

            std.debug.print("{d} {d} {d}\n", .{ i, j, k });

            const nx = @as(i32, @intFromBool(b.occupancy(i + 1, j, k))) - @as(i32, @intFromBool(b.occupancy(i - 1, j, k)));
            const ny = @as(i32, @intFromBool(b.occupancy(i, j + 1, k))) - @as(i32, @intFromBool(b.occupancy(i, j - 1, k)));
            const nz = @as(i32, @intFromBool(b.occupancy(i, j, k + 1))) - @as(i32, @intFromBool(b.occupancy(i, j, k - 1)));

            var normal_in_b = -m.Vec3{
                @floatFromInt(nx),
                @floatFromInt(ny),
                @floatFromInt(nz),
            };

            if (m.vec.len(normal_in_b) == 0.0) {
                // Handle the case where the normal cannot be computed
                // TODO: its prob inside, so we continue
                // continue;
                // std.debug.print("TODO: normal is default\n", .{});
                normal_in_b = m.Vec3{ 0.0, 1.0, 0.0 };
            } else {
                normal_in_b = m.vec.normalize(normal_in_b);
            }

            // penetration = min_element((sign(normal) - delta) / normal)
            const px = if (std.math.sign(normal_in_b[0]) > 0.0)
                (std.math.sign(normal_in_b[0]) - pos_a_in_b[0]) / normal_in_b[0]
            else
                (std.math.sign(normal_in_b[0] + 1.0) - pos_a_in_b[0]) / normal_in_b[0];

            const py = if (std.math.sign(normal_in_b[1]) > 0.0)
                (std.math.sign(normal_in_b[1]) - pos_a_in_b[1]) / normal_in_b[1]
            else
                (std.math.sign(normal_in_b[1] + 1.0) - pos_a_in_b[1]) / normal_in_b[1];

            const pz = if (std.math.sign(normal_in_b[2]) > 0.0)
                (std.math.sign(normal_in_b[2]) - pos_a_in_b[2]) / normal_in_b[2]
            else
                (std.math.sign(normal_in_b[2] + 1.0) - pos_a_in_b[2]) / normal_in_b[2];

            const penetration = @min(@min(px, py), pz);

            const normal_in_world: m.Vec3 = m.vec.normalize(m.vec.xyz(b_transform.multiplyVec4(m.Vec4{ normal_in_b[0], normal_in_b[1], normal_in_b[2], 1.0 })));
            try contacts.append(.{
                .body_a = a_handle,
                .body_b = b_handle,
                .world_position = contact_point_world,
                .world_normal = normal_in_world,
                .penetration = penetration,
            });
            std.debug.print("[a: {d}] -  {d} {d} {d}\n", .{ a_handle.index, contact_point_world, normal_in_world, penetration });
        }
    }
};
const Bodies = ga.MultiArena(RigidBody, u16, u16);

const ContactPoint = struct {
    body_a: Bodies.Index,
    body_b: Bodies.Index,
    world_position: m.Vec3,
    world_normal: m.Vec3,
    penetration: f32,
};

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
        const contacts = try self.getContactPoints(pairs);

        if (contacts.len > 0) {
            std.debug.print("{d} pairs {d} contacts\n", .{ pairs.len, contacts.len });
        }

        try self.resolveCollisions(contacts);
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

        var iter1 = self.bodies.denseIterator();
        while (iter1.next()) |handle1| {
            var iter2 = self.bodies.denseIterator();
            while (iter2.next()) |handle2| {
                if (handle1.index != handle2.index) {
                    try list.append(.{ handle1, handle2 });
                }
            }
        }

        return list.items;
    }

    fn getContactPoints(self: *Self, pairs: [][2]Bodies.Index) ![]ContactPoint {
        var contacts = std.ArrayList(ContactPoint).init(std.heap.c_allocator);

        for (pairs) |pair| {
            const body1 = self.bodies.getUnchecked(pair[0]);
            const body2 = self.bodies.getUnchecked(pair[1]);
            try RigidBody.collidesWith(body1, body2, pair[0], pair[1], &contacts);
        }

        return contacts.items;
    }

    // use sequential impulse to resolve collisions
    fn resolveCollisions(self: *Self, contacts: []ContactPoint) !void {
        const restitution = 0.5; // Coefficient of restitution (bounciness)
        const friction = 0.5; // Coefficient of friction
        const percent = 0.8; // Penetration correction percentage (typically between 0.2 and 0.8)
        const slop = 0.01; // Penetration allowance (prevents jitter)

        for (contacts) |contact| {
            var body_a = self.bodies.getUnchecked(contact.body_a);
            var body_b = self.bodies.getUnchecked(contact.body_b);
            if (body_a.is_static and body_b.is_static) continue;

            const ra = contact.world_position - body_a.position;
            const rb = contact.world_position - body_b.position;

            // Relative velocity at contact point
            const va = body_a.velocity + m.vec.cross(body_a.angular_velocity, ra);
            const vb = body_b.velocity + m.vec.cross(body_b.angular_velocity, rb);
            const relative_velocity = va - vb;

            // Compute relative velocity along the normal
            const vel_along_normal = m.vec.dot(relative_velocity, contact.world_normal);

            // Do not resolve if velocities are separating
            if (vel_along_normal > 0) continue;

            // Compute impulse scalar
            const inv_mass_a = body_a.inverse_mass;
            const inv_mass_b = body_b.inverse_mass;

            const inv_inertia_a = body_a.inverse_inertia_tensor;
            const inv_inertia_b = body_b.inverse_inertia_tensor;

            var ra_cross_n = m.vec.cross(ra, contact.world_normal);
            var rb_cross_n = m.vec.cross(rb, contact.world_normal);

            const inv_inertia_term_a = m.vec.dot(ra_cross_n, inv_inertia_a.multiplyVec3(ra_cross_n));
            const inv_inertia_term_b = m.vec.dot(rb_cross_n, inv_inertia_b.multiplyVec3(rb_cross_n));

            const denom = inv_mass_a + inv_mass_b + inv_inertia_term_a + inv_inertia_term_b;
            var j = -(1 + restitution) * vel_along_normal;
            j /= denom;

            // Apply impulse
            const impulse = contact.world_normal * @as(m.Vec3, @splat(j));

            if (!body_a.is_static) {
                body_a.velocity += impulse * @as(m.Vec3, @splat(inv_mass_a));
                body_a.angular_velocity += inv_inertia_a.multiplyVec3(m.vec.cross(ra, impulse));
            }

            if (!body_b.is_static) {
                body_b.velocity -= impulse * @as(m.Vec3, @splat(inv_mass_b));
                body_b.angular_velocity -= inv_inertia_b.multiplyVec3(m.vec.cross(rb, impulse));
            }

            // **Friction Impulse**
            // Compute tangent vector
            const relative_velocity_after_impulse = (body_a.velocity + m.vec.cross(body_a.angular_velocity, ra)) -
                (body_b.velocity + m.vec.cross(body_b.angular_velocity, rb));

            var tangent = relative_velocity_after_impulse - contact.world_normal * @as(m.Vec3, @splat(m.vec.dot(relative_velocity_after_impulse, contact.world_normal)));
            const tangent_length = m.vec.len(tangent);
            if (tangent_length > 0.0) {
                tangent /= @as(m.Vec3, @splat(tangent_length));

                // Compute friction impulse scalar
                ra_cross_n = m.vec.cross(ra, tangent);
                rb_cross_n = m.vec.cross(rb, tangent);

                const inv_inertia_tangent_a = m.vec.dot(ra_cross_n, inv_inertia_a.multiplyVec3(ra_cross_n));
                const inv_inertia_tangent_b = m.vec.dot(rb_cross_n, inv_inertia_b.multiplyVec3(rb_cross_n));

                const denom_tangent = inv_mass_a + inv_mass_b + inv_inertia_tangent_a + inv_inertia_tangent_b;

                var jt = -m.vec.dot(relative_velocity_after_impulse, tangent);
                jt /= denom_tangent;

                // Coulomb's law
                const mu = friction;
                const friction_impulse = if (@abs(jt) < j * mu)
                    tangent * @as(m.Vec3, @splat(jt))
                else
                    tangent * @as(m.Vec3, @splat(-j)) * @as(m.Vec3, @splat(mu));

                // Apply friction impulse
                if (!body_a.is_static) {
                    body_a.velocity += friction_impulse * @as(m.Vec3, @splat(inv_mass_a));
                    body_a.angular_velocity += inv_inertia_a.multiplyVec3(m.vec.cross(ra, friction_impulse));
                }

                if (!body_b.is_static) {
                    body_b.velocity -= friction_impulse * @as(m.Vec3, @splat(inv_mass_b));
                    body_b.angular_velocity -= inv_inertia_b.multiplyVec3(m.vec.cross(rb, friction_impulse));
                }
            }

            // **Position Correction**
            // Calculate the total inverse mass
            const total_inv_mass = inv_mass_a + inv_mass_b;

            // Skip if total inverse mass is zero (both bodies are static)
            if (total_inv_mass == 0.0) continue;

            // Compute the penetration depth to correct, considering the slop
            var penetration_correction = (contact.penetration - slop);
            if (penetration_correction < 0.0) penetration_correction = 0.0;
            penetration_correction *= percent / total_inv_mass;

            // Compute the correction vector
            const correction = contact.world_normal * @as(m.Vec3, @splat(penetration_correction));

            // Apply position correction
            if (!body_a.is_static) {
                body_a.position += correction * @as(m.Vec3, @splat(inv_mass_a));
            }
            if (!body_b.is_static) {
                body_b.position -= correction * @as(m.Vec3, @splat(inv_mass_b));
            }

            // Update the bodies in the arena
            try self.bodies.set(contact.body_a, body_a);
            try self.bodies.set(contact.body_b, body_b);
        }
    }
};

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    var world = World{
        .bodies = Bodies.init(allocator),
        .gravity = m.Vec3{ 0.0, -9.81, 0.0 },
    };

    _ = try world.addBody(
        .{ 0, 4, 0 },
        m.Quat.identity(),
        try voxelCube(allocator, .{ 1, 1, 1 }),
        .{ 1, 1, 1 },
        false,
    );
    // _ = try world.addBody(
    //     .{ 0, 40, 0 },
    //     m.Quat.fromEulerAngles(.{ m.radians(45.0), m.radians(45.0), m.radians(0.0) }),
    //     try voxelCube(allocator, .{ 1, 1, 1 }),
    //     .{ 1, 1, 1 },
    //     false,
    // );

    _ = try world.addBody(
        .{ 0, 2, 0 },
        m.Quat.identity(),
        // m.Quat.fromEulerAngles(.{ m.radians(45.0), m.radians(45.0), m.radians(0.0) }),
        try voxelCube(allocator, .{ 2, 2, 2 }),
        .{ 2, 2, 2 },
        true,
    );

    c.InitWindow(600, 450, "Gustaf");
    c.SetTargetFPS(60);

    var camera = c.Camera{
        .position = .{ .x = 0.0, .y = 10.0, .z = 10.0 },
        .target = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .projection = c.CAMERA_PERSPECTIVE,
        .fovy = 45.0,
        .up = .{ .x = 0.0, .y = 1.0, .z = 0.0 },
    };

    const cube_mesh = c.GenMeshCube(1.0, 1.0, 1.0);

    const material = c.LoadMaterialDefault();

    while (!c.WindowShouldClose()) {
        c.UpdateCamera(&camera, c.CAMERA_ORBITAL);

        if (c.IsKeyDown(c.KEY_SPACE)) {
            try world.update(c.GetFrameTime());
        }

        c.BeginDrawing();
        c.ClearBackground(c.RAYWHITE);

        c.BeginMode3D(camera);
        c.SetRandomSeed(0);

        var iter = world.bodies.denseIterator();
        while (iter.next()) |index| {
            const body = world.bodies.getUnchecked(index);

            const mat1 = m.Mat4.translationVec3(body.position);
            const mat2 = m.Mat4.fromQuaternion(body.rotation);
            const mat3 = m.Mat4.scaling(
                @floatFromInt(body.voxel_grid_size[0]),
                @floatFromInt(body.voxel_grid_size[1]),
                @floatFromInt(body.voxel_grid_size[2]),
            );
            const model = mat1.multiply(mat2.multiply(mat3)).transpose();
            material.maps.*.color = .{ .r = @intCast(c.GetRandomValue(0, 255)), .g = @intCast(c.GetRandomValue(0, 255)), .b = @intCast(c.GetRandomValue(0, 255)), .a = 255 };
            c.DrawMesh(cube_mesh, material, .{
                .m0 = model.data[0],
                .m1 = model.data[1],
                .m2 = model.data[2],
                .m3 = model.data[3],
                .m4 = model.data[4],
                .m5 = model.data[5],
                .m6 = model.data[6],
                .m7 = model.data[7],
                .m8 = model.data[8],
                .m9 = model.data[9],
                .m10 = model.data[10],
                .m11 = model.data[11],
                .m12 = model.data[12],
                .m13 = model.data[13],
                .m14 = model.data[14],
                .m15 = model.data[15],
            });
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
