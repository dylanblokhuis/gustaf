const zm = @import("zm");
const cross_platform_determinism = false;
const builtin = @import("builtin");
const std = @import("std");
const math = std.math;
const assert = std.debug.assert;
const expect = std.testing.expect;

const cpu_arch = builtin.cpu.arch;
const has_avx = if (cpu_arch == .x86_64) std.Target.x86.featureSetHas(builtin.cpu.features, .avx) else false;
const has_avx512f = if (cpu_arch == .x86_64) std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f) else false;
const has_fma = if (cpu_arch == .x86_64) std.Target.x86.featureSetHas(builtin.cpu.features, .fma) else false;

pub const vec = zm.vec;

pub const UVec3 = @Vector(3, u32);
pub const Vec3 = zm.Vec3f;
pub const Vec4 = zm.Vec4f;
pub const Quat = zm.Quaternionf;
pub const Mat4 = zm.Mat4f;
pub const Mat3 = zm.Mat3f;

fn sin(v: f32) f32 {
    var y = v - math.tau * @round(v * 1.0 / math.tau);

    if (y > 0.5 * math.pi) {
        y = math.pi - y;
    } else if (y < -math.pi * 0.5) {
        y = -math.pi - y;
    }
    const y2 = y * y;

    // 11-degree minimax approximation
    var sinv = mulAdd(@as(f32, -2.3889859e-08), y2, 2.7525562e-06);
    sinv = mulAdd(sinv, y2, -0.00019840874);
    sinv = mulAdd(sinv, y2, 0.0083333310);
    sinv = mulAdd(sinv, y2, -0.16666667);
    return y * mulAdd(sinv, y2, 1.0);
}
fn cos(v: f32) f32 {
    var y = v - math.tau * @round(v * 1.0 / math.tau);

    const sign = blk: {
        if (y > 0.5 * math.pi) {
            y = math.pi - y;
            break :blk @as(f32, -1.0);
        } else if (y < -math.pi * 0.5) {
            y = -math.pi - y;
            break :blk @as(f32, -1.0);
        } else {
            break :blk @as(f32, 1.0);
        }
    };
    const y2 = y * y;

    // 10-degree minimax approximation
    var cosv = mulAdd(@as(f32, -2.6051615e-07), y2, 2.4760495e-05);
    cosv = mulAdd(cosv, y2, -0.0013888378);
    cosv = mulAdd(cosv, y2, 0.041666638);
    cosv = mulAdd(cosv, y2, -0.5);
    return sign * mulAdd(cosv, y2, 1.0);
}
fn sincos(v: f32) [2]f32 {
    var y = v - math.tau * @round(v * 1.0 / math.tau);

    const sign = blk: {
        if (y > 0.5 * math.pi) {
            y = math.pi - y;
            break :blk @as(f32, -1.0);
        } else if (y < -math.pi * 0.5) {
            y = -math.pi - y;
            break :blk @as(f32, -1.0);
        } else {
            break :blk @as(f32, 1.0);
        }
    };
    const y2 = y * y;

    // 11-degree minimax approximation
    var sinv = mulAdd(@as(f32, -2.3889859e-08), y2, 2.7525562e-06);
    sinv = mulAdd(sinv, y2, -0.00019840874);
    sinv = mulAdd(sinv, y2, 0.0083333310);
    sinv = mulAdd(sinv, y2, -0.16666667);
    sinv = y * mulAdd(sinv, y2, 1.0);

    // 10-degree minimax approximation
    var cosv = mulAdd(@as(f32, -2.6051615e-07), y2, 2.4760495e-05);
    cosv = mulAdd(cosv, y2, -0.0013888378);
    cosv = mulAdd(cosv, y2, 0.041666638);
    cosv = mulAdd(cosv, y2, -0.5);
    cosv = sign * mulAdd(cosv, y2, 1.0);

    return .{ sinv, cosv };
}
fn asin(v: f32) f32 {
    const x = @abs(v);
    var omx = 1.0 - x;
    if (omx < 0.0) {
        omx = 0.0;
    }
    const root = @sqrt(omx);

    // 7-degree minimax approximation
    var result = mulAdd(@as(f32, -0.0012624911), x, 0.0066700901);
    result = mulAdd(result, x, -0.0170881256);
    result = mulAdd(result, x, 0.0308918810);
    result = mulAdd(result, x, -0.0501743046);
    result = mulAdd(result, x, 0.0889789874);
    result = mulAdd(result, x, -0.2145988016);
    result = root * mulAdd(result, x, 1.5707963050);

    return if (v >= 0.0) 0.5 * math.pi - result else result - 0.5 * math.pi;
}

pub fn acos(v: f32) f32 {
    const x = @abs(v);
    var omx = 1.0 - x;
    if (omx < 0.0) {
        omx = 0.0;
    }
    const root = @sqrt(omx);

    // 7-degree minimax approximation
    var result = mulAdd(@as(f32, -0.0012624911), x, 0.0066700901);
    result = mulAdd(result, x, -0.0170881256);
    result = mulAdd(result, x, 0.0308918810);
    result = mulAdd(result, x, -0.0501743046);
    result = mulAdd(result, x, 0.0889789874);
    result = mulAdd(result, x, -0.2145988016);
    result = root * mulAdd(result, x, 1.5707963050);

    return if (v >= 0.0) result else math.pi - result;
}

pub fn mat3Inverse(m: Mat3) Mat3 {
    const det = mat3Determinant(m);
    if (det == 0.0) {
        return Mat3.zero();
    }

    const adj = Mat3{
        .data = .{
            // first row
            m.data[4] * m.data[8] - m.data[5] * m.data[7],
            m.data[2] * m.data[7] - m.data[1] * m.data[8],
            m.data[1] * m.data[5] - m.data[2] * m.data[4],
            // second row
            m.data[5] * m.data[6] - m.data[3] * m.data[8],
            m.data[0] * m.data[8] - m.data[2] * m.data[6],
            m.data[2] * m.data[3] - m.data[0] * m.data[5],
            // third row
            m.data[3] * m.data[7] - m.data[4] * m.data[6],
            m.data[1] * m.data[6] - m.data[0] * m.data[7],
            m.data[0] * m.data[4] - m.data[1] * m.data[3],
        },
    };

    return adj.scale(1.0 / det);
}

pub fn mat3Determinant(m: Mat3) f32 {
    return m.data[0] * (m.data[4] * m.data[8] - m.data[5] * m.data[7]) - m.data[1] * (m.data[3] * m.data[8] - m.data[5] * m.data[6]) + m.data[2] * (m.data[3] * m.data[7] - m.data[4] * m.data[6]);
}
pub fn quatToAxisAngle(q: Quat) struct { Vec3, f32 } {
    const angle = 2.0 * acos(q.w);
    return .{ Vec3{ q.x, q.y, q.z }, angle };
}

pub inline fn toRadians(degrees: f32) f32 {
    return degrees * math.pi / 180.0;
}

pub inline fn toDegrees(radians: f32) f32 {
    return radians * 180.0 / math.pi;
}

pub inline fn mulAdd(v0: anytype, v1: anytype, v2: anytype) @TypeOf(v0, v1, v2) {
    const T = @TypeOf(v0, v1, v2);
    if (cross_platform_determinism) {
        return v0 * v1 + v2;
    } else {
        if (cpu_arch == .x86_64 and has_avx and has_fma) {
            return @mulAdd(T, v0, v1, v2);
        } else {
            return v0 * v1 + v2;
        }
    }
}

pub fn quatMulVec3(q: Quat, v: Vec3) Vec3 {
    const qv = Vec3{ q.x, q.y, q.z };
    const uv = vec.cross(qv, v);
    const uuv = vec.cross(qv, uv);
    return v + uv * (Vec3{ 2.0, 2.0, 2.0 } * Vec3{ q.w, q.w, q.w }) + uuv * Vec3{ 2.0, 2.0, 2.0 };
}
