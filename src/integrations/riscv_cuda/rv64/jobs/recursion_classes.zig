//! Install-boundary interface. Construction lives at the executable root;
//! request job creation lives in the narrowly scanned native_node_class adapter.
const std = @import("std");
const CPU = @import("stwo_riscv_cpu_integration");
const G = @import("gpu_native.zig");
const OD = @import("outer_device.zig");
const S2 = @import("native_device_stage2.zig");
const TF = @import("truth_factory.zig");
const W = @import("worker.zig").Worker;
pub const Runtime = @import("../native_node_class.zig");
pub const Leaf = struct {
    wire_words: u32,
    image_words: u32,
    tail_seeds: u32,
    output_words: u32,
    pub fn suffixWords(self: Leaf) usize {
        return @as(usize, self.image_words) + 4 * @as(usize, self.tail_seeds) + self.output_words;
    }
    pub fn regionWords(self: Leaf) !u32 {
        return std.math.cast(u32, @as(u64, self.wire_words) + self.suffixWords()) orelse error.GnNodeChildRegion;
    }
};

/// Borrowed only during class installation. The inventory binds the worker
/// framing independently of the eventual MAIN commitments and shared claims.
pub const RecipeInputs = struct {
    inventory: *const CPU.rv64_gkr_execution.Inventory,
    cpus: []const *const CPU.rv64_services_roster.CpuSetup,
    requests: []const @FieldType(CPU.rv64_om31_execution_wrap.ClassSetup.Cpu, "request"),
    truths: []const CPU.rv64_services_roster.Truth,
    config: CPU.rv64_gkr_worker_protocol.PcsConfig,
    fixed: *const CPU.rv64_gkr_fixed_authority_set.Producer,
    worker: *W,
    capacity: @import("api.zig").SetupCapacity,
    grouped_recursion: bool = false,
};

pub const Classes = struct {
    ctx: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        deinit: *const fn (*anyopaque) void,
        register: *const fn (*anyopaque, u64, *OD.Recursion, []const G.ClassKey, *S2.Factory, []const TF.Request, *TF.Factory, ?RecipeInputs) anyerror!void,
        relayout: *const fn (*anyopaque, *G.ClassCache) anyerror!void,
        warm: *const fn (*anyopaque, []const *W) anyerror!void,
        resource_peak: *const fn (*anyopaque) anyerror!?usize,
        leaf: *const fn (*anyopaque, u64, usize) anyerror!Leaf,
        coefficients: *const fn (*anyopaque, u64, usize, *const CPU.rv64_gkr_execution.Inventory) anyerror![]const u32,
        identity: *const fn (*anyopaque, u64, usize) anyerror![32]u8,
        pinned: *const fn (*anyopaque, u64) anyerror!usize,
    };
    pub fn deinit(self: Classes) void {
        self.vtable.deinit(self.ctx);
    }
    pub fn register(self: Classes, block: u64, registry: *OD.Recursion, native: []const G.ClassKey, stage2: *S2.Factory, truths: []const TF.Request, factory: *TF.Factory, recipes: ?RecipeInputs) !void {
        return self.vtable.register(self.ctx, block, registry, native, stage2, truths, factory, recipes);
    }
    pub fn relayout(self: Classes, cache: *G.ClassCache) !void {
        return self.vtable.relayout(self.ctx, cache);
    }
    pub fn warm(self: Classes, workers: []const *W) !void {
        return self.vtable.warm(self.ctx, workers);
    }
    pub fn resourcePeak(self: Classes) !?usize {
        return self.vtable.resource_peak(self.ctx);
    }
    pub fn leaf(self: Classes, block: u64, index: usize) !Leaf {
        return self.vtable.leaf(self.ctx, block, index);
    }
    pub fn coefficients(self: Classes, block: u64, index: usize, inventory: *const CPU.rv64_gkr_execution.Inventory) ![]const u32 {
        return self.vtable.coefficients(self.ctx, block, index, inventory);
    }
    pub fn identity(self: Classes, block: u64, index: usize) ![32]u8 {
        return self.vtable.identity(self.ctx, block, index);
    }
    pub fn requestPinnedWords(self: Classes, block: u64) !usize {
        return self.vtable.pinned(self.ctx, block);
    }
};
