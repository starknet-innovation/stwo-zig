//! Weight setup helpers shared by the transformer tests and other test files.
const std = @import("std");
const f = @import("field.zig");
const proto = @import("protocol.zig");
const xt = @import("ext.zig");
const model = @import("model.zig");
const tf = @import("transformer.zig");

const TensorInfo = proto.TensorInfo;
const Ref = xt.Ref;
const LayerCfg = tf.LayerCfg;
const layerWeightInfos = tf.layerWeightInfos;

pub const TestSetup = struct {
    pool: f.Pool,
    ws: proto.WeightStore,
    wdata: [][]i8,
    winfos: []TensorInfo,
    pc: proto.Config,

    pub fn deinit(self: *TestSetup, a: std.mem.Allocator) void {
        self.ws.deinit(a);
        for (self.wdata) |w| a.free(w);
        a.free(self.wdata);
        a.free(self.winfos);
        self.pool.deinit();
    }
};

pub const WeightSetup = struct { ws: proto.WeightStore, wdata: [][]i8, winfos: []TensorInfo };

pub fn setupWeights(a: std.mem.Allocator, pool: *f.Pool, cfg: LayerCfg, pc: proto.Config, seed: u64) !WeightSetup {
    return setupWeightsWith(a, pool, cfg, pc, seed, false);
}

/// tie_router: copy router column 0 into column 1 (test: tied experts 0 / 1).
pub fn setupWeightsWith(a: std.mem.Allocator, pool: *f.Pool, cfg: LayerCfg, pc: proto.Config, seed: u64, tie_router: bool) !WeightSetup {
    const winfos = try layerWeightInfos(a, cfg);
    const wdata = try a.alloc([]i8, winfos.len);
    const wv = try a.alloc([]const i8, winfos.len);
    defer a.free(wv);
    for (wdata, 0..) |*w, i| {
        w.* = try model.randomWeights(a, winfos[i].size(), seed + i, -8, 7);
        wv[i] = w.*;
    }
    if (tie_router and cfg.isMoe()) {
        for (0..cfg.n_layers) |l| {
            const r = wdata[l * cfg.weightsPerLayer() + cfg.wRouter()];
            const ncol = @as(usize, 1) << @intCast(cfg.routerLogCols());
            for (0..r.len / ncol) |row| r[row * ncol + 1] = r[row * ncol];
        }
    }
    const ws = try proto.commitWeights(a, pool, pc, winfos, wv);
    return .{ .ws = ws, .wdata = wdata, .winfos = winfos };
}
