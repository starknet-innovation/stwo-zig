"""Human-readable tables for recursive benchmark measurements."""
from .metrics import DEVICES


def fmt(value, digits=1):
    return '-' if value is None else f'{value:.{digits}f}'


def markdown(result):
    """Short tables: warm/cold proving per arm next to M3/M7, then the warm critical path per arm."""
    arms = result['arms']
    config = result.get('config', {})
    lines = ['# RV64 M8 15-block benchmark', '',
             'proving = service request wall incl. witness generation, excl. fetch/submit. Warm/cold = median over headline services [M].',
             'M3 = one request per block (`rv64-m3-20261104`) [M]; M7 = default warm where measured [M].', '',
             f"Selected GPUs: {config.get('gpus', DEVICES)}; logical cap per device: {config.get('device_cap_bytes', 28 << 30)} bytes. Admission alone establishes neither measured peak nor 5090 fit.", '',
             '| block | Minstr | seg | M3 s | M7 s | ' + ' | '.join(f'{a} warm / cold s' for a in arms) + ' | status |',
             '|---|---:|---:|---:|---:|' + '---:|' * len(arms) + '---|']
    for b in result['blocks']:
        m7 = b['m7'] or {}
        m7s = m7.get('warm_proving_s') or m7.get('cold_proving_s')
        cells = []
        for a in arms:
            arm = b['arms'].get(a)
            if a in (b.get('arm_failures') or {}):
                cells.append('FAIL')
            elif arm and arm['warm']['proving_s'] is not None:
                cells.append(f"{fmt(arm['warm']['proving_s'], 2)} / {fmt(arm['cold']['proving_s'], 2)}")
            else:
                cells.append('-')
        lines.append(f"| {b['block']} | {b['instructions'] / 1e6:.1f} | {b['cpu_segments']} | {fmt(b['m3_request_s'])} | "
                     f"{fmt(m7s)}{' (cold)' if m7 and not m7.get('warm_proving_s') else ''} | " + ' | '.join(cells) + f" | {b['status']} |")
    if result['totals']:
        lines += ['', '| arm | blocks | warm sum s | cold sum s | M3 sum s (same blocks) | warm / M3 [D] |', '|---|---:|---:|---:|---:|---:|']
        for a, t in result['totals'].items():
            lines.append(f"| {a} | {t['blocks']} | {t['warm_proving_s_sum']:.1f} | {t['cold_proving_s_sum']:.1f} | {t['m3_request_s_sum_same_blocks']:.1f} | {t['warm_over_m3']:.3f} |")
        lines += ['', '| arm | warm p50 / p90 / max s | cold p50 / p90 / max s | warm request p50 / p90 / max s |', '|---|---:|---:|---:|']
        q = lambda d: '-' if not d else f"{d['p50']:.2f} / {d['p90']:.2f} / {d['max']:.2f}"
        for a, t in result['totals'].items():
            lines.append(f"| {a} | {q(t.get('warm_proving_s'))} | {q(t.get('cold_proving_s'))} | {q(t.get('warm_request_s'))} |")
    failures = []
    for b in result['blocks']:
        if b['status'] == 'failed':
            failures.append(f"- {b['block']} FAILED: `{b.get('error')}`")
        for a, failure in (b.get('arm_failures') or {}).items():
            failures.append(f"- {b['block']} {a} FAILED ({failure.get('kind')}): `{failure.get('reason')}`")
    if failures:
        lines += ['', '## Failures', ''] + failures
    lines += ['', '## Warm critical path per arm', '',
              '| block | arm | warm request s | capture | setup | native window | recursion after native | tail [D] | kernel GPU-s | GPU busy [D] | device peak GB | per-node peak GB | launches/node | syncs/node |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for b in result['blocks']:
        for a in arms:
            arm = b['arms'].get(a)
            if not arm or arm['warm']['proving_s'] is None:
                continue
            w, p = arm['warm'], arm.get('profile_warm') or {}
            launches = arm.get('job_launches_per_node', p.get('launches_per_node'))
            syncs = arm.get('job_syncs_per_node', p.get('syncs_per_node'))
            peak = arm.get('per_node_peak_bytes')
            device = arm.get('device_peak_bytes_sampled')
            lines.append(f"| {b['block']} | {a} | {fmt(w['request_s'], 2)} | {fmt(w['capture_s'], 2)} | {fmt(w['setup_s'], 2)} | {fmt(w['native_window_s'], 2)} | "
                         f"{fmt(w['recursion_after_native_s'], 2)} | {fmt(w['tail_s'], 2)} | {fmt(p.get('kernel_gpu_s'), 2)} | "
                         f"{fmt(p.get('gpu_busy') and p['gpu_busy'] * 100, 0)}{'%' if p.get('gpu_busy') else ''} | "
                         f"{fmt(device and device / 1e9, 1)} | {fmt(peak and peak / 1e9, 1)} | {fmt(launches, 0)} | {fmt(syncs, 1)} |")
    verified = [(block['block'], name, trial) for block in result['blocks']
                for name, arm in block['arms'].items() for trial in arm.get('verified_requests', [])]
    if verified:
        lines += ['', '## Timed requests with fresh CPU verification', '',
                  'Only retained proofs appear here. Startup is charged once to the first request. Totals sum the observed phases for the same proof.', '',
                  '| block | arm | round | trial | service startup s | request s | verify s | request and verify s | first request incl. startup s |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for block, arm, trial in verified:
            lines.append(f"| {block} | {arm} | {trial['round'] + 1} | {trial['trial']} | "
                         f"{fmt(trial['service_start_s'], 2)} | {fmt(trial['request_s'], 2)} | {fmt(trial['verify_s'], 2)} | "
                         f"{fmt(trial['request_and_verify_s'], 2)} | {fmt(trial['startup_request_and_verify_s'], 2)} |")
    lines += ['', 'launches/syncs per node: job receipts (`GPU_M8_OUTER_NODE launches/host_waits`) where the arm runs OuterNodeJobs [M], '
              'else Nsight `cudaLaunchKernel`/`cudaStreamSynchronize` in the profiled warm window / nodes [D]. '
              'device peak: NVML-sampled max used bytes over selected GPUs in the headline services [M]. '
              'per-node peak: job arena peak words x 4 B [D]. GPU busy: CUPTI kernel union / (selected GPU count x profiled window) [D].', '']
    return '\n'.join(lines)
