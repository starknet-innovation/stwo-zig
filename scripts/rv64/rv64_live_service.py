"""Admission and reporting for the single-process recursive Ethereum benchmark."""
import json
import os
from pathlib import Path

from scripts.rv64.rv64_m7_integration_stack import write
from scripts.rv64.rv64_validation import sha, validate_receivers


def prepare_inputs(root, base, blocks, guest_request, guest_elf, census, *, guest_execution="cpu"):
    """Install only request metadata. Every process shares the real GPU lock."""
    common = json.loads(guest_request.read_text())
    if (type(common.get('terminal_pc')) is not int or not common.get('symbols') or
            common.get('execution_profile') not in ('ethereum_services_v3', 'ethereum_services_v3_zbb')):
        raise ValueError('a complete proving request with the measured ELF ABI and terminal PC is required')
    elf = (guest_elf or (guest_request.parent / common['elf'])).resolve(strict=True)
    if sha(elf) != common['elf_sha256']:
        raise ValueError('guest ELF does not match its qualified proving request')
    inputs = root / 'inputs'
    inputs.mkdir(parents=True, exist_ok=False)
    for name in ('locks', 'src', 'build', 'catalogues-v3'):
        target = (base / name).resolve(strict=True)
        if not target.is_dir():
            raise ValueError(f'expected existing pod directory for {name}')
        (root / name).symlink_to(target, target_is_directory=True)
    (inputs / 'guest.elf').symlink_to(elf)
    for name in ('source-authority.json', 'accepted-v7.json'):
        if (base / 'inputs' / name).is_file():
            (inputs / name).symlink_to((base / 'inputs' / name).resolve())
    plan = json.loads((base / 'inputs/plan-template.json').read_text())
    jobs = [job for job in plan['jobs'] if job['block'] in blocks]
    if sorted(job['block'] for job in jobs) != sorted(blocks):
        raise ValueError('basket blocks missing or duplicated in the original template')
    for job in jobs:
        block = str(job['block'])
        row = census[block]
        original = json.loads((base / f'inputs/requests/{block}.json').read_text())
        steps = common.get('batch_steps', 1 << 20)
        instructions = row['instructions']
        terminal = row.get('terminal_instructions', instructions)
        if (type(instructions) is not int or type(terminal) is not int or
                not 0 < instructions <= terminal <= common.get('max_steps', 1000000000) or
                type(steps) is not int or steps <= 0):
            raise ValueError(f'{block} instruction census exceeds the proving request')
        expected_segments = (terminal + steps - 1) // steps
        if (row['elf_sha256'] != common['elf_sha256'] or row['cpu_segments'] != expected_segments or
                any(source['expected_output'] != row['output'] or source['input_sha256'] != row['input_sha256']
                    for source in (original, job))):
            raise ValueError(f'{block} original statement, maintained guest or measured CPU geometry disagree')
        data = Path(original['input'])
        if not data.is_file():
            data = base / f'inputs/{block}.bin'
        request = dict(common, elf=str(elf), input=str(data.resolve(strict=True)),
                       input_sha256=row['input_sha256'], expected_output=row['output'], output_directory=None)
        path = inputs / f'requests/{block}.json'
        write(path, request)
        job.update(segments=row['cpu_segments'], request=str(path), request_sha256=sha(path))
    write(inputs / 'plan-template.json', dict(plan, elf_sha256=common['elf_sha256'], jobs=jobs))
    return pin_workload(inputs, blocks, census, guest_execution=guest_execution)


def pin_workload(inputs, blocks, census, *, guest_execution="cpu"):
    """Bind the measured guest, block bytes, output and resulting segment count."""
    jobs = json.loads((inputs / 'plan-template.json').read_text())['jobs']
    by_block = {job['block']: job for job in jobs}
    if len(by_block) != len(jobs):
        raise ValueError('duplicate blocks in plan template')
    hashes, pinned = {}, {}
    for block in blocks:
        if str(block) not in census or block not in by_block:
            raise ValueError(f'{block} missing from the guest census or plan')
        row, job = census[str(block)], by_block[block]
        path = inputs / 'requests' / f'{block}.json'
        request = json.loads(path.read_text())
        if guest_execution == 'cuda' and request.get('execution_profile') != 'ethereum_services_v3':
            raise ValueError(f'{block} requires the authenticated CUDA guest profile ethereum_services_v3')
        if (job['segments'] != row['cpu_segments'] or job['expected_output'] != row['output'] or
                request['expected_output'] != row['output'] or job['input_sha256'] != row['input_sha256']):
            raise ValueError(f'{block} pinned geometry or statement disagrees with the guest census')
        for field in ('elf', 'input'):
            key = field + '_sha256'
            asset = Path(request[field]).resolve()
            if asset not in hashes:
                hashes[asset] = sha(asset)
            if hashes[asset] != row[key] or request[key] != row[key]:
                raise ValueError(f'{block} {field} differs from the measured guest workload')
        pinned[block] = dict(cpu_segments=row['cpu_segments'], elf_sha256=row['elf_sha256'],
                             input_sha256=row['input_sha256'], output=row['output'], request_sha256=sha(path))
        if 'terminal_instructions' in row:
            pinned[block]['terminal_instructions'] = row['terminal_instructions']
    return pinned


def accepted_run(run, blocks, assets, qualified, *, require_native=False, guest_census=None):
    receipt = json.loads((run / 'receipt.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    live, ready = receipt['live_service'], receipt['service_ready_record']
    requests = receipt['requests']
    if guest_census is not None and (not require_native or manifest.get('resources', {}).get('guest_execution') != 'cuda'):
        raise ValueError('CUDA guest mode was not selected by the service manifest')
    setup_mode = ready.get('setup_mode', 'unreported')
    preinstalled = ready.get('preinstalled_blocks', len(blocks) if require_native else 0)
    if type(preinstalled) is not int or preinstalled not in (0, len(blocks)):
        raise ValueError('invalid preinstalled block count')
    if guest_census is not None and qualified and (setup_mode != 'device_geometry' or preinstalled != 0):
        raise ValueError('qualified CUDA proving requires on-demand proofless device geometry setup')
    if require_native:
        if (manifest.get('schema') != 'rv64-ethereum-basket-inputs/v1' or
                manifest.get('execution_order') != blocks or set(manifest.get('blocks', {})) != set(map(str, blocks)) or
                manifest.get('warm_order') or ready.get('runtime') != 'm8-gpu-native' or
                receipt['environment'].get('STWO_RV64_M8_GPU_NATIVE') != '1'):
            raise ValueError('GPU-native service or installed roster evidence is missing')
    elif (preinstalled != 0 or manifest.get('schema') != 'rv64-ethereum-live-service/v1' or manifest.get('blocks') != {} or
          manifest.get('execution_order') != [] or manifest.get('warm_order', []) != [] or
          ready.get('live_admission') is not True):
        raise ValueError('live producer started with a preinstalled block')
    if (ready['pid'] != live['producer_pid'] or ready['pid'] <= 0 or
            live['preinstalled_blocks'] != preinstalled or live['distinct_blocks'] != len(blocks) or
            live['qualified_latency'] is not qualified or receipt['repeat'] != 1 or receipt['blocks'] != blocks or
            receipt['environment'].get('CUDA_VISIBLE_DEVICES') != '0' or
            receipt['producer_process']['exit_code'] != 0 or [r['block'] for r in requests] != blocks):
        raise ValueError('incomplete single-device live service evidence')
    for key in ('driver', 'library', 'receiver'):
        if receipt['assets'][key] != assets[key]:
            raise ValueError(f'live service loaded a different {key}')
    verdict = validate_receivers(receipt['receivers'], [r['proof'] for r in requests], assets['receiver'],
                                 require_artifact_binding=True)
    by_proof = {r['proof']: r for r in receipt['receivers']}
    first = last = 0
    rows = []
    for index, request in enumerate(requests):
        submission = json.loads((run / f'submission-{index:04d}.json').read_text())
        submitted = submission['submitted_monotonic_ns']
        latency, proving = request['client_latency_ns'], request['marker']['request_wall_ns']
        if (submission['block'] != blocks[index] or request['ordinal'] != index or request['trial'] != 0 or
                request['output_matches_pin'] is not True or request['proof_bytes_root'] <= 0 or
                submitted < last or not 0 < proving <= latency):
            raise ValueError(f'inconsistent published proof or clock for block {blocks[index]}')
        receiver = by_proof[request['proof']]
        expected_steps = expected_launches = None
        if guest_census is not None:
            census = guest_census[str(request['block'])]
            # The census stops at the guest halt flag. The authenticated
            # statement also covers the epilogue up to the terminal PC.
            expected_steps = receiver.get('verified_total_steps')
            expected_launches = census['cpu_segments']
            if (type(expected_steps) is not int or type(census['instructions']) is not int or
                    not 0 < census['instructions'] <= expected_steps or
                    ('terminal_instructions' in census and
                     (type(census['terminal_instructions']) is not int or census['terminal_instructions'] != expected_steps)) or
                    type(expected_launches) is not int or expected_launches <= 0):
                raise ValueError('fresh CPU receiver did not report verified retirements consistent with the census')
        device = native_request_metrics(request, index, proving, expected_steps, expected_launches,
                                        on_demand=setup_mode == 'device_geometry' and preinstalled == 0) if require_native else {}
        if guest_census is not None:
            device['census_instructions'] = census['instructions']
        if index == 0:
            first = submitted
        last = submitted + latency
        rows.append(dict(block=blocks[index], proving_s=proving / 1e9, ticket_to_root_s=latency / 1e9,
                         fresh_verify_s=receiver['process']['wall_ns'] / 1e9,
                         root_bytes=request['proof_bytes_root'], **device))
    clocks = {}
    clock_file = run / 'service-ready-clock.json'
    if clock_file.is_file():
        clock = json.loads(clock_file.read_text())
        launch, ready_at = clock['launch_monotonic_ns'], clock['observed_ready_monotonic_ns']
        if clock['pid'] != ready['pid'] or not 0 < launch <= ready_at <= first:
            raise ValueError('service clock disagrees with producer or first submission')
        clocks = dict(ready_to_first_ticket_s=(first - ready_at) / 1e9,
                      launch_to_first_root_s=(first - launch) / 1e9 + rows[0]['ticket_to_root_s'],
                      launch_to_last_root_s=(last - launch) / 1e9)
    if guest_census is not None:
        modes = {row['guest']['mode'] for row in rows}
        if len(modes) != 1:
            raise ValueError('CUDA guest modes changed between tickets')
        if modes == {'cuda_resident'}:
            if not clocks:
                raise ValueError('resident CUDA guest timing requires the installed service clock')
            executors = {row['guest'].get('executor', 'unreported') for row in rows}
            if len(executors) != 1 or not executors <= {'cuda', 'host_witness', 'unreported'}:
                raise ValueError('inconsistent guest executor identity')
            executor = executors.pop()
            backend = {'cuda': 'CUDA guest execution', 'host_witness': 'CPU guest execution and handoff',
                       'unreported': 'Unreported guest executor'}[executor] + ' and GPU-native proofs with resident trace ingress'
        else:
            if qualified:
                raise ValueError('qualified CUDA proving requires resident guest ingress')
            backend = 'CUDA guest execution and GPU-native proof jobs with staged trace ingress'
    else:
        backend = 'GPU-native proof jobs with host guest capture' if require_native else 'resident live GPU'
    ticket_total = sum(row['ticket_to_root_s'] for row in rows)
    proving_total = sum(row['proving_s'] for row in rows)
    return dict(backend=backend, devices=1, setup_mode=setup_mode, preinstalled_blocks=live['preinstalled_blocks'], producer_pid=ready['pid'], qualified_latency=qualified,
                service_start_s=receipt['service_start_s'], service_stop_s=receipt['service_stop_s'],
                first_ticket_to_last_root_s=(last - first) / 1e9,
                ticket_to_root_s_sum=ticket_total, ticket_to_root_mean_s=ticket_total / len(rows),
                proving_mean_s=proving_total / len(rows),
                startup_amortized_proving_mean_s=(receipt['service_start_s'] + proving_total) / len(rows),
                fresh_verify_mean_s=sum(row['fresh_verify_s'] for row in rows) / len(rows),
                requests=rows, receiver=verdict, **clocks)



def native_request_metrics(request, index, proving, expected_steps=None, expected_launches=None, *, on_demand=False):
    native = json.loads((Path(request['proof']).parents[1] / 'm8-jobs.json').read_text())
    receipt = native['receipt']
    spans = receipt['device_gpu_us']
    if (native.get('schema') != 'rv64-m8-gpu-native-request/v1' or native['block'] != request['block'] or
            native['sequence'] != index or native['request_wall_ns'] != proving or
            native['proofs'] <= 0 or receipt['host_waits'] != 1 or receipt['d2h'] != 1 or
            not spans or spans[0] <= 0 or any(spans[1:])):
        raise ValueError('missing single-GPU native proof and terminal readback evidence')
    admission = native.get('admission_ns', 0)
    if type(admission) is not int or not 0 <= admission < proving or (on_demand and admission == 0):
        raise ValueError('missing or inconsistent measured block admission')
    out = dict(device_gpu_s=spans[0] / 1e6, device_to_host_bytes=receipt['d2h_bytes'],
               capture_wait_s=receipt['capture_wait_ns'] / 1e9, admission_s=admission / 1e9)
    guest = native.get('guest')
    if expected_steps is not None:
        if (not isinstance(guest, dict) or guest.get('mode') not in ('cuda_staged', 'cuda_resident') or
                type(expected_steps) is not int or expected_steps <= 0 or guest.get('retirements') != expected_steps or
                any(type(guest.get(k)) is not int or guest[k] <= 0 for k in
                    ('launches', 'state_readback_bytes', 'device_allocated_bytes',
                     'device_us', 'compact_d2d_bytes')) or
                guest['compact_d2d_bytes'] <= expected_steps * 16):
            raise ValueError('missing or inconsistent CUDA guest execution evidence')
        if guest['mode'] == 'cuda_resident':
            for key in ('trace_readback_bytes', 'trace_readback_ns', 'projection_ns',
                        'retired_host_metadata_bytes', 'retired_host_ls_bytes'):
                if type(guest.get(key)) is not int or guest[key] != 0:
                    raise ValueError(f'resident CUDA guest reports host trace work: {key}')
            windows = guest.get('replayed_windows', 0)
            retained = guest.get('retained_capture_bytes', 0)
            retries = guest.get('discovery_retry_launches', 0)
            segments = expected_launches if expected_launches is not None else guest['launches']
            if (type(retries) is not int or not 0 <= retries < guest['launches'] or (retries and not on_demand) or
                    type(windows) is not int or type(retained) is not int or not 0 <= windows <= segments or
                    (windows == 0 and retained != 0) or
                    (windows > 0 and not (expected_steps if windows == segments else windows) * 16
                     <= retained <= guest['device_allocated_bytes'])):
                raise ValueError('inconsistent retained CUDA guest capture evidence')
            if expected_launches is not None:
                if on_demand and guest['launches'] != retries + 2 * segments - windows:
                    raise ValueError('inconsistent retained CUDA guest launch count')
                if not on_demand and guest['launches'] != segments:
                    raise ValueError('missing or inconsistent CUDA guest execution evidence')
        elif (type(guest.get('trace_readback_bytes')) is not int or
              guest['trace_readback_bytes'] < expected_steps * 16 or
              (expected_launches is not None and guest['launches'] != expected_launches)):
            raise ValueError('missing or inconsistent CUDA guest execution evidence')
        out['guest'] = guest
        out['guest_execution_and_handoff_s'] = guest['device_us'] / 1e6
        if guest.get('executor') == 'cuda':
            out['guest_device_s'] = out['guest_execution_and_handoff_s']
    elif guest is not None:
        raise ValueError('unexpected CUDA guest execution in the host capture comparison')
    return out


def compare_cpu(run, gpu, output, receiver_workers):
    """Same guest and proof profile, with a CPU-only producer and isolated receivers."""
    from scripts.rv64 import rv64_m4_resident as resident

    plan = json.loads((run / 'plan.json').read_text())
    if plan['store']['directory'] is not None:
        raise ValueError('comparison requires an isolated proof receiver')
    env = resident.R.receiver_environment({})
    for key in resident.P.FORMAT_ENV_KEYS:
        if key in plan['resident_environment']:
            env[key] = plan['resident_environment'][key]
    cpu = Path(plan['assets']['recursive_receiver'])
    receiver_pin = gpu['receiver']['receiver_sha256']
    if sha(cpu) != receiver_pin:
        raise ValueError('CPU producer and receiver must use the pinned CPU executable')
    output.mkdir()
    jobs = {job['block']: job for job in plan['jobs']}
    rows = []
    for measured in gpu['requests']:
        block = measured['block']
        job = jobs[block]
        case = output / str(block)
        case.mkdir()
        resources = case / 'resources.json'
        cpu_resources = dict(job['resources'], park_main=False)
        cpu_resources.pop('guest_execution', None)
        cpu_resources.pop('grouped_service_recursion', None)
        schedule = dict(cpu_resources['schedule'])
        schedule.pop('device_affinity', None)
        cpu_resources['schedule'] = schedule
        write(resources, cpu_resources)
        request = case / 'request.json'
        write(request, dict(json.loads(Path(job['request']).read_text()), output_directory=str(case / 'capture')))
        (case / 'capture').mkdir()
        proof = case / 'proof'
        argv = [str(cpu), '--security-profile', plan['profile'], 'produce-cpu-and-recurse-candidate',
                str(request), '-', str(proof), '0', str(resources)]
        load_before = os.getloadavg()
        process = resident.B.measure(argv, case, 'producer', env, cwd=plan['producer_cwd'])
        process['load_avg_before'] = load_before
        process['load_avg_after'] = os.getloadavg()
        combined = json.loads((proof / 'combined-report.json').read_text())
        if (combined.get('accepted') is not True or combined.get('completed_ethereum_block') is not True or
                combined['recursive']['outer_backend'] != 'cpu' or
                combined['recursion_execution']['physical_device_ids']):
            raise RuntimeError(f'{block} CPU comparison did not publish a CPU-only recursive proof')
        received = resident.receive_one(plan, block, proof, case, receiver_workers, grouped=False)
        received['store_present_during_receive'] = False
        validate_receivers([received], [str(proof)], receiver_pin, require_artifact_binding=True)
        if (guest := measured.get('guest')) is not None:
            if received.get('verified_total_steps') != guest['retirements']:
                raise RuntimeError(f'{block} CPU and CUDA guest proof coverage disagree')
        elapsed = process['wall_ns'] / 1e9
        rows.append(dict(block=block, gpu_ticket_to_root_s=measured['ticket_to_root_s'],
                         gpu_device_span_s=measured['device_gpu_s'], cpu_prove_s=elapsed,
                         cpu_fresh_verify_s=received['process']['wall_ns'] / 1e9,
                         per_request_speedup=elapsed / measured['ticket_to_root_s'],
                         cpu_producer=process, receiver=received))
        write(case / 'comparison.json', rows[-1])
    total_cpu = sum(r['cpu_prove_s'] for r in rows)
    return dict(rows=rows, cpu_prove_sum_s=total_cpu, cpu_mean_s=total_cpu / len(rows),
                gpu_mean_s=gpu['ticket_to_root_s_sum'] / len(rows),
                request_speedup=total_cpu / gpu['ticket_to_root_s_sum'],
                speedup_including_gpu_startup=total_cpu / gpu['launch_to_last_root_s'],
                gpu_first_launch_to_last_root_s=gpu['launch_to_last_root_s'],
                cpu_producer_policy='one CPU process per block, no CUDA, fresh receivers timed separately')
