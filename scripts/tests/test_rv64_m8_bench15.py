import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rv64_m8_bench15 as bench

NODE = ('GPU_M8_OUTER_NODE ordinal=0 device=0 leaf=true outcome=direct launches=1118 kernel_nodes=0 host_waits=1 '
        'gpu_ns=418025970 arena_peak_words=4553800832 plan_words={ 351748352, 703496704 } placement=bottom')
NODE2 = 'GPU_M8_OUTER_NODE ordinal=1 device=1 leaf=true outcome=captured launches=1 host_waits=1 gpu_ns=400000000 arena_peak_words=4000000000'
HOST = 'GPU_M8_OUTER_NODE_HOST ordinal=0 device=0 prestage_ns=391964464 launches=99999'
MEM = 'GPU_M8_OUTER_NODE_MEM ordinal=0 device=0 heap_high_words=3309229888 arena_peak_words=9999999999'
FIXED = 'GPU_M8_OUTER_FIXED ordinal=0 device=0 columns=45 outcome=direct launches=53 gpu_ns=7645887 host_waits=1'
NATIVE = ('GPU_M8_NATIVE_HYBRID stage=2 request=1 ordinal=3 device=1 gpu_ms=230.500 enqueue_ns=5 wait_ns=6 '
          'arena_peak_words=1000 host_waits=1')
EVENT = 'GPU_M8_NATIVE_HYBRID event=boot workers=2 arena_words=5 wide=true'


def request(ordinal, wall_s, latency_s):
    return {'ordinal': ordinal, 'marker': {'request_wall_ns': int(wall_s * 1e9)}, 'client_latency_ns': int(latency_s * 1e9)}


COMBINED = {'capture_ns': 2_000_000_000, 'native_setup_ns': 140_000_000, 'native_prove_ns': 4_750_000_000,
            'recursion_after_native_ns': 13_200_000_000, 'recursive': {'leaf_proofs': 14, 'fold_proofs': 5, 'root_proof_bytes': 749688}}


def service(arm, role, warm_s, cold_s=None, status='ok', **extra):
    rows = [dict(proving_s=cold_s or 2 * warm_s, request_s=(cold_s or 2 * warm_s) + 0.4, capture_s=2.0, setup_s=0.1,
                 native_window_s=4.0, recursion_after_native_s=5.0, tail_s=0.5, jobs={}),
            dict(proving_s=warm_s, request_s=warm_s + 0.4, capture_s=1.5, setup_s=0.1, native_window_s=3.0,
                 recursion_after_native_s=warm_s - 5.0, tail_s=0.4, jobs={})]
    return dict(arm=arm, role=role, status=status, requests=rows, **extra)


class BasketTests(unittest.TestCase):
    def test_verified_request_cost_joins_only_the_matching_proof(self):
        row = service('gpu-native', 'headline', 5, run='retained', service_start_s=3, round=0)
        for ordinal, request in enumerate(row['requests']):
            request['ordinal'] = ordinal
        verdicts = {'retained': [{'trial': 0, 'accepted': True, 'wall_s': 2},
                                 {'trial': 1, 'accepted': False, 'wall_s': 4}]}
        costs = bench.verified_requests([row, dict(row, run='discarded')], verdicts)
        self.assertEqual(1, len(costs))
        self.assertEqual(12.4, costs[0]['request_and_verify_s'])
        self.assertEqual(15.4, costs[0]['startup_request_and_verify_s'])

    def test_subset_rejects_unknown_blocks_without_discarding_known_ones(self):
        expected = [row['block'] for row in bench.basket() if row['block'] in (24628607, 26235156)]
        self.assertEqual(expected, [row['block'] for row in bench.selected_blocks(expected[::-1])])
        with self.assertRaisesRegex(SystemExit, 'regression basket: 1'):
            bench.selected_blocks([24628607, 1])

    def test_check_reports_missing_mount_without_creating_gpu_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'unmounted'
            with patch.object(bench, 'M3', root), patch.object(bench, 'GPU_LOCK', root / 'locks/gpu.lock'), patch('builtins.print'):
                result = bench.check(argparse.Namespace(assets=None, out=None, blocks=[24628607]))
            self.assertEqual('MISSING', result['status'])
            self.assertEqual([24628607], [row['block'] for row in result['blocks']])
            self.assertIsNone(result['disk_free_bytes'])
            self.assertIsNone(result['gpu_lock_free_now'])
            self.assertFalse(root.exists())

    def test_prune_unlinks_profile_symlink_without_following_target(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bench, 'M3', Path(tmp)):
            run = Path(tmp) / 'runs/test/daemon-case'
            (run / 'spool/out').mkdir(parents=True)
            external = Path(tmp) / 'profiles'
            external.mkdir()
            evidence = external / 'timeline.nsys-rep'
            evidence.write_bytes(b'profile')
            (run / 'nsys').symlink_to(external, target_is_directory=True)
            worker = object.__new__(bench.Bench)
            worker.stream = 'test'
            worker.prune(run)
            worker.prune(run)
            self.assertFalse((run / 'spool/out').exists())
            self.assertFalse((run / 'nsys').is_symlink())
            self.assertEqual(b'profile', evidence.read_bytes())

    def test_capacity_cli_selects_one_gpu_native_round_and_explicit_preset(self):
        argv = ['bench', 'run', '--assets', 'assets.json', '--out', 'run', '--arms', 'gpu-native',
                '--gpus', '1', '--device-cap-bytes', str(48 << 30), '--preset', 'candidate.json',
                '--rounds', '1', '--profile', 'warm']
        with patch.object(sys, 'argv', argv), patch.object(bench, 'Bench') as factory:
            bench.main()
            args = factory.call_args.args[0]
            self.assertEqual((1, 48 << 30, Path('candidate.json')), (args.gpus, args.device_cap_bytes, args.preset))
            self.assertEqual((['gpu-native'], 1, 'warm'), (args.arms, args.rounds, args.profile))
            factory.return_value.run.assert_called_once()
        for value in ('0', '-1'):
            with self.assertRaises(ValueError):
                bench.positive_capacity(value)

    def test_selected_gpu_capacity_and_preset_bind_service_and_resume(self):
        preset = {'security_profile': 'd2_native_b4_q54_p20_fold4', 'common': {}, 'arms': {'baseline': {}},
                  'operational': {'baseline': {'recursion_max_pending_authorities': 21}},
                  'measurement': {'quiet_load_max': 8, 'quiet_steal_max': .01,
                                  'quiet_timeout_s': 300, 'quiet_stable_samples': 5}}
        manifest = {'source_root': '/source', 'block': 24628607, 'preset': preset,
                    'assets': {'driver': '/driver', 'library': '/library', 'receiver': '/receiver'},
                    'gpus': 1, 'device_cap_bytes': 48 << 30}
        argv, _ = bench.stack.command(manifest, 'baseline', 0, 'stream', False)
        self.assertEqual('1', argv[argv.index('--gpus') + 1])
        self.assertEqual(str(48 << 30), argv[argv.index('--device-cap-bytes') + 1])
        self.assertNotIn('--process-budget-guard', argv)
        config = {'gpus': 1, 'device_cap_bytes': 48 << 30, 'preset_sha256': 'a' * 64}
        for key, changed in (('gpus', 2), ('device_cap_bytes', 28 << 30), ('preset_sha256', 'b' * 64)):
            self.assertNotEqual(bench.resume_identity(config), bench.resume_identity(dict(config, **{key: changed})))

    def test_missing_unused_sparse_is_allowed_but_consumers_and_assets_remain_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            assets = {}
            for name in ('driver', 'receiver', 'library', 'observer', 'addon', 'outer_fs'):
                path = base / name
                path.write_text(name)
                assets[name] = str(path)
            pins = {name: bench.stack.sha(path) for name, path in assets.items()}
            preset = {'common': {'LD_PRELOAD': '{library}:{observer}'}, 'arms': {'baseline': {}}}
            environment = bench.stack.arm_environment(preset, 'baseline', assets)
            bench.validate_service_environment(environment, assets, pins)
            with self.assertRaises(RuntimeError):
                bench.validate_service_environment({'LD_PRELOAD': assets['library']}, assets, pins)
            with self.assertRaises(KeyError):
                bench.stack.arm_environment({'common': {'LD_PRELOAD': '{sparse}'}, 'arms': {'baseline': {}}}, 'baseline', assets)
            manifest = {'schema': 'rv64-m8-bench-assets/v1', 'assets': assets, 'asset_sha256': pins}
            path = base / 'assets.json'
            path.write_text(json.dumps(manifest))
            bench.load_assets(path)
            manifest['assets']['sparse'] = str(base / 'missing-sparse')
            manifest['asset_sha256']['sparse'] = '0' * 64
            path.write_text(json.dumps(manifest))
            with self.assertRaises((RuntimeError, FileNotFoundError)):
                bench.load_assets(path)
            (base / 'observer').unlink()
            with self.assertRaises((RuntimeError, FileNotFoundError)):
                bench.load_assets(path)

    def test_profile_busy_uses_actual_gpu_count(self):
        for count, expected in ((1, .4), (2, .2), (4, .1)):
            row = bench.profile_metrics({'kernel_busy_gpu_seconds': 10., 'any_busy_gpu_seconds': 10.}, None, 1, 25., device_count=count)
            self.assertEqual(expected, row['gpu_busy'])

    def test_basket_is_fifteen_blocks_ascending_and_matches_m3_sum(self):
        rows = bench.basket()
        self.assertEqual(len(rows), 15)
        self.assertEqual([r['request_s'] for r in rows], sorted(r['request_s'] for r in rows))
        self.assertAlmostEqual(sum(r['request_s'] for r in rows), 1843.0, delta=0.5)
        self.assertEqual(rows[0]['block'], bench.SMOKE_BLOCK)
        self.assertTrue(set(bench.DEV_BLOCKS) <= {r['block'] for r in rows})

    def test_basket_refuses_a_changed_summary(self):
        with self.assertRaises(RuntimeError):
            bench.basket({'rows': [{'request_s': 1}], 'totals': {'request_s_sum_15': 1843.0}})

    def test_large_blocks_release_the_idle_pcs_cache(self):
        self.assertEqual(bench.block_environment({'cpu_segments': 39}), {})
        self.assertEqual(bench.block_environment({'cpu_segments': 40}), {'STWO_RV64_PCS_IDLE_CACHE_BYTES': '0'})

    def test_arm_environment_substitutes_assets_and_refuses_unknown_arms(self):
        assets = {'addon': '/a/libjobs.so', 'outer_fs': '/a/libfs.so'}
        self.assertEqual(bench.arm_environment('default', assets), {})
        self.assertEqual(set(bench.ARMS), {'default', 'gpu-native'})
        self.assertTrue(all(k.startswith(('STWO_RV64_M8_', 'STWO_RV64_M9_')) for arm in bench.ARMS for k in bench.ARMS[arm]))
        native = bench.arm_environment('gpu-native', assets)
        self.assertEqual(native['STWO_RV64_M8_JOBS_LIBRARY'], '/a/libjobs.so')
        self.assertEqual(native['STWO_RV64_M8_OUTER_FS_LIBRARY'], '/a/libfs.so')
        self.assertEqual(native['STWO_RV64_M8_GPU_NATIVE_REUSE'], '1')
        self.assertEqual(native['STWO_RV64_M9_CLASS_PACKED_INGRESS'], '1')
        with self.assertRaises(ValueError):
            bench.arm_environment('fullstack', assets)

    def test_arms_option_validation(self):
        self.assertEqual(bench.arms_type('default,gpu-native'), ['default', 'gpu-native'])
        for bad in ('', 'default,default', 'nope'):
            with self.assertRaises(argparse.ArgumentTypeError):
                bench.arms_type(bad)

    def test_common_environment_is_explicit_and_conflicts_fail_closed(self):
        flag = 'STWO_RV64_M8_TRUTH_SPLIT=1'
        self.assertEqual(bench.extra_environment([flag, flag]), {'STWO_RV64_M8_TRUTH_SPLIT': '1'})
        for bad in (['PATH=/tmp'], ['STWO_FLAG'], ['STWO_bad=1'], [flag, flag[:-1] + '0']):
            with self.assertRaises(ValueError):
                bench.extra_environment(bad)


class LiveBasketTests(unittest.TestCase):
    def test_maintained_overlay_reuses_the_actual_gpu_lock_and_original_block_bytes(self):
        from scripts.rv64.rv64_live_service import prepare_inputs, pin_workload
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'assets'
            for directory in ('locks', 'src', 'build', 'catalogues-v3', 'inputs/requests'):
                (root / directory).mkdir(parents=True)
            (root / 'locks/gpu.lock').write_bytes(b'')
            for name in ('accepted-v7.json', 'source-authority.json'):
                (root / 'inputs' / name).write_text('{}')
            old_elf, new_elf, data = root/'old.elf', root/'new.elf', root/'10.bin'
            old_elf.write_bytes(b'original'); new_elf.write_bytes(b'updated'); data.write_bytes(b'block')
            digest = bench.stack.sha(data)
            old = dict(elf=str(old_elf), elf_sha256=bench.stack.sha(old_elf), input=str(data),
                       input_sha256=digest, expected_output='01', symbols={'heap': 20})
            request_path = root/'inputs/requests/10.json'
            request_path.write_text(json.dumps(old))
            job = dict(block=10, expected_output='01', input_sha256=digest, segments=9)
            plan_path = root/'inputs/plan-template.json'
            plan_path.write_text(json.dumps({'jobs': [job]}))
            common = root/'guest.json'
            common.write_text(json.dumps(dict(old, elf='/former/guest.elf',
                                             elf_sha256=bench.stack.sha(new_elf), symbols={'heap': 101}, terminal_pc=444,
                                             execution_profile='ethereum_services_v3', batch_steps=64)))
            pin = {'10': dict(elf_sha256=bench.stack.sha(new_elf), input_sha256=digest, output='01', instructions=100, cpu_segments=2)}
            work = Path(tmp)/'work'
            prepare_inputs(work, root, [10], common, new_elf, pin)
            self.assertTrue((work/'locks/gpu.lock').samefile(root/'locks/gpu.lock'))
            self.assertTrue((work/'inputs/guest.elf').samefile(new_elf))
            self.assertEqual(pin_workload(work/'inputs', [10], pin)[10]['cpu_segments'], 2)
            staged = json.loads((work/'inputs/requests/10.json').read_text())
            self.assertEqual(staged['symbols'], {'heap': 101})
            self.assertEqual(json.loads(request_path.read_text()), old)
            self.assertEqual(json.loads(plan_path.read_text())['jobs'], [job])
            self.assertEqual(staged['terminal_pc'], 444)
            epilogue = {'10': dict(pin['10'], instructions=64, terminal_instructions=65)}
            prepared = prepare_inputs(Path(tmp)/'epilogue', root, [10], common, new_elf, epilogue)
            self.assertEqual(prepared[10]['cpu_segments'], 2)
            replay = root / 'replay-only.json'
            replay.write_text(json.dumps({k: v for k, v in json.loads(common.read_text()).items() if k != 'terminal_pc'}))
            with self.assertRaisesRegex(ValueError, 'terminal PC'):
                prepare_inputs(Path(tmp)/'replay', root, [10], replay, new_elf, pin)
            with self.assertRaises(ValueError):
                prepare_inputs(Path(tmp)/'bad', root, [10], common, old_elf, pin)
            with self.assertRaises(ValueError):
                prepare_inputs(Path(tmp)/'wrong', root, [10], common, new_elf, {'10': dict(pin['10'], output='00')})
            (root/'locks/gpu.lock').unlink()
            (root/'locks').rmdir()
            with self.assertRaises(FileNotFoundError):
                prepare_inputs(Path(tmp)/'unlocked', root, [10], common, new_elf, pin)

    def test_workload_rejects_stale_guest_geometry_and_output(self):
        from scripts.rv64.rv64_live_service import pin_workload
        with tempfile.TemporaryDirectory() as tmp:
            inputs = Path(tmp)
            (inputs / 'requests').mkdir()
            elf, data = inputs / 'guest.elf', inputs / '10.bin'
            elf.write_bytes(b'guest')
            data.write_bytes(b'block')
            e, i = bench.stack.sha(elf), bench.stack.sha(data)
            pin = {'10': {'cpu_segments': 2, 'input_sha256': i, 'elf_sha256': e, 'output': '01'}}
            req = dict(elf=str(elf), input=str(data), elf_sha256=e, input_sha256=i, expected_output='01')
            (inputs / 'requests/10.json').write_text(json.dumps(req))
            job = dict(block=10, segments=2, input_sha256=i, expected_output='01')
            (inputs / 'plan-template.json').write_text(json.dumps({'jobs': [job]}))
            self.assertEqual(pin_workload(inputs, [10], pin)[10]['cpu_segments'], 2)
            req['execution_profile'] = 'ethereum_services_v3_zbb'
            (inputs / 'requests/10.json').write_text(json.dumps(req))
            with self.assertRaisesRegex(ValueError, 'authenticated CUDA guest profile'):
                pin_workload(inputs, [10], pin, guest_execution='cuda')
            req['execution_profile'] = 'ethereum_services_v3'
            (inputs / 'requests/10.json').write_text(json.dumps(req))
            self.assertEqual(pin_workload(inputs, [10], pin, guest_execution='cuda')[10]['cpu_segments'], 2)
            for key, value in [('cpu_segments', 3), ('elf_sha256', 'x'), ('output', '02')]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    pin_workload(inputs, [10], {'10': dict(pin['10'], **{key: value})})
            data.write_bytes(b'substitution')
            with self.assertRaises(ValueError):
                pin_workload(inputs, [10], pin)

    def test_live_receipt_accounts_for_all_distinct_roots_and_inter_request_gaps(self):
        from scripts.rv64.rv64_live_service import accepted_run
        from scripts.rv64.rv64_validation import proof_artifact_binding
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            manifest = dict(schema='rv64-ethereum-live-service/v1', blocks={}, execution_order=[])
            (run / 'manifest.json').write_text(json.dumps(manifest))
            pins = {name: 'a' * 64 for name in ('library', 'driver', 'receiver')}
            blocks = [10, 11]
            live = dict(preinstalled_blocks=0, distinct_blocks=2, qualified_latency=False, producer_pid=123)
            receipt = dict(service_ready_record=dict(live_admission=True, pid=123), live_service=live,
                           producer_process=dict(exit_code=0), assets=pins, repeat=1, blocks=blocks,
                           environment={'CUDA_VISIBLE_DEVICES': '0'}, service_start_s=1, service_stop_s=2,
                           requests=[], receivers=[])
            for index, block in enumerate(blocks):
                proof = str(run / 'out' / str(index) / str(block) / 'proof')
                root = Path(proof) / 'recursion-om31/root.proof'
                root.parent.mkdir(parents=True)
                root.write_bytes(f'proof-{block}'.encode())
                receipt['requests'].append(dict(block=block, proof=proof, ordinal=index, trial=0,
                    output_matches_pin=True, proof_bytes_root=123, client_latency_ns=2_000_000_000,
                    marker={'request_wall_ns': 1_999_000_000}))
                isolation = {'CUDA_VISIBLE_DEVICES': '', 'STWO_RV64_GPU_PCS_HELPER': '/bin/false',
                             'STWO_ZIG_RV64_SETUP_CACHE': 'unset', 'STWO_RV64_OUTER_PLAN_CLASS_STORE': 'unset',
                             'STWO_RV64_FOLD_CLASS_CACHE': '0'}
                receipt['receivers'].append(dict(proof=proof, accepted=True, status='accepted', receiver_sha256=pins['receiver'],
                    process=dict(exit_code=0, wall_ns=5_000_000_000), store_present_during_receive=False,
                    isolation=isolation, artifact_binding=proof_artifact_binding(proof)))
                (run / f'submission-{index:04d}.json').write_text(json.dumps(dict(block=block, submitted_monotonic_ns=4_000_000_001+3_000_000_000*index)))
            path = run / 'receipt.json'
            path.write_text(json.dumps(receipt))
            report = accepted_run(run, blocks, pins, False)
            self.assertEqual(report['first_ticket_to_last_root_s'], 5)
            self.assertEqual(report['ticket_to_root_s_sum'], 4)
            self.assertEqual(report['ticket_to_root_mean_s'], 2)
            self.assertEqual(report['proving_mean_s'], 1.999)
            self.assertEqual(report['fresh_verify_mean_s'], 5)
            self.assertNotIn('launch_to_first_root_s', report)
            root.write_bytes(b'replacement root')
            with self.assertRaisesRegex(RuntimeError, 'different proof content'):
                accepted_run(run, blocks, pins, False)
            root.write_bytes(f'proof-{blocks[-1]}'.encode())
            old_binding = receipt['receivers'][0].pop('artifact_binding')
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(RuntimeError, 'proof content binding is missing'):
                accepted_run(run, blocks, pins, False)
            receipt['receivers'][0]['artifact_binding'] = old_binding
            path.write_text(json.dumps(receipt))
            clock_file = run / 'service-ready-clock.json'
            clock = dict(pid=123, launch_monotonic_ns=1, observed_ready_monotonic_ns=1_000_000_001)
            clock_file.write_text(json.dumps(clock))
            measured = accepted_run(run, blocks, pins, False)
            self.assertEqual(measured['ready_to_first_ticket_s'], 3)
            self.assertEqual(measured['launch_to_first_root_s'], 6)
            self.assertEqual(measured['launch_to_last_root_s'], 9)
            clock_file.write_text(json.dumps(dict(clock, pid=1)))
            with self.assertRaisesRegex(ValueError, 'service clock'):
                accepted_run(run, blocks, pins, False)
            clock_file.unlink()
            self.assertEqual([row['fresh_verify_s'] for row in report['requests']], [5, 5])
            with self.assertRaises(ValueError):
                accepted_run(run, blocks[::-1], pins, False)
            with self.assertRaises(ValueError):
                accepted_run(run, blocks, pins, True)
            (run / 'manifest.json').write_text(json.dumps(dict(manifest, warm_order=[10])))
            with self.assertRaises(ValueError):
                accepted_run(run, blocks, pins, False)
            (run / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'GPU-native service'):
                accepted_run(run, blocks, pins, False, require_native=True)
            native_manifest = dict(schema='rv64-ethereum-basket-inputs/v1', execution_order=blocks,
                                   blocks={str(block): {} for block in blocks})
            (run / 'manifest.json').write_text(json.dumps(native_manifest))
            receipt['service_ready_record'] = dict(runtime='m8-gpu-native', pid=123)
            receipt['environment']['STWO_RV64_M8_GPU_NATIVE'] = '1'
            live['preinstalled_blocks'] = len(blocks)
            path.write_text(json.dumps(receipt))
            for index, request in enumerate(receipt['requests']):
                folder = Path(request['proof']).parents[1]
                folder.mkdir(parents=True, exist_ok=True)
                native = dict(schema='rv64-m8-gpu-native-request/v1', block=request['block'],
                              sequence=index, request_wall_ns=request['marker']['request_wall_ns'], proofs=15,
                              receipt=dict(host_waits=1, d2h=1, d2h_bytes=199, capture_wait_ns=5,
                                           device_gpu_us=[1_500_000, 0, 0, 0]))
                (folder / 'm8-jobs.json').write_text(json.dumps(native))
            accepted = accepted_run(run, blocks, pins, False, require_native=True)
            self.assertEqual(accepted['preinstalled_blocks'], 2)
            self.assertEqual([row['device_gpu_s'] for row in accepted['requests']], [1.5, 1.5])
            native_manifest['resources'] = {'guest_execution': 'cuda'}
            (run / 'manifest.json').write_text(json.dumps(native_manifest))
            census = {str(block): {'instructions': 11, 'cpu_segments': 1} for block in blocks}
            with self.assertRaisesRegex(ValueError, 'verified retirements'):
                accepted_run(run, blocks, pins, False, require_native=True, guest_census=census)
            for request, receiver in zip(receipt['requests'], receipt['receivers']):
                receiver['verified_total_steps'] = 12
                job = Path(request['proof']).parents[1] / 'm8-jobs.json'
                native = json.loads(job.read_text())
                native['guest'] = dict(mode='cuda_staged', retirements=12, launches=1,
                                       trace_readback_bytes=240, state_readback_bytes=416,
                                       device_allocated_bytes=8192, device_us=5, compact_d2d_bytes=576)
                job.write_text(json.dumps(native))
            path.write_text(json.dumps(receipt))
            measured = accepted_run(run, blocks, pins, False, require_native=True, guest_census=census)
            self.assertIn('staged trace ingress', measured['backend'])
            self.assertEqual([row['guest']['retirements'] for row in measured['requests']], [12, 12])
            self.assertEqual([row['census_instructions'] for row in measured['requests']], [11, 11])
            receipt['live_service']['qualified_latency'] = True
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'proofless device geometry setup'):
                accepted_run(run, blocks, pins, True, require_native=True, guest_census=census)
            ready = receipt['service_ready_record']
            ready['setup_mode'] = 'device_geometry'
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'on-demand proofless device geometry setup'):
                accepted_run(run, blocks, pins, True, require_native=True, guest_census=census)
            live['preinstalled_blocks'] = ready['preinstalled_blocks'] = 0
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'measured block admission'):
                accepted_run(run, blocks, pins, True, require_native=True, guest_census=census)
            for request in receipt['requests']:
                job = Path(request['proof']).parents[1] / 'm8-jobs.json'
                native = json.loads(job.read_text())
                native['admission_ns'] = 3
                job.write_text(json.dumps(native))
            with self.assertRaisesRegex(ValueError, 'requires resident guest ingress'):
                accepted_run(run, blocks, pins, True, require_native=True, guest_census=census)
            receipt['live_service']['qualified_latency'] = False
            path.write_text(json.dumps(receipt))
            for i, request in enumerate(receipt['requests']):
                job = Path(request['proof']).parents[1] / 'm8-jobs.json'
                native = json.loads(job.read_text())
                native['guest'].update(mode='cuda_resident', launches=2, trace_readback_bytes=0, trace_readback_ns=0,
                                       projection_ns=0, retired_host_metadata_bytes=0, retired_host_ls_bytes=0)
                job.write_text(json.dumps(native))
                if i == 0:
                    with self.assertRaisesRegex(ValueError, 'modes changed'):
                        accepted_run(run, blocks, pins, False, require_native=True, guest_census=census)
            with self.assertRaisesRegex(ValueError, 'installed service clock'):
                accepted_run(run, blocks, pins, False, require_native=True, guest_census=census)
            clock_file.write_text(json.dumps(clock))
            measured = accepted_run(run, blocks, pins, False, require_native=True, guest_census=census)
            self.assertIn('resident trace ingress', measured['backend'])
            self.assertEqual(measured['launch_to_last_root_s'], 9)
            self.assertEqual(measured['setup_mode'], 'device_geometry')
            self.assertEqual(measured['preinstalled_blocks'], 0)
            self.assertEqual([row['admission_s'] for row in measured['requests']], [3e-9, 3e-9])
            census[str(blocks[1])]['terminal_instructions'] = 13
            with self.assertRaisesRegex(ValueError, 'verified retirements consistent'):
                accepted_run(run, blocks, pins, False, require_native=True, guest_census=census)
            del census[str(blocks[1])]['terminal_instructions']
            self.assertAlmostEqual(measured['startup_amortized_proving_mean_s'],
                                   measured['proving_mean_s'] + receipt['service_start_s'] / len(blocks))
            receipt['receivers'][1]['verified_total_steps'] = 11
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'inconsistent CUDA'):
                accepted_run(run, blocks, pins, False, require_native=True, guest_census=census)
            (run / 'manifest.json').write_text(json.dumps(manifest))
            receipt['service_ready_record'] = dict(live_admission=True, pid=123)
            live['preinstalled_blocks'] = 0
            receipt['receivers'][1]['proof'] = receipt['receivers'][0]['proof']
            path.write_text(json.dumps(receipt))
            with self.assertRaises(RuntimeError):
                accepted_run(run, blocks, pins, False)

    def test_gpu_native_gate_rejects_host_and_non_attested_receipts(self):
        from scripts.rv64.rv64_live_service import native_request_metrics

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm8-jobs.json'
            request = dict(block=17, proof=str(Path(tmp) / '17' / 'proof'))
            receipt = dict(schema='rv64-m8-gpu-native-request/v1', block=17, sequence=2,
                           request_wall_ns=9_000, proofs=6, receipt=dict(device_gpu_us=[8, 0], host_waits=1, d2h=1, d2h_bytes=10, capture_wait_ns=2))
            path.write_text(json.dumps(receipt))
            self.assertEqual(native_request_metrics(request, 2, 9_000)['device_gpu_s'], 8e-6)
            for patch in [dict(schema='mixed'), dict(sequence=1), dict(proofs=0),
                          dict(receipt=receipt['receipt'] | {'device_gpu_us': [8, 2]}),
                          dict(receipt=receipt['receipt'] | {'host_waits': 2}),
                          dict(receipt=receipt['receipt'] | {'d2h': 2})]:
                with self.subTest(patch=patch), self.assertRaisesRegex(ValueError, 'single-GPU'):
                    path.write_text(json.dumps(receipt | patch))
                    native_request_metrics(request, 2, 9_000)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                native_request_metrics(request, 2, 9_000)
            guest = dict(mode='cuda_staged', retirements=12, launches=1, trace_readback_bytes=240,
                         state_readback_bytes=416, device_allocated_bytes=8192, device_us=5,
                         compact_d2d_bytes=576)
            path.write_text(json.dumps(receipt | {'guest': guest}))
            self.assertEqual(native_request_metrics(request, 2, 9_000, 12)['guest'], guest)
            for executor in ('host_witness', 'cuda'):
                path.write_text(json.dumps(receipt | {'guest': guest | {'executor': executor}}))
                metrics = native_request_metrics(request, 2, 9_000, 12)
                self.assertEqual(metrics['guest_execution_and_handoff_s'], 5e-6)
                self.assertEqual('guest_device_s' in metrics, executor == 'cuda')
            with self.assertRaisesRegex(ValueError, 'unexpected CUDA'):
                native_request_metrics(request, 2, 9_000)
            for bad in (None, guest | {'trace_readback_bytes': 15}, guest | {'retirements': 11},
                        guest | {'compact_d2d_bytes': 0}):
                path.write_text(json.dumps(receipt | {'guest': bad}))
                with self.assertRaisesRegex(ValueError, 'inconsistent CUDA'):
                    native_request_metrics(request, 2, 9_000, 12)
            guest.update(mode='cuda_resident', trace_readback_bytes=0, trace_readback_ns=0,
                         projection_ns=0, retired_host_metadata_bytes=0, retired_host_ls_bytes=0)
            path.write_text(json.dumps(receipt | {'guest': guest}))
            self.assertEqual(native_request_metrics(request, 2, 9_000, 12, 1)['guest'], guest)
            with self.assertRaisesRegex(ValueError, 'inconsistent CUDA'):
                native_request_metrics(request, 2, 9_000, 12, 2)
            for key in ('trace_readback_bytes', 'trace_readback_ns', 'projection_ns',
                        'retired_host_metadata_bytes', 'retired_host_ls_bytes'):
                path.write_text(json.dumps(receipt | {'guest': guest | {key: 1}}))
                with self.assertRaisesRegex(ValueError, 'reports host trace work'):
                    native_request_metrics(request, 2, 9_000, 12)
            replay = guest | dict(replayed_windows=1, retained_capture_bytes=256)
            path.write_text(json.dumps(receipt | {'guest': replay}))
            self.assertEqual(native_request_metrics(request, 2, 9_000, 12, 1)['guest'], replay)
            for bad in (dict(replayed_windows=0), dict(replayed_windows=2), dict(retained_capture_bytes=191),
                        dict(retained_capture_bytes=8193)):
                path.write_text(json.dumps(receipt | {'guest': replay | bad}))
                with self.assertRaisesRegex(ValueError, 'inconsistent retained CUDA'):
                    native_request_metrics(request, 2, 9_000, 12, 1)

    def test_on_demand_cuda_counts_discovery_and_only_unretained_execution(self):
        from scripts.rv64.rv64_live_service import native_request_metrics

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm8-jobs.json'
            request = dict(block=17, proof=str(Path(tmp) / '17' / 'proof'))
            report = dict(schema='rv64-m8-gpu-native-request/v1', block=17, sequence=0,
                          request_wall_ns=9_000, admission_ns=3_000, proofs=6,
                          receipt=dict(device_gpu_us=[8], host_waits=1, d2h=1, d2h_bytes=10, capture_wait_ns=2))
            guest = dict(mode='cuda_resident', retirements=12, trace_readback_bytes=0, trace_readback_ns=0,
                         projection_ns=0, retired_host_metadata_bytes=0, retired_host_ls_bytes=0,
                         state_readback_bytes=3200, device_allocated_bytes=8192, device_us=5,
                         compact_d2d_bytes=576)
            for windows, retained, launches in ((0, 0, 6), (1, 96, 5), (3, 512, 3)):
                with self.subTest(windows=windows):
                    evidence = guest | dict(replayed_windows=windows, retained_capture_bytes=retained, launches=launches)
                    path.write_text(json.dumps(report | {'guest': evidence}))
                    got = native_request_metrics(request, 0, 9_000, 12, 3, on_demand=True)
                    self.assertEqual(got['guest'], evidence)
            retry = guest | dict(replayed_windows=3, retained_capture_bytes=512, launches=5, discovery_retry_launches=2)
            path.write_text(json.dumps(report | {'guest': retry}))
            self.assertEqual(native_request_metrics(request, 0, 9_000, 12, 3, on_demand=True)['guest'], retry)
            for patch in (dict(launches=3, replayed_windows=1, retained_capture_bytes=96),
                          dict(launches=5, replayed_windows=3, retained_capture_bytes=512, discovery_retry_launches=-2),
                          dict(launches=6, replayed_windows=-1, retained_capture_bytes=0),
                          dict(launches=2, replayed_windows=4, retained_capture_bytes=512),
                          dict(launches=5, replayed_windows=1, retained_capture_bytes=15),
                          dict(launches=6, replayed_windows=0, retained_capture_bytes=96)):
                path.write_text(json.dumps(report | {'guest': guest | patch}))
                with self.assertRaisesRegex(ValueError, 'inconsistent retained CUDA'):
                    native_request_metrics(request, 0, 9_000, 12, 3, on_demand=True)

    def test_command_sets_one_visible_device_and_one_service(self):
        manifest = dict(source_root='/src', assets={k: '/bin/' + k for k in ('library', 'driver', 'receiver')},
                        preset={'common': {'CUDA_VISIBLE_DEVICES': '0,1'}, 'arms': {'baseline': {}},
                                'security_profile': 'test', 'operational': {'baseline': {'recursion_max_pending_authorities': 17}},
                                'measurement': dict(quiet_load_max=1, quiet_steal_max=0, quiet_timeout_s=1, quiet_stable_samples=1)},
                        gpus=1)
        cmd, _ = bench.stack.command(manifest, 'baseline', 0, 'test', False, live_blocks=[12, 11])
        self.assertEqual(cmd[2:5], ['daemon', '12', '11'])
        self.assertIn('--live', cmd)
        for option, expected in [('--repeat', '1'), ('--receive', 'all'), ('--gpus', '1')]:
            self.assertEqual(cmd[cmd.index(option) + 1], expected)
        self.assertIn('CUDA_VISIBLE_DEVICES=0', cmd)
        self.assertNotIn('CUDA_VISIBLE_DEVICES=0,1', cmd)
        cuda, _ = bench.stack.command(dict(manifest, guest_execution='cuda'), 'baseline', 0, 'test', False, live_blocks=[12, 11])
        self.assertEqual(cuda[cuda.index('--guest-execution') + 1], 'cuda')
        for bad, blocks in [(2, [11]), (1, [11, 11]), (1, [])]:
            with self.subTest(bad=bad, blocks=blocks), self.assertRaises(ValueError):
                bench.stack.command(dict(manifest, gpus=bad), 'baseline', 0, 'test', False, live_blocks=blocks)


class ParsingTests(unittest.TestCase):
    def test_fields_parse_ints_floats_and_braced_lists(self):
        row = bench.fields('a=1 b=2.5 c=true d={ 1, 2 } e=-3')
        self.assertEqual(row, {'a': 1, 'b': 2.5, 'c': 'true', 'd': '{ 1, 2 }', 'e': -3})

    def test_split_requests_ends_each_request_at_its_marker(self):
        text = '\n'.join(['boot', NODE, 'GPU_RESIDENT_BLOCK block=1 trial=0 iteration=0 request_wall_ns=5', NODE2,
                          'GPU_RESIDENT_BLOCK block=1 trial=1 iteration=1 request_wall_ns=4', 'shutdown'])
        parts = bench.split_requests(text)
        self.assertEqual(len(parts), 2)
        self.assertIn(NODE, parts[0])
        self.assertEqual(parts[1][0], NODE2)

    def test_nested_install_proofs_do_not_become_request_metrics(self):
        text = '\n'.join(['GPU_RESIDENT_SERVICE ready init_ns=1 runtime=inner', NODE,
                          'GPU_RESIDENT_BLOCK block=1 iteration=0 request_wall_ns=900',
                          'GPU_RESIDENT_SERVICE ready init_ns=1000 runtime=m8-gpu-native',
                          'actual-first', 'GPU_RESIDENT_BLOCK block=1 iteration=0 request_wall_ns=5',
                          'actual-second', 'GPU_RESIDENT_BLOCK block=1 iteration=1 request_wall_ns=4'])
        parts = bench.split_requests(text)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0][0], 'actual-first')
        self.assertEqual(parts[1][0], 'actual-second')
        self.assertNotIn(NODE, [line for part in parts for line in part])

    def test_job_metrics_count_only_job_lines(self):
        jobs = bench.job_metrics([NODE, NODE2, HOST, MEM, FIXED, NATIVE, EVENT])
        node = jobs['outer_node']
        self.assertEqual(node['count'], 2)  # NODE_HOST / NODE_MEM are not nodes
        self.assertEqual(node['launches']['sum'], 1119)
        self.assertEqual(node['launches']['max'], 1118)
        self.assertEqual(node['host_waits']['median'], 1)
        self.assertAlmostEqual(node['gpu_s'], 0.818, places=3)
        self.assertEqual(node['peak_bytes'], 4553800832 * 4)
        self.assertEqual(jobs['outer_fixed']['launches']['sum'], 53)
        native = jobs['native_stage']
        self.assertEqual(native['count'], 1)  # event= lines are not jobs
        self.assertAlmostEqual(native['gpu_s'], 0.2305, places=4)
        self.assertIsNone(native['launches'])
        self.assertEqual(bench.job_metrics(['nothing here']), {})

    def test_request_metrics_stages_and_derived_tail(self):
        row = bench.request_metrics(request(1, 20.13, 20.57), COMBINED, [NODE])
        self.assertEqual(row['proving_s'], 20.13)
        self.assertEqual(row['request_s'], 20.57)
        self.assertEqual(row['native_window_s'], 4.75)
        self.assertEqual(row['recursion_after_native_s'], 13.2)
        self.assertAlmostEqual(row['tail_s'], 20.13 - 2.0 - 0.14 - 4.75 - 13.2, places=3)
        self.assertEqual(row['nodes'], 19)
        self.assertEqual(row['jobs']['outer_node']['count'], 1)
        partial = bench.request_metrics(request(0, 30.0, 31.0), {'capture_ns': 1}, [])
        self.assertIsNone(partial['tail_s'])
        self.assertIsNone(partial['nodes'])

    def test_native_receipt_keeps_driver_spans_separate_from_unknown_stage_partitions(self):
        native = {'schema': 'rv64-m8-gpu-native-request/v1', 'receipt': {
            'host_waits': 1, 'd2h': 2, 'd2h_bytes': 1234,
            'device_gpu_us': [8_000_000, 9_000_000, 0, 0],
            'device_enqueue_ns': [7_000_000_000, 8_000_000_000, 0, 0],
            'device_gate_ns': [10_000_000, 20_000_000, 0, 0],
            'device_memory_serializations': [3, 4, 0, 0]}}
        row = bench.request_metrics(request(1, 10.0, 10.1), {'native_prove_ns': None}, [], native)
        self.assertEqual(row['driver']['device_span_s'], [8.0, 9.0])
        self.assertEqual(row['driver']['device_enqueue_s'], [7.0, 8.0])
        self.assertEqual(row['driver']['memory_serializations'], [3, 4])
        self.assertEqual(row['driver']['host_waits'], 1)
        self.assertIsNone(row['driver']['job_slab_d2h'])
        self.assertIsNone(row['driver']['host_callbacks'])
        self.assertIsNone(row['native_window_s'])
        self.assertIsNone(row['tail_s'])
        self.assertEqual(row['jobs'], {})

    def test_profile_metrics_busy_and_per_node_counts(self):
        out = bench.profile_metrics({'kernel_busy_gpu_seconds': 10.0, 'any_busy_gpu_seconds': 12.0},
                                    {'cudaLaunchKernel': 19000, 'cudaStreamSynchronize': 190}, 19, 25.0)
        self.assertEqual(out['kernel_gpu_s'], 10.0)
        self.assertEqual(out['gpu_busy'], 0.2)
        self.assertEqual(out['launches_per_node'], 1000.0)
        self.assertEqual(out['syncs_per_node'], 10.0)
        self.assertEqual(bench.profile_metrics(None, None, 19, 25.0), {'profiled_proving_s': 25.0})


class AggregationTests(unittest.TestCase):
    def test_aggregate_arm_medians_exclude_failed_and_profiled_services(self):
        services = [service('outer', 'headline', 20.0, device_peak_bytes=30e9, nvml_kernel_active_percent=14.0),
                    service('outer', 'headline', 22.0, device_peak_bytes=31e9, nvml_kernel_active_percent=16.0),
                    service('outer', 'headline', 21.0),
                    service('outer', 'headline', 99.0, status='failed'),
                    service('outer', 'profile', 40.0, profile={'kernel_gpu_s': 9.0})]
        services[2]['requests'][1]['jobs'] = bench.job_metrics([NODE, NODE2])
        arm = bench.aggregate_arm(services)
        self.assertEqual(arm['warm']['n'], 3)
        self.assertEqual(arm['warm']['proving_s'], 21.0)
        self.assertEqual(arm['cold']['proving_s'], 42.0)
        self.assertEqual(arm['device_peak_bytes_sampled'], 31e9)
        self.assertEqual(arm['nvml_kernel_active_percent'], 15.0)
        self.assertEqual(arm['profile_warm'], {'kernel_gpu_s': 9.0})
        self.assertEqual(arm['per_node_peak_bytes'], 4553800832 * 4)
        self.assertEqual(arm['job_launches_per_node'], 559.5)
        self.assertEqual(arm['job_syncs_per_node'], 1)

    def test_aggregate_totals_count_only_done_blocks_and_compare_m3(self):
        rows = bench.basket()
        small, dev = rows[0], next(r for r in rows if r['block'] == 24628607)
        blocks = {small['block']: {'block': small['block'], 'status': 'done', 'parity': {'requests_identical': 4},
                                   'services': [service('default', 'headline', 10.0), service('outer', 'headline', 16.0)]},
                  dev['block']: {'block': dev['block'], 'status': 'done',
                                 'services': [service('default', 'headline', 13.0)]},
                  rows[-1]['block']: {'block': rows[-1]['block'], 'status': 'failed',
                                      'services': [service('default', 'headline', 50.0)]}}
        result = bench.aggregate(blocks, ['default', 'outer'], rows)
        self.assertEqual(result['schema'], bench.SCHEMA)
        self.assertEqual(len(result['blocks']), 15)
        totals = result['totals']
        self.assertEqual(totals['default']['blocks'], 2)  # the failed block is excluded
        self.assertEqual(totals['default']['warm_proving_s_sum'], 23.0)
        self.assertEqual(totals['default']['cold_proving_s_sum'], 46.0)
        self.assertAlmostEqual(totals['default']['m3_request_s_sum_same_blocks'], small['request_s'] + dev['request_s'], places=2)
        self.assertEqual(totals['outer']['blocks'], 1)
        by_block = {b['block']: b for b in result['blocks']}
        self.assertEqual(by_block[24628607]['m7']['warm_proving_s'], 12.9)
        self.assertEqual(by_block[rows[5]['block']]['status'], 'not_run')
        text = bench.markdown(result)
        self.assertIn('| 26235156 |', text)
        self.assertIn('10.00 / 20.00', text)
        self.assertIn('| outer | 1 |', text)
        self.assertEqual(text.count('\n| 24628607 | default |'), 1)

    def test_distribution_is_nearest_rank(self):
        values = list(range(1, 16))  # 15 blocks
        self.assertEqual(bench.distribution(values), {'n': 15, 'p50': 8, 'p90': 14, 'max': 15})
        self.assertEqual(bench.distribution([3.0]), {'n': 1, 'p50': 3.0, 'p90': 3.0, 'max': 3.0})
        self.assertIsNone(bench.distribution([None]))

    def test_arm_failure_reason_prefers_the_named_error(self):
        log = 'GATE boot\nsomething failed here\nerror: GpuNativeNativeShapeHookMissing\n/x/driver_serve.zig:103: in install\n'
        self.assertEqual(bench.arm_failure_reason(log), 'error: GpuNativeNativeShapeHookMissing')
        self.assertEqual(bench.arm_failure_reason('a\nservice never READY: M8RequestProviderNotLanded\n'),
                         'service never READY: M8RequestProviderNotLanded')

    def test_producer_tail_reads_plain_and_gzipped_stderr(self):
        import gzip, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(bench.producer_tail(tmp), '')
            (Path(tmp) / 'producer.stderr.gz').write_bytes(gzip.compress(b'x\nerror: GpuNativeNativeShapeHookMissing\n'))
            text = 'service exited before READY (1)\n' + bench.producer_tail(tmp)
            self.assertEqual(bench.arm_failure_reason(text), 'error: GpuNativeNativeShapeHookMissing')

    def test_arm_failures_are_reported_and_excluded_from_totals(self):
        rows = bench.basket()
        small = rows[0]
        failure = {'kind': 'service', 'reason': 'error: GpuNativeNativeShapeHookMissing', 'run': 'r', 'log': 'l'}
        blocks = {small['block']: {'block': small['block'], 'status': 'done', 'arm_failures': {'gpu-native': failure},
                                   'services': [service('default', 'headline', 10.0),
                                                dict(arm='gpu-native', role='headline', status='failed')]}}
        result = bench.aggregate(blocks, ['default', 'gpu-native'], rows)
        self.assertEqual(result['arm_failures'], {'gpu-native': [small['block']]})
        self.assertNotIn('gpu-native', result['totals'])
        self.assertEqual(result['totals']['default']['warm_proving_s'], {'n': 1, 'p50': 10.0, 'p90': 10.0, 'max': 10.0})
        text = bench.markdown(result)
        self.assertIn('10.00 / 20.00 | FAIL |', text)
        self.assertIn('gpu-native FAILED (service)', text)
        self.assertNotIn('| gpu-native | -', text)

    def test_estimate_is_monotonic_in_block_size_and_labelled(self):
        rows = bench.basket()
        estimate = bench.estimate_walls(rows, ['default', 'outer', 'native', 'hybrid', 'jobs'], 3)
        self.assertEqual(estimate['label'], 'E')
        walls = [estimate['arms']['default']['blocks'][r['block']]['warm_s'] for r in rows]
        self.assertEqual(walls, sorted(walls))
        dev = estimate['arms']['default']['blocks'][24628607]
        self.assertAlmostEqual(dev['warm_s'], 12.8, places=1)
        self.assertGreater(estimate['arms']['outer']['gpu_wall_s'], estimate['arms']['default']['gpu_wall_s'])
        lighter = bench.estimate_walls(rows, ['default'], 1, profile=False)
        self.assertLess(lighter['arms']['default']['gpu_wall_s'], estimate['arms']['default']['gpu_wall_s'])


class ToleratedArmTests(unittest.TestCase):
    def setUp(self):
        self.free_bytes = bench.free_bytes
        bench.free_bytes = lambda path: 1 << 50

    def tearDown(self):
        bench.free_bytes = self.free_bytes

    def bench(self, tmp, outcomes):
        b = object.__new__(bench.Bench)
        b.out = Path(tmp)
        b.args = argparse.Namespace(rounds=3, arms=['default', 'gpu-native'], tolerate_arm_failure=['gpu-native'],
                                    receivers='reference', profile='warm', quiet_retries=0, min_free_gb=0)
        calls = []

        def fake_service(block, row, arm, role, round_index, attempt):
            calls.append((arm, role, round_index))
            log = Path(tmp) / f'{arm}-{role}-{round_index}.log'
            status = outcomes(arm, role)
            log.write_text('error: GpuNativeNativeShapeHookMissing\n' if status != 'ok' else 'ok\n')
            return {'arm': arm, 'role': role, 'round': round_index, 'run': str(Path(tmp) / 'none' / f'{arm}{round_index}'),
                    'status': status, 'log': str(log), 'exit_code': 0 if status == 'ok' else 1,
                    'requests': [{'proving_s': 1.0, 'nodes': 1}, {'proving_s': 1.0, 'nodes': 1}]}

        def fake_parity(block, entry, state):
            state['reference'] = state.get('reference') or {'run': entry['run']}

        b.service = fake_service
        b.parity = fake_parity
        b.receive = lambda block, run: [{'accepted': True}]
        b.gpu_seconds = lambda entry, ref: {'kernel_gpu_s': 1.0}
        b.prune = lambda run, keep_proofs=False: None
        return b, calls

    def test_failed_tolerated_arm_is_recorded_once_and_the_block_completes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'blocks').mkdir()
            b, calls = self.bench(tmp, lambda arm, role: 'failed' if arm == 'gpu-native' else 'ok')
            state = b.block({'block': 26235156})
        self.assertEqual(state['status'], 'done')
        self.assertEqual(calls.count(('gpu-native', 'headline', 0)), 1)
        self.assertNotIn(('gpu-native', 'headline', 1), calls)
        self.assertNotIn(('gpu-native', 'profile', 0), calls)
        self.assertEqual([c for c in calls if c[0] == 'default'],
                         [('default', 'headline', 0), ('default', 'headline', 1), ('default', 'headline', 2), ('default', 'profile', 0)])
        self.assertEqual(state['arm_failures']['gpu-native']['reason'], 'error: GpuNativeNativeShapeHookMissing')

    def test_untolerated_arm_failure_still_stops(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'blocks').mkdir()
            b, _ = self.bench(tmp, lambda arm, role: 'failed' if arm == 'default' else 'ok')
            with self.assertRaises(bench.Stop) as stop:
                b.block({'block': 26235156})
        self.assertIn('GpuNativeNativeShapeHookMissing', str(stop.exception))


if __name__ == '__main__':
    unittest.main()
