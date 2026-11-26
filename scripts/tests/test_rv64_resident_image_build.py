import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rv64 import rv64_cuda_resident_image_build as B


class ImagePlan(unittest.TestCase):
    def test_guest_aot_pins_elf_and_emits_only_admitted_straight_line_regions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            elf, manifest, inc = (root / name for name in ("guest.elf", "profile.json", "rv64_guest_aot.inc"))
            bytes_ = bytearray(132)
            bytes_[:7] = b"\x7fELF\x02\x01\x01"
            struct.pack_into("<HH", bytes_, 16, 2, 243)
            struct.pack_into("<Q", bytes_, 32, 64)
            struct.pack_into("<HH", bytes_, 54, 56, 1)
            struct.pack_into("<II6Q", bytes_, 64, 1, 5, 120, 4096, 4096, 12, 12, 4)
            struct.pack_into("<3I", bytes_, 120, 0x0000b283, 0xfff28293, 0x00000073)
            elf.write_bytes(bytes_)
            config = {"schema": "rv64-guest-aot/v1", "elf": elf.name, "elf_sha256": B.N.sha(elf),
                      "regions": [{"pc": 4096, "instructions": 2}]}
            manifest.write_text(json.dumps(config))
            receipt = B.N.GuestAot.generate(manifest, inc)
            self.assertEqual(receipt["instructions"], 2)
            self.assertEqual(receipt["catalogue_sha256"], B.N.sha(inc))
            header = inc.read_text()
            self.assertIn("X(0, 4096u, 0u, ConstantInstruction<Op::LD, 5, 1, 0, UINT64_C(0)>", header)
            self.assertIn("ConstantInstruction<Op::ADDI, 5, 5, 0, UINT64_C(18446744073709551615)>", header)
            for regions, error in [
                ([{"pc": 4096, "instructions": 3}], "unsupported guest instruction"),
                ([{"pc": 4104, "instructions": 2}], "outside executable"),
                (config["regions"] * 2, "overlapping guest AOT regions"),
            ]:
                manifest.write_text(json.dumps({**config, "regions": regions}))
                with self.assertRaisesRegex(ValueError, error):
                    B.N.GuestAot.generate(manifest, inc)
            manifest.write_text(json.dumps(config))
            elf.write_bytes(bytes_ + b"changed")
            with self.assertRaisesRegex(ValueError, "ELF fingerprint mismatch"):
                B.N.GuestAot.generate(manifest, inc)

    def test_split_core_keeps_original_power_offsets_and_local_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i in range(49):
                text = f'extern "C" __global__ void rv64_joined_domain_{i}(void){{}}'
                (root / f"case-{i}-domain.cu").write_text(text)
            source = root / "case-8-domain.cu"
            source.write_text('''#include "rv64/composition.cuh"
__device__ __noinline__ void rv64_joined_domain_8_part_0(void){use(powers+1136u);}
__device__ __noinline__ void rv64_joined_domain_8_part_1(void){use(powers+880u);}
extern "C" __global__ void rv64_joined_domain_8(void){uint32_t i=0;switch(blockIdx.y){}}
''')
            build = root / "build"
            build.mkdir()
            with mock.patch.object(B.Grouped, "partition", wraps=B.Grouped.partition) as partition:
                sources = B.split_core(root, build, [8], 32768)
            self.assertEqual(partition.call_args.args[-1], 32768)
            with self.assertRaisesRegex(ValueError, "positive"):
                B.split_core(root, build, [8], 0)
            self.assertEqual(len(sources), 51)  # 48 untouched + 2 fragments + dispatcher.
            first = (build / "core-parts/8/rv64_joined_domain_8_shard_0.cu").read_text()
            second = (build / "core-parts/8/rv64_joined_domain_8_shard_1.cu").read_text()
            self.assertIn("powers+1136u", first)
            self.assertNotIn("powers+880u", first)
            self.assertIn("powers+880u", second)
            self.assertIn("case 0u:rv64_joined_domain_8_part_1", second)
            header = (build / "real_composition_launches.h").read_text()
            self.assertIn("rv64_joined_domain_8_shard_1,32u,1u", header)
            self.assertNotIn("(const void*)rv64_joined_domain_8,", header)

    def test_cold_ssa_lowering_keeps_original_manifest_source_and_stable_cache_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "role.cu"
            text = "__device__ __noinline__ void role_part_0() {" + " " * 100 + "}"
            source.write_text(text)
            obj = root / "role.o"
            with mock.patch.object(B.Grouped, "bound_ssa_module", return_value={"instruction_slices": 2}):
                lowered, metadata = B.bounded_compile_source(source, obj, root, 16)
                self.assertNotEqual(lowered, source)
                self.assertEqual(source.read_text(), text)
                self.assertEqual(metadata["source_sha256"], B.N.sha(source))
                obj.write_bytes(b"candidate")
                self.assertEqual(B.bounded_compile_source(source, obj, root, 16)[0], lowered)
            other = root / "existing.cu"
            other.write_text(text)
            self.assertEqual(B.bounded_compile_source(other, obj, root, 16), (other, None))

    def test_prebounded_recovery_uses_original_authority_group_with_smaller_slices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "rv64_truth_domain_0_shard_0.cu"
            source.write_text("// volatile OM31 *ext_bank\n" + " " * (512 << 10) +
                              '__device__ __noinline__ void rv64_truth_domain_0_part_0(){}\nextern "C" __global__ void original_kernel(){}')
            original = source.read_text()
            (root / "case-0-domain.cu").write_text(
                'using namespace stwo::cuda::rv64;\n__device__ __noinline__ void rv64_truth_domain_0_part_0(){use(powers+1136u);mul(a,b);q::mul(1,2); }\nextern "C" __global__ void raw_kernel(){}')
            with mock.patch.object(B.Grouped, "bound_ssa_module", return_value={"instruction_slices": 1}) as lower:
                actual, metadata = B.bounded_compile_source(source, root / "missing.o", root, raw_directory=root)
            self.assertEqual(lower.call_args.args[1], 64)
            self.assertEqual(metadata["recovered_original_group"], 0)
            self.assertIn("powers+1136u", actual.read_text())
            self.assertIn("rv64_bounded_mul(a,b)", actual.read_text())
            self.assertIn("q::mul(1,2)", actual.read_text())
            self.assertEqual(metadata["bounded_multiply_calls"], 1)
            self.assertNotIn("volatile OM31 *ext_bank", actual.read_text())
            self.assertEqual(source.read_text(), original)

    def test_global_catalogue_is_required_distinct_and_exports_declared_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name = "stwo_rv64_global_contiguous_" + "a" * 64
            source = root / (name + ".cu")
            source.write_text("extern \"C\" int " + name + "(void){return 0;}")
            path = root / "required-symbols.json"
            manifest = {"schema": "rv64-resident-global-aot/v1", "symbols": [name]}
            path.write_text(json.dumps(manifest))
            self.assertEqual(B.global_sources(root), ([source], [name]))
            source.write_text("extern \"C\" int different(void){return 0;}")
            with self.assertRaisesRegex(ValueError, "missing global"):
                B.global_sources(root)
            manifest["symbols"] = [name, name]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "distinct"):
                B.global_sources(root)

    def test_outer_catalogue_requires_complete_order_and_authority_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            groups = []
            for i in range(12):
                filename = f"outer-group-{i}.cu"
                (root / filename).write_text(str(i))
                groups.append({"id": i, "filename": filename, "program_identity": "a" * 64, "source_identity": "b" * 64})
            (root / "outer-dispatch.cu").write_text("dispatch")
            manifest = {"profile": "native-om31-circuit", "geometry_independent": True, "groups": groups}
            path = root / "outer-manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(len(B.outer_sources(root)), 13)
            (root / "outer-group-12.cu").write_text("mixed")
            groups.append({"id": 12, "filename": "outer-group-12.cu", "program_identity": "c" * 64,
                           "source_identity": "d" * 64, "component": 1, "part": 0, "first": 0, "count": 6})
            path.write_text(json.dumps(manifest))
            self.assertEqual(len(B.outer_sources(root)), 14)
            groups[-1]["component"] = 2
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "mixed arithmetic"):
                B.outer_sources(root)
            groups.pop()
            groups[11]["id"] = 10
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "coverage"):
                B.outer_sources(root)

    def test_all_core_roles_required_and_group_geometry_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i in range(49):
                text = f'extern "C" __global__ void rv64_joined_domain_{i}(void){{}}'
                if i == 41:
                    text += f'void rv64_joined_domain_{i}_part_0(void){{}}'
                (root / f"case-{i}-domain.cu").write_text(text)
            sources, header = B.core_catalog(root)
            self.assertEqual(len(sources), 49)
            self.assertEqual(header.count("(const void*)"), 49)
            groups = header.split("real_root_group_sizes[]={")[1].split("}")[0].split(",")
            self.assertEqual(groups[41], "32")
            self.assertEqual(groups[0], "0")
            wide = root / "case-49-domain.cu"
            wide.write_text('extern "C" __global__ void rv64_joined_domain_49(void){}')
            wide_sources, wide_header = B.core_catalog(root)
            self.assertEqual(wide_sources[:-1], sources)
            self.assertEqual(wide_sources[-1], wide)
            self.assertIn("rv64_joined_domain_48,(const void*)rv64_joined_domain_49", wide_header)
            unknown = root / "case-50-domain.cu"
            unknown.touch()
            with self.assertRaisesRegex(ValueError, "unexpected CPU role"):
                B.core_catalog(root)
            unknown.unlink()
            sources[48].unlink()
            with self.assertRaises(FileNotFoundError):
                B.core_catalog(root)

    def test_catalogue_exact_coverage_hashes_and_no_ambiguous_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, parts = root / "raw", root / "parts"
            raw.mkdir()
            parts.mkdir()
            roles = []
            role_count = B.Grouped.TRUTH_ROLE_COUNT
            for i in range(role_count):
                source = parts / f"role-{i}.cu"
                source.write_text(str(i))
                roles.append({"input": f"/producer/case-{i}-domain.cu", "modules": [{"source": str(source), "sha256": B.N.sha(source)}]})
            manifest = parts / "partition.json"
            manifest.write_text(json.dumps({"roles": roles}))
            self.assertEqual(len(B.truth_sources(raw, parts)), role_count)
            (parts / "role-14.cu").write_text("changed")
            with self.assertRaisesRegex(ValueError, "partition changed"):
                B.truth_sources(raw, parts)
            (parts / "role-14.cu").write_text("14")
            (raw / "role-14.cu").write_text("14")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                B.truth_sources(raw, parts)
            (raw / "role-14.cu").unlink()
            manifest.write_text(json.dumps({"roles": roles[:-1]}))
            with self.assertRaisesRegex(ValueError, f"all {role_count} truth roles"):
                B.truth_sources(raw, parts)


if __name__ == "__main__":
    unittest.main()
