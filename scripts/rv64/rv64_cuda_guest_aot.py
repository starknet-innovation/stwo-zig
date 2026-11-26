"""Lower bounded straight-line regions of a hash-pinned RV64 ELF for CUDA."""

import hashlib
import json
import re
import struct


def signed(value, bits):
    bit = 1 << (bits - 1)
    return (value ^ bit) - bit


def decode(raw):
    """Emit constants only for straight-line instructions. Device admission
    independently compares these operands to the receiver's decoded ROM.
    """
    op, rd = raw & 127, (raw >> 7) & 31
    f3, f7 = (raw >> 12) & 7, raw >> 25
    a, b = (raw >> 15) & 31, (raw >> 20) & 31
    imm = signed(raw >> 20, 12)
    if op == 3 and f3 != 7:
        return ("LB LH LW LD LBU LHU LWU".split()[f3], rd, a, 0, imm)
    if op == 0x23 and f3 < 4:
        return ("SB SH SW SD".split()[f3], 0, a, b, signed(f7 << 5 | rd, 12))
    if op in (0x17, 0x37):
        return ("LUI" if op == 0x37 else "AUIPC", rd, 0, 0, signed(raw & 0xfffff000, 32))
    word = op in (0x1b, 0x3b)
    if op in (0x33, 0x3b) and (not word or f3 in (0, 1, 5)):
        name = {0: "SUB", 5: "SRA"}.get(f3) if f7 == 32 else (
            "ADD SLL SLT SLTU XOR SRL OR AND".split()[f3] if f7 == 0 else None)
        if name:
            return (name + ("W" if word else ""), rd, a, b, 0)
    if op in (0x13, 0x1b) and (not word or f3 in (0, 1, 5)):
        name = "ADDI SLLI SLTI SLTIU XORI SRLI ORI ANDI".split()[f3]
        if f3 in (1, 5):
            shift_bits = 5 if word else 6
            control = (raw >> 20) >> shift_bits
            if f3 == 5 and control == 1 << (30 - 20 - shift_bits):
                name = "SRAI"
            elif control:
                raise ValueError(f"unsupported guest instruction 0x{raw:08x}")
            imm = (raw >> 20) & ((1 << shift_bits) - 1)
        return (name + ("W" if word else ""), rd, a, 0, imm)
    raise ValueError(f"unsupported guest instruction 0x{raw:08x}")


def executable_segments(elf):
    if (len(elf) < 64 or elf[:7] != b"\x7fELF\x02\x01\x01" or
            struct.unpack_from("<HH", elf, 16) != (2, 243)):
        raise ValueError("guest AOT requires a little-endian RV64 executable ELF")
    table, = struct.unpack_from("<Q", elf, 32)
    stride, count = struct.unpack_from("<HH", elf, 54)
    if stride != 56 or table > len(elf) or stride * count > len(elf) - table:
        raise ValueError("invalid guest ELF program header table")
    segments = []
    for i in range(count):
        kind, flags, offset, address, _, size, memory, _ = struct.unpack_from("<II6Q", elf, table + i * stride)
        if kind != 1 or not flags & 1:
            continue
        if flags & 2 or not size or (address | size) & 3 or size != memory or offset + size > len(elf) or address + size >= 1 << 32:
            raise ValueError("invalid executable guest ELF segment")
        segments.append((address, address + size, offset))
    segments.sort()
    if not segments or any(a[1] > b[0] for a, b in zip(segments, segments[1:])):
        raise ValueError("ambiguous executable guest ELF segments")
    return segments


def generate(profile, output):
    config = json.loads(profile.read_text())
    if config.get("schema") != "rv64-guest-aot/v1" or not re.fullmatch(r"[0-9a-f]{64}", config.get("elf_sha256", "")):
        raise ValueError("invalid guest AOT profile identity")
    elf = (profile.parent / config["elf"]).read_bytes()
    digest = hashlib.sha256(elf).hexdigest()
    if digest != config["elf_sha256"]:
        raise ValueError("guest AOT ELF fingerprint mismatch")
    segments = executable_segments(elf)
    regions = config["regions"]
    if not 1 <= len(regions) <= 32:
        raise ValueError("guest AOT requires 1 to 32 regions")
    staged = []
    for region in regions:
        pc, count = region["pc"], region["instructions"]
        if (type(pc) is not int or type(count) is not int or pc & 3 or not 2 <= count <= 16 or
                pc >> 12 != (pc + 4 * count - 1) >> 12):
            raise ValueError("invalid guest AOT region geometry")
        for start, end, offset in segments:
            if start <= pc and pc + 4 * count <= end:
                words = struct.unpack_from(f"<{count}I", elf, offset + pc - start)
                staged.append((pc, tuple(decode(word) for word in words)))
                break
        else:
            raise ValueError("guest AOT region outside executable file bytes")
    staged.sort()
    if any(pc + 4 * len(code) > following for (pc, code), (following, _) in zip(staged, staged[1:])):
        raise ValueError("overlapping guest AOT regions")
    lines = [f"// RV64 guest ELF SHA-256 {digest}", "#define STWO_GUEST_STATIC_REGIONS(X)"]
    for ordinal, (pc, code) in enumerate(staged):
        slot = (pc - segments[0][0]) // 4
        steps = [f"ConstantInstruction<Op::{op}, {rd}, {a}, {b}, UINT64_C({imm % (1 << 64)})>" for op, rd, a, b, imm in code]
        lines.append(f"    X({ordinal}, {pc}u, {slot}u, " + ", ".join(steps) + ")")
    regions_source = " \\\n".join(lines[1:])
    output.write_text(f"{lines[0]}\n{regions_source}\n")
    return {"elf_sha256": digest, "regions": len(staged), "instructions": sum(len(code) for _, code in staged),
            "catalogue_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
