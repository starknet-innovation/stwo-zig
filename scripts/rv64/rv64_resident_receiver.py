"""Verify completed recursive GPU requests in a fresh CPU-only process."""
import json
import os
from pathlib import Path

from scripts.rv64 import rv64_m3_pod as P
from scripts.rv64 import rv64_resident_basket as R
from scripts.rv64.rv64_validation import proof_artifact_binding

B = R.B


def use_grouped_verifier(resources, proof):
    """Choose from the declared roster. The chosen verifier authenticates all declarations."""
    if resources.get("grouped_service_recursion") is not True:
        return False
    truths = B.load(proof / "bundle.json")["truths"]
    return [worker["declaration"]["family"] for worker in truths[-3:]] == ["add", "sha256", "arith256_mod"]


def receive_one(plan, block, proof, out_root, workers, *, grouped=None):
    job = next(j for j in plan["jobs"] if j["block"] == block)
    report = B.load(proof / "report.json")
    count = report["cpu_proofs"] + report["truth_proofs"]
    if grouped is None:
        grouped = use_grouped_verifier(job["resources"], proof)
    if grouped:
        R.recursive_snapshot(proof, count, grouped=True)
    out = out_root / f"receiver-{block}-{proof.parent.parent.name}-{P.now()}"
    out.mkdir()
    request = B.load(job["request"])
    request["workers"] = workers
    request["output_directory"] = str(out / "capture")
    request_path = out / "receiver-request.json"
    B.atomic_json(request_path, request)
    environment = {k: v for k, v in os.environ.items() if k not in ("STWO_ZIG_RV64_SETUP_CACHE", "STWO_RV64_OUTER_PLAN_CLASS_STORE")}
    environment = R.receiver_environment(environment, out / "gpu-disabled")
    environment = P.format_environment(environment, plan)
    environment["STWO_RV64_FOLD_CLASS_CACHE"] = "0"
    receiver = Path(plan["assets"]["recursive_receiver"])
    before = proof_artifact_binding(proof)
    mode = "verify-service-group-candidate" if grouped else "verify-recursion-candidate"
    command = ["nice", "-n", "10", *B.command(receiver, plan["profile"], mode, request_path, proof,
                                               job["resources"]["fixed_setup_bytes"], 32 << 30)]
    usage = B.measure(command, out, "strict-receiver", environment, cwd=plan["producer_cwd"])
    verdict = {"block": block, "proof": str(proof), "status": "rejected", "accepted": False, "receiver_sha256": P.sha(receiver),
               "process": usage, "directory": str(out),
               "isolation": {"CUDA_VISIBLE_DEVICES": "", "STWO_RV64_GPU_PCS_HELPER": "/bin/false",
                             "STWO_ZIG_RV64_SETUP_CACHE": "unset", "STWO_RV64_OUTER_PLAN_CLASS_STORE": "unset",
                             "STWO_RV64_FOLD_CLASS_CACHE": "0"}}
    try:
        if usage.get("exit_code") != 0:
            raise ValueError(f"receiver exit {usage.get('exit_code')}")
        received = B.load(out / "strict-receiver.stdout")
        R.check_trial(report, received["native"], job, plan["profile"])
        if received.get("accepted") is not True or received.get("completed_ethereum_block") is not True:
            raise ValueError("fresh recursive receiver rejected the complete statement")
        R.check_recursive_report(received["recursive"], count - (2 if grouped else 0), restored=True)
        after = proof_artifact_binding(proof)
        if after != before:
            raise ValueError("receiver changed producer artifacts")
        verdict.update(status="accepted", accepted=True, artifact_binding=after)
        if "total_steps" in received["native"]:
            verdict["verified_total_steps"] = received["native"]["total_steps"]
    except Exception as error:
        verdict["error"] = repr(error)
    return verdict


def receive_run(run, which="all", workers=16):
    """Fresh CPU-only receivers while the ArtifactStore directory is DELETED (moved out of reach)."""
    run = Path(run)
    receipt = B.load(run / "receipt.json")
    plan = B.load(run / "plan.json")
    store = plan.get("store", {}).get("directory")
    quarantine = None
    if store and Path(store).exists():
        quarantine = Path(str(store) + f".quarantine-{P.now()}")
        Path(store).rename(quarantine)
    try:
        if store and Path(store).exists():
            raise SystemExit("store still visible to the receiver")
        verdicts = []
        for entry in receipt["requests"]:
            if "error" in entry or (which == "first" and entry["trial"] != 0):
                continue
            verdict = receive_one(plan, entry["block"], Path(entry["proof"]), run, workers)
            verdict["trial"] = entry["trial"]
            verdict["store_present_during_receive"] = bool(store and Path(store).exists())
            verdicts.append(verdict)
            print(json.dumps({k: verdict[k] for k in ("block", "trial", "status")} | {"wall_s": round(verdict["process"]["wall_ns"] / 1e9, 1)}))
    finally:
        if quarantine is not None:
            quarantine.rename(store)
    receipt["receivers"] = verdicts
    receipt["all_received"] = bool(verdicts) and all(v["accepted"] for v in verdicts)
    B.atomic_json(run / "receipt.json", receipt)
    return verdicts
