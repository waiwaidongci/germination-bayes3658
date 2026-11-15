import math

from fastapi import HTTPException
from ..models import BayesRun, ReplantPlan
from ..utils.serialize import record
from ..utils.fingerprint import fingerprint
from .audit_service import log
from .replant_service import solve_replant, MAX_BATCHES, MAX_CAP

PLAN_VERSION = "integer-branch-bound-betabinomial-replant-v1"


def _is_finite_positive_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _is_finite_number_in_open_unit(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0.0 < value < 1.0


def list_plans(db):
    return [record(p) for p in db.query(ReplantPlan).order_by(ReplantPlan.id.desc())]


def get_plan(db, pid):
    row = db.get(ReplantPlan, pid)
    if not row:
        raise HTTPException(404, "补种方案不存在")
    return row


def _completed_run(db, run_id):
    run = db.get(BayesRun, run_id)
    if not run or run.status != "completed":
        raise HTTPException(409, "只能基于已完成的批次概率研究制定补种方案")
    if not run.result or not run.result.get("lots"):
        raise HTTPException(409, "研究结果缺少冻结后验，无法制定补种方案")
    return run


def _build_lot_inputs(run, entries):
    """把前端逐批输入与研究冻结后验对齐，冻结后验形状只取自完成研究结果。"""
    result = run.result
    by_code = {row["lot"]["code"]: row for row in result["lots"]}
    if len(by_code) != len(result["lots"]):
        raise HTTPException(422, "研究结果批次编号重复，拒绝制定方案")
    seen = set()
    inputs = []
    for e in entries:
        code = e.lot_code
        if code in seen:
            raise HTTPException(422, f"批次 {code} 重复填写")
        seen.add(code)
        row = by_code.get(code)
        if row is None:
            raise HTTPException(422, f"批次 {code} 不属于选定研究")
        cap = e.available_seeds
        cost = e.cost_per_seed
        if type(cap) is not int or not 0 <= cap <= MAX_CAP:
            raise HTTPException(422, f"批次 {code} 可用粒数须为0–{MAX_CAP}整数")
        if not _is_finite_positive_number(cost):
            raise HTTPException(422, f"批次 {code} 每粒成本须为有限正数")
        inputs.append({
            "code": code,
            "cap": cap,
            "cost": float(cost),
            "post_alpha": float(row["post_alpha"]),
            "post_beta": float(row["post_beta"]),
            "frozen_lot": row["lot"],
            "frozen_count": row["count"],
        })
    missing = set(by_code) - seen
    if missing:
        raise HTTPException(422, "缺少批次填报：" + "、".join(sorted(missing)))
    if not 1 <= len(inputs) <= MAX_BATCHES:
        raise HTTPException(422, f"批次数须在1–{MAX_BATCHES}之间")
    return inputs


def create_plan(db, data, user):
    run = _completed_run(db, data.run_id)
    lot_inputs = _build_lot_inputs(run, data.lots)
    if type(data.target) is not int or data.target < 0:
        raise HTTPException(422, "总萌发目标须为非负整数")
    if not _is_finite_number_in_open_unit(data.required_probability):
        raise HTTPException(422, "达标概率须为(0,1)内数值")
    if not (isinstance(data.time_limit, (int, float)) and math.isfinite(data.time_limit) and 0.0 < data.time_limit <= 60.0):
        raise HTTPException(422, "时限须为(0,60]秒")

    frozen_snapshot = {
        "plan_version": PLAN_VERSION,
        "run_id": run.id,
        "run_name": run.name,
        "fit_hash": run.result.get("fit_hash"),
        "input_hash": run.result.get("input_hash"),
        "fit_alpha": run.result["fit"]["alpha"],
        "fit_beta": run.result["fit"]["beta"],
        "threshold": run.threshold,
        "future_n": run.future_n,
        "study": run.result.get("study"),
        "lots": [{
            "lot_code": x["code"], "available_seeds": x["cap"], "cost_per_seed": x["cost"],
            "post_alpha": x["post_alpha"], "post_beta": x["post_beta"],
            "lot": x["frozen_lot"], "count": x["frozen_count"],
        } for x in lot_inputs],
        "target": data.target,
        "required_probability": data.required_probability,
        "time_limit": data.time_limit,
    }
    frozen_snapshot["snapshot_hash"] = fingerprint({
        "version": PLAN_VERSION,
        "run_input_hash": frozen_snapshot["input_hash"],
        "fit_hash": frozen_snapshot["fit_hash"],
        "lots": [{"code": x["lot_code"], "cap": x["available_seeds"], "cost": x["cost_per_seed"],
                  "a": x["post_alpha"], "b": x["post_beta"]} for x in frozen_snapshot["lots"]],
        "target": data.target, "required_probability": data.required_probability,
    })

    solver_inputs = [{"code": x["code"], "cap": x["cap"], "cost": x["cost"],
                     "post_alpha": x["post_alpha"], "post_beta": x["post_beta"]} for x in lot_inputs]
    try:
        result = solve_replant(solver_inputs, data.target, data.required_probability, data.time_limit)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    result["snapshot_hash"] = frozen_snapshot["snapshot_hash"]

    row = ReplantPlan(
        name=data.name, run_id=run.id, target=data.target,
        required_probability=data.required_probability, time_limit=data.time_limit,
        status=result["status"], frozen_snapshot=frozen_snapshot, result=result, owner=user.name,
    )
    db.add(row)
    db.flush()
    log(db, user.email, "replant.create", row.id)
    db.commit()
    return record(row)
