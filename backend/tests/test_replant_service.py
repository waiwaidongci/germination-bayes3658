"""补种求解器测试：小规模穷举对照、均值贪心反例、限时降级与边界语义。

运行：cd backend && python -m pytest tests/test_replant_service.py -q
（纯算法测试不需要数据库或 Redis。）
"""
import itertools

import numpy as np
import pytest

from src.services.replant_service import (
    beta_binomial_pmf, solve_replant,
    build_factors, aggregate, PROB_TOL, MAX_BATCHES, MAX_CAP,
)


def brute_force_optimum(lots, target, req):
    """独立穷举实现：枚举全部整数分配，返回最低可行成本（无可行解返回 None）。"""
    caps = [l["cap"] for l in lots]
    costs = [l["cost"] for l in lots]
    best_cost = np.inf
    best_alloc = None
    for alloc in itertools.product(*[range(c + 1) for c in caps]):
        cost = sum(x * w for x, w in zip(alloc, costs))
        if cost >= best_cost - 1e-12:
            continue
        pmf = np.ones(1)
        for x, lot in zip(alloc, lots):
            pmf = np.convolve(pmf, beta_binomial_pmf(int(x), lot["post_alpha"], lot["post_beta"]))
        tail = 0.0 if target >= len(pmf) else float(pmf[target:].sum())
        if tail + 1e-10 >= req:
            best_cost = cost
            best_alloc = alloc
    return (best_alloc, best_cost) if best_alloc is not None else (None, np.inf)


def random_lots(rng, j_count, cap_high):
    a = rng.uniform(0.4, 15, size=j_count)
    b = rng.uniform(0.4, 15, size=j_count)
    caps = rng.integers(0, cap_high + 1, size=j_count)
    costs = np.round(rng.uniform(0.5, 6, size=j_count), 2)
    return [
        {"code": f"L{j}", "cap": int(caps[j]), "cost": float(costs[j]),
         "post_alpha": float(a[j]), "post_beta": float(b[j])}
        for j in range(j_count)
    ]


@pytest.mark.parametrize("seed", list(range(40)))
def test_branch_bound_matches_brute_force(seed):
    """小规模穷举对照：状态、最优成本、可行方案实际概率全部一致。"""
    rng = np.random.default_rng(1000 + seed)
    j_count = int(rng.integers(1, 5))
    lots = random_lots(rng, j_count, 6)
    total_cap = sum(l["cap"] for l in lots)
    target = int(rng.integers(0, max(1, total_cap + 1)))
    req = float(rng.choice([0.5, 0.8, 0.9, 0.95]))

    result = solve_replant(lots, target, req, time_limit=10.0)
    bf_alloc, bf_cost = brute_force_optimum(lots, target, req)

    if bf_alloc is None:
        assert result["status"] == "infeasible"
        assert result["lower_bound"] is None and result["upper_bound"] is None
        assert result["optimal"] is False
        return

    assert result["status"] == "optimal"
    assert result["optimal"] is True
    assert result["gap"] == pytest.approx(0.0, abs=1e-9)
    assert result["lower_bound"] == pytest.approx(bf_cost, abs=1e-9)
    assert result["upper_bound"] == pytest.approx(bf_cost, abs=1e-9)
    assert result["total_cost"] == pytest.approx(bf_cost, abs=1e-9)
    # 上界方案确实满足概率要求，且实际概率由独立卷积复算
    sol = {r["lot_code"]: r["seeds"] for r in result["allocation"]}
    pmf = np.ones(1)
    for lot in lots:
        pmf = np.convolve(pmf, beta_binomial_pmf(sol[lot["code"]], lot["post_alpha"], lot["post_beta"]))
    tail = float(pmf[target:].sum())
    assert tail + 1e-9 >= req
    assert result["actual_probability"] == pytest.approx(tail, abs=1e-8)


def test_mean_greedy_counterexample():
    """固定夹具证明“按期望萌发/成本贪心”不保证最优，甚至不保证可行。

    批次0单价低但后验高度不确定；均值贪心把预算几乎全投给批次0，
    得到的合计达标概率不足；分支定界给出成本更高但确实达标的最优方案。
    """
    lots = [
        {"code": "cheap", "cap": 6, "cost": 1.0, "post_alpha": 1.2, "post_beta": 1.8},
        {"code": "steady", "cap": 6, "cost": 2.0, "post_alpha": 9.0, "post_beta": 1.0},
    ]
    target, req = 5, 0.9

    # 均值/成本比贪心：逐粒选择期望萌发增量比最高的批，直到期望合计达到目标
    means_rate = [
        l["post_alpha"] / (l["post_alpha"] + l["post_beta"]) for l in lots
    ]
    alloc = [0, 0]
    expected = 0.0
    order = sorted(range(2), key=lambda j: -means_rate[j] / lots[j]["cost"])
    for j in order:
        while expected < target and alloc[j] < lots[j]["cap"]:
            alloc[j] += 1
            expected += means_rate[j]
    pmf = np.ones(1)
    for x, lot in zip(alloc, lots):
        pmf = np.convolve(pmf, beta_binomial_pmf(x, lot["post_alpha"], lot["post_beta"]))
    greedy_tail = float(pmf[target:].sum())

    result = solve_replant(lots, target, req, time_limit=10.0)
    bf_alloc, bf_cost = brute_force_optimum(lots, target, req)
    assert bf_alloc is not None
    # 反例成立条件：贪心方案不可行，或贪心成本严格高于最优成本
    assert greedy_tail + 1e-9 < req or sum(
        x * l["cost"] for x, l in zip(alloc, lots)
    ) > bf_cost + 1e-9
    assert result["status"] == "optimal"
    assert result["total_cost"] == pytest.approx(bf_cost, abs=1e-9)
    assert result["actual_probability"] + 1e-9 >= req


def test_timeout_returns_bounds_without_optimal_claim():
    """限时降级：open 状态返回已证下界与可行上界，绝不标记最优，界限保持有序。"""
    rng = np.random.default_rng(2026)
    lots = [
        {"code": f"L{j:02d}", "cap": 200,
         "cost": float(round(rng.uniform(0.5, 4.0), 2)),
         "post_alpha": float(rng.uniform(20, 120)),
         "post_beta": float(rng.uniform(8, 40))}
        for j in range(MAX_BATCHES)
    ]
    target, req = 900, 0.9
    result = solve_replant(lots, target, req, time_limit=0.05, node_limit=10_000_000)
    assert result["status"] == "open"
    assert result["timed_out"] is True
    assert result["optimal"] is False
    assert result["lower_bound"] is not None and result["upper_bound"] is not None
    assert result["lower_bound"] <= result["upper_bound"] + 1e-9
    assert result["gap"] >= -1e-9
    # 上界方案必须真实可行
    assert result["actual_probability"] + PROB_TOL >= req
    assert 0 <= result["total_seeds"] <= sum(l["cap"] for l in lots)


def test_node_limit_degrades_to_open():
    """节点上限同样触发 open 语义，与超时等价处理。"""
    lots = [{"code": f"H{j}", "cap": 200, "cost": 1.0 + 0.01 * j,
             "post_alpha": 8.0, "post_beta": 4.0} for j in range(MAX_BATCHES)]
    result = solve_replant(lots, 2500, 0.9, time_limit=60.0, node_limit=2000)
    assert result["status"] == "open"
    assert result["optimal"] is False
    assert result["lower_bound"] <= result["upper_bound"] + 1e-9
    assert result["actual_probability"] + PROB_TOL >= 0.9


def test_proven_infeasible_distinct_from_open():
    """已证无解（满配仍不达标）必须标 infeasible，与搜索未决 open 区分。"""
    lots = [{"code": f"L{j}", "cap": 5, "cost": 1.0,
             "post_alpha": 2.0, "post_beta": 20.0} for j in range(4)]
    result = solve_replant(lots, 18, 0.95, time_limit=5.0)
    assert result["status"] == "infeasible"
    assert result["infeasible_reason"] == "max_allocation_probability_below_requirement"
    assert result["optimal"] is False
    assert result["lower_bound"] is None and result["upper_bound"] is None
    assert result["max_tail_probability"] is not None


def test_target_above_capacity_is_infeasible():
    lots = [{"code": "A", "cap": 10, "cost": 1.0, "post_alpha": 5.0, "post_beta": 2.0}]
    result = solve_replant(lots, 11, 0.5, time_limit=5.0)
    assert result["status"] == "infeasible"
    assert result["infeasible_reason"] == "target_above_total_capacity"


def test_zero_target_prefers_zero_allocation():
    lots = [{"code": "A", "cap": 10, "cost": 1.0, "post_alpha": 5.0, "post_beta": 2.0},
            {"code": "B", "cap": 8, "cost": 2.0, "post_alpha": 4.0, "post_beta": 3.0}]
    result = solve_replant(lots, 0, 0.5, time_limit=5.0)
    assert result["status"] == "optimal"
    assert result["total_cost"] == 0.0
    assert all(r["seeds"] == 0 for r in result["allocation"])
    assert result["actual_probability"] == pytest.approx(1.0)


def test_zero_capacity_lot_skipped():
    lots = [
        {"code": "empty", "cap": 0, "cost": 1.0, "post_alpha": 3.0, "post_beta": 2.0},
        {"code": "work", "cap": 10, "cost": 2.0, "post_alpha": 5.0, "post_beta": 5.0},
    ]
    result = solve_replant(lots, 3, 0.8, time_limit=5.0)
    rows = {r["lot_code"]: r for r in result["allocation"]}
    assert rows["empty"]["seeds"] == 0
    assert rows["work"]["seeds"] >= 1
    assert result["actual_probability"] + PROB_TOL >= 0.8


def test_deterministic_results():
    lots = [{"code": f"L{j:02d}", "cap": 50, "cost": 1.0 + 0.03 * j,
             "post_alpha": 8.0 + j, "post_beta": 4.0} for j in range(12)]
    r1 = solve_replant(lots, 200, 0.9, time_limit=1.0)
    r2 = solve_replant(lots, 200, 0.9, time_limit=1.0)
    assert [r["seeds"] for r in r1["allocation"]] == [r["seeds"] for r in r2["allocation"]]
    assert r1["total_cost"] == r2["total_cost"]


def test_rescaled_pmf_keeps_deep_tail_representable():
    """混合表示在深卷积下不得整体下溢为 0（回归普通卷积的缺陷）。"""
    caps = np.full(MAX_BATCHES, 200)
    alphas = np.full(MAX_BATCHES, 2.0)
    betas = np.full(MAX_BATCHES, 5.0)
    factors, _ = build_factors(caps, alphas, betas)
    h = aggregate(caps.astype(np.int64), factors)
    assert np.isfinite(h.p).all() and h.p.sum() > 0
    # 与 log 空间真值对照
    from scipy.special import logsumexp
    log_pmf = np.zeros(1)
    for a, b in zip(alphas, betas):
        q = beta_binomial_pmf(200, float(a), float(b))
        log_pmf = _log_convolve(log_pmf, np.log(np.maximum(q, 1e-320)))
    truth = float(np.exp(logsumexp(log_pmf[1500:])))
    assert h.tail(1500) == pytest.approx(truth, rel=1e-6, abs=1e-10)


def _log_convolve(x, y):
    nx, ny = len(x), len(y)
    if nx < ny:
        x, y = y, x
        nx, ny = ny, nx
    out = np.full(nx + ny - 1, -np.inf)
    for j in range(ny):
        out[j:j + nx] = np.logaddexp(out[j:j + nx], x + y[j])
    return out


def test_input_validation():
    base = [{"code": "A", "cap": 10, "cost": 1.0, "post_alpha": 3.0, "post_beta": 2.0}]
    with pytest.raises(ValueError):
        solve_replant(base, 1, 0.9, time_limit=0)
    with pytest.raises(ValueError):
        solve_replant(base, 1, 0.0, time_limit=5.0)
    with pytest.raises(ValueError):
        solve_replant([{**base[0], "post_alpha": -1.0}], 1, 0.9, time_limit=5.0)
    with pytest.raises(ValueError):
        solve_replant([{**base[0], "cap": MAX_CAP + 1}], 1, 0.9, time_limit=5.0)
    too_many = [{"code": f"L{j}", "cap": 5, "cost": 1.0,
                 "post_alpha": 3.0, "post_beta": 2.0} for j in range(MAX_BATCHES + 1)]
    with pytest.raises(ValueError):
        solve_replant(too_many, 1, 0.9, time_limit=5.0)
