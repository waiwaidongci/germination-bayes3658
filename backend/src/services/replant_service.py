"""补种整数分配的最低成本分支定界求解。

工作模型（沿用已完成研究冻结的条件后验与批间独立假设）：
每批 j 补种 x_j 粒（0≤x_j≤可用粒数 cap_j，整数），该批萌发数
K_j | x_j ~ BetaBinomial(x_j, a_j, b_j)，其中 (a_j,b_j) 是研究冻结后验形状；
跨批独立，合计萌发 S=ΣK_j。目标：

    min  Σ c_j x_j  （c_j 为每粒成本）
    s.t. Pr(S ≥ T) ≥ p        （T 总萌发目标，p 指定达标概率）
         0 ≤ x_j ≤ cap_j，x_j 为整数

状态语义（严格区分）：
- optimal   ：搜索在时限内完整闭合，成本下界与可行方案上界相等，才标记最优；
- open      ：超时或节点上限触发，返回已证成本下界 lower_bound 与可行方案成本上界 upper_bound，
              上界方案保证满足概率要求，但不声称最优；
- infeasible：在搜索前即证明无可行解（目标超过总可用粒数，或全部满配仍达不到概率要求）。

数值口径：合计 PMF 用“普通概率—按最大值缩放”混合表示，避免深卷积在普通浮点下下溢为 0；
缩放只改变常数因子，尾部概率经归一化后与真实概率一致（对可行性判定所需的概率区间足够精确）。
"""
import numpy as np
from scipy.special import betaln, gammaln, logsumexp
from time import perf_counter

PROB_TOL = 1e-10
COST_EPS = 1e-12
TIME_MASK = 4095
MAX_BATCHES = 28
MAX_CAP = 200
DEFAULT_NODE_LIMIT = 2_000_000


def beta_binomial_pmf(m, a, b):
    """标准归一化 BB(m,a,b) 概率质量，取值 0..m；m=0 为 0 处点质量。"""
    if type(m) is not int or m < 0:
        raise ValueError("补种粒数须为非负整数")
    if m == 0:
        return np.ones(1)
    if not np.isfinite([a, b]).all() or min(a, b) <= 0:
        raise ValueError("Beta形状须为有限正数")
    j = np.arange(m + 1)
    lp = gammaln(m + 1) - gammaln(j + 1) - gammaln(m - j + 1) + betaln(a + j, b + m - j) - betaln(a, b)
    z = logsumexp(lp)
    if not np.isfinite(z):
        raise ValueError("预测PMF计算失败")
    return np.exp(lp - z)


class HybridPMF:
    """合计 PMF 的混合表示。

    scaled=False：p 为真实概率（和为 1）；
    scaled=True ：普通卷积即将下溢，p 改为 p_true / max(p_true)，
                  尾部概率用 sum(tail)/sum(total) 恢复（常数缩放因子相消）。
    """
    __slots__ = ("p", "scaled")

    def __init__(self, p, scaled):
        self.p = p
        self.scaled = scaled

    def tail(self, threshold):
        if threshold >= len(self.p):
            return 0.0
        if not self.scaled:
            return float(self.p[threshold:].sum())
        total = float(self.p.sum())
        return float(self.p[threshold:].sum()) / total if total > 0 else 0.0


def convolve(h1, h2):
    if not h1.scaled and not h2.scaled:
        r = np.convolve(h1.p, h2.p)
        if r.sum() >= 0.999:
            return HybridPMF(r / r.sum(), False)
        return HybridPMF(np.convolve(h1.p / h1.p.max(), h2.p / h2.p.max()), True)
    q1 = h1.p if h1.scaled else h1.p / h1.p.max()
    q2 = h2.p if h2.scaled else h2.p / h2.p.max()
    return HybridPMF(np.convolve(q1, q2), True)


def support_cost_lower_bound(caps, costs, depth, need):
    """可接纳成本下界：从 depth 起各批再补 need 粒支持量的最小可能花费。

    批已按每粒成本升序排列，故顺序取满最便宜的粒即得下界；容量不足返回 inf。
    """
    if need <= 0:
        return 0.0
    total = 0.0
    for j in range(depth, len(caps)):
        take = min(int(caps[j]), need)
        total += take * float(costs[j])
        need -= take
        if need == 0:
            return total
    return np.inf


def build_factors(caps, alphas, betas):
    """预计算每批在所有补种粒数 m=0..cap 下的 BB(m) 混合 PMF 与均值/方差。"""
    factors, moments = [], []
    for cap, a, b in zip(caps, alphas, betas):
        row = [HybridPMF(np.ones(1), False)]
        stat = [(0.0, 0.0)]
        for m in range(1, int(cap) + 1):
            q = beta_binomial_pmf(m, float(a), float(b))
            if q.sum() >= 0.999 and np.isfinite(q).all():
                row.append(HybridPMF(q, False))
            else:
                j = np.arange(m + 1)
                lp = (gammaln(m + 1) - gammaln(j + 1) - gammaln(m - j + 1)
                      + betaln(a + j, b + m - j) - betaln(a, b))
                row.append(HybridPMF(np.exp(lp - lp.max()), True))
            mean = float(q.dot(np.arange(m + 1)))
            var = float((q * np.arange(m + 1) ** 2).sum() - mean * mean)
            stat.append((mean, max(var, 0.0)))
        factors.append(row)
        moments.append(stat)
    return factors, moments


def aggregate(alloc, factors):
    h = HybridPMF(np.ones(1), False)
    for x, row in zip(alloc, factors):
        h = convolve(h, row[int(x)])
    return h


def coordinate_descent_heuristic(caps, costs, factors, target, req, passes=3):
    """可行初始方案：从满配（预检保证可行）出发，逐批二分求保持全局可行的最小粒数。

    单批萌发分布关于补种粒数随机占优递增，故可行性对 x_j 单调，二分成立。
    """
    j_count = len(caps)
    alloc = caps.copy().astype(np.int64)
    for _ in range(passes):
        changed = False
        for j in range(j_count):
            if alloc[j] == 0:
                continue
            others = HybridPMF(np.ones(1), False)
            for k in range(j_count):
                if k != j:
                    others = convolve(others, factors[k][int(alloc[k])])
            cur = int(alloc[j])

            def feasible(x):
                return convolve(others, factors[j][x]).tail(target) + PROB_TOL >= req

            if not feasible(cur):
                continue
            lo, hi = 0, cur
            while lo < hi:
                mid = (lo + hi) // 2
                if feasible(mid):
                    hi = mid
                else:
                    lo = mid + 1
            if lo < cur:
                alloc[j] = lo
                changed = True
        if not changed:
            break
    return alloc


def solve_replant(lot_inputs, target, required_prob, time_limit, node_limit=DEFAULT_NODE_LIMIT):
    """求解最低成本整数补种分配。

    lot_inputs: 每批 dict，含 code/cap/cost/post_alpha/post_beta。
    返回完整求解记录（状态、界限、逐批分配、实际概率）。
    """
    if not 1 <= len(lot_inputs) <= MAX_BATCHES:
        raise ValueError(f"批次数须在1–{MAX_BATCHES}之间")
    if type(target) is not int or target < 0:
        raise ValueError("总萌发目标须为非负整数")
    if type(required_prob) not in (int, float) or not 0.0 < required_prob < 1.0:
        raise ValueError("达标概率须为(0,1)内有限数值")
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("时限须为正数")
    if type(node_limit) is not int or node_limit <= 0:
        raise ValueError("节点上限须为正整数")

    order = sorted(range(len(lot_inputs)), key=lambda j: (lot_inputs[j]["cost"], lot_inputs[j]["code"]))
    ordered = [lot_inputs[j] for j in order]
    caps = np.array([x["cap"] for x in ordered], dtype=np.int64)
    costs = np.array([x["cost"] for x in ordered], dtype=float)
    alphas = np.array([x["post_alpha"] for x in ordered], dtype=float)
    betas = np.array([x["post_beta"] for x in ordered], dtype=float)
    if np.any(caps < 0) or np.any(caps > MAX_CAP):
        raise ValueError(f"每批可用粒数须为0–{MAX_CAP}整数")
    if np.any(~np.isfinite(costs)) or np.any(costs <= 0):
        raise ValueError("每粒成本须为有限正数")
    if np.any(~np.isfinite(alphas)) or np.any(~np.isfinite(betas)) or np.any(alphas <= 0) or np.any(betas <= 0):
        raise ValueError("冻结后验形状须为有限正数")

    factors, moments = build_factors(caps, alphas, betas)
    total_capacity = int(caps.sum())
    if target > total_capacity:
        return _infeasible(ordered, caps, costs, target, required_prob, time_limit,
                           "target_above_total_capacity")

    remaining_cap = np.zeros(len(caps) + 1, dtype=np.int64)
    for t in range(len(caps) - 1, -1, -1):
        remaining_cap[t] = remaining_cap[t + 1] + caps[t]

    deadline = perf_counter() + float(time_limit)
    full = aggregate(caps, factors)
    full_tail = full.tail(target)
    if full_tail + PROB_TOL < required_prob:
        return _infeasible(ordered, caps, costs, target, required_prob, time_limit,
                           "max_allocation_probability_below_requirement", full_tail)

    # Cantelli（单边切比雪夫）常数：μ−T ≥ √(κ²σ²) 是 Pr(S≥T)≥req 的充分条件。
    kappa2 = required_prob / (1.0 - required_prob)
    heuristic_alloc = coordinate_descent_heuristic(caps, costs, factors, target, required_prob)
    best_alloc = heuristic_alloc.copy()
    best_cost = float(heuristic_alloc.dot(costs))
    best_tail = aggregate(best_alloc, factors).tail(target)
    state = {"nodes": 0, "timed_out": False, "proven_lb": [], "frontier_lb": [], "certified": 0}

    def dfs(depth, current_cost, h, prefix, mu, var):
        nonlocal best_cost, best_alloc, best_tail
        state["nodes"] += 1
        if state["nodes"] > node_limit or (state["nodes"] & TIME_MASK == 0 and perf_counter() >= deadline):
            state["timed_out"] = True
            return
        # Cantelli 充分可行性证书：当前已提交分配（后续批补 0）即保证达标，整支可剪。
        if (mu - target) >= np.sqrt(kappa2 * var) - 1e-12:
            if current_cost < best_cost - COST_EPS:
                comp = np.zeros(len(caps), dtype=np.int64)
                comp[:depth] = prefix
                best_cost = current_cost
                best_alloc = comp
                state["certified"] += 1
            state["proven_lb"].append(current_cost)
            return
        if depth == len(caps):
            tail = h.tail(target)
            if tail + PROB_TOL >= required_prob and current_cost < best_cost - COST_EPS:
                best_cost = current_cost
                best_alloc = np.array(prefix, dtype=np.int64)
                best_tail = tail
            return
        committed_support = len(h.p) - 1
        if committed_support + int(remaining_cap[depth]) < target:
            return
        need = max(0, target - committed_support)
        lb = current_cost + support_cost_lower_bound(caps, costs, depth, need)
        if not np.isfinite(lb) or lb > best_cost + COST_EPS:
            if np.isfinite(lb):
                state["proven_lb"].append(lb)
            return
        for x in range(0, int(caps[depth]) + 1):
            mx, vx = moments[depth][x]
            dfs(depth + 1, current_cost + x * costs[depth],
                convolve(h, factors[depth][x]), prefix + (x,), mu + mx, var + vx)
            if state["timed_out"]:
                state["frontier_lb"].append(lb)
                return
        state["proven_lb"].append(lb)

    dfs(0, 0.0, HybridPMF(np.ones(1), False), (), 0.0, 0.0)

    if state["timed_out"]:
        status = "open"
        candidates = state["proven_lb"] + state["frontier_lb"] + [best_cost]
        lower_bound = min(v for v in candidates if np.isfinite(v))
    else:
        status = "optimal"
        lower_bound = best_cost
    best_tail = aggregate(best_alloc, factors).tail(target)

    rows = []
    for pos, lot in enumerate(ordered):
        x = int(best_alloc[pos])
        mean_x, _ = moments[pos][x]
        rows.append({
            "lot_code": lot["code"], "cap": int(caps[pos]), "cost_per_seed": float(costs[pos]),
            "seeds": x, "line_cost": round(x * float(costs[pos]), 10),
            "expected_germination": float(mean_x),
            "post_alpha": float(alphas[pos]), "post_beta": float(betas[pos]),
        })
    return {
        "status": status,
        "infeasible_reason": None,
        "target": target,
        "required_probability": float(required_prob),
        "time_limit": float(time_limit),
        "elapsed_nodes": int(state["nodes"]),
        "timed_out": bool(state["timed_out"]),
        "certified_nodes": int(state["certified"]),
        "lower_bound": float(lower_bound),
        "upper_bound": float(best_cost),
        "gap": float(best_cost - lower_bound),
        "optimal": bool(status == "optimal" and best_cost - lower_bound <= 1e-9),
        "allocation": rows,
        "total_cost": float(best_cost),
        "actual_probability": float(best_tail),
        "total_capacity": total_capacity,
        "total_seeds": int(sum(r["seeds"] for r in rows)),
        "total_expected_germination": float(sum(r["expected_germination"] for r in rows)),
        "method": "integer-branch-bound-betabinomial-replant-v1",
        "assumptions": "各批补种萌发数为冻结后验Beta形状下的Beta-Binomial，批间独立；概率条件于点估超参数，不含超参数估计误差。",
    }


def _infeasible(ordered, caps, costs, target, req, time_limit, reason, full_tail=None):
    rows = [{
        "lot_code": lot["code"], "cap": int(c), "cost_per_seed": float(w),
        "seeds": 0, "line_cost": 0.0, "expected_germination": 0.0,
        "post_alpha": lot["post_alpha"], "post_beta": lot["post_beta"],
    } for lot, c, w in zip(ordered, caps, costs)]
    return {
        "status": "infeasible", "infeasible_reason": reason,
        "target": target, "required_probability": float(req), "time_limit": float(time_limit),
        "elapsed_nodes": 1, "timed_out": False, "certified_nodes": 0,
        "lower_bound": None, "upper_bound": None, "gap": None, "optimal": False,
        "allocation": rows, "total_cost": None, "actual_probability": full_tail,
        "total_capacity": int(caps.sum()), "total_seeds": 0, "total_expected_germination": 0.0,
        "max_tail_probability": full_tail,
        "method": "integer-branch-bound-betabinomial-replant-v1",
        "assumptions": "各批补种萌发数为冻结后验Beta形状下的Beta-Binomial，批间独立；概率条件于点估超参数，不含超参数估计误差。",
    }
