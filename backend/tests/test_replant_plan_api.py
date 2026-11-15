"""补种方案的服务与 HTTP 集成测试：持久化、冻结不随来源变化、校验、鉴权。

使用内存 SQLite 与 FastAPI 依赖覆盖，不需要 PostgreSQL/Redis。
"""
import os

os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config.database import Base, get_db
from src.main import app
from src.models import User, GerminationStudy, SeedLot, LotCount, BayesRun, ReplantPlan
from src.schemas.replant import ReplantPlanInput, ReplantLotInput
from src.services.replant_plan_service import create_plan

engine = create_engine(
    "sqlite+pysqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
Session = sessionmaker(bind=engine, expire_on_commit=False)
Base.metadata.create_all(engine)


@pytest.fixture
def db():
    with Session() as session:
        yield session


def make_run(db):
    import uuid
    code = "ST" + uuid.uuid4().hex[:8]
    study = GerminationStudy(code=code, name="研究", description="d", source="csv")
    db.add(study)
    db.flush()
    lot_rows = []
    for j in range(3):
        lot = SeedLot(study_id=study.id, code=f"L{j}", name=f"批{j}", origin="o")
        db.add(lot)
        db.flush()
        db.add(LotCount(lot_id=lot.id, n=50 + j, k=40, note=""))
        a, b = 8.0 + j, 4.0
        lot_rows.append({
            "lot": {"id": lot.id, "code": lot.code},
            "count": {"id": j, "lot_id": lot.id, "n": 50 + j, "k": 40},
            "post_alpha": a, "post_beta": b,
        })
    user = User(email=f"a-{code}@x.demo", name="分析员", role="analyst", password_hash="x")
    db.add(user)
    db.flush()
    run = BayesRun(
        name="基线", study_id=study.id, threshold=0.7, future_n=50,
        input_snapshot={}, status="completed", owner=user.name,
        result={
            "algorithm_version": "v", "fit": {"alpha": 8.5, "beta": 3.4},
            "fit_hash": "fit123", "input_hash": "in456",
            "study": {"id": study.id, "code": "ST"}, "lots": lot_rows,
        },
    )
    db.add(run)
    db.commit()
    return run, user


def plan_payload(run):
    return ReplantPlanInput(
        name="补种方案A", run_id=run.id, target=8, required_probability=0.8,
        time_limit=5.0,
        lots=[
            ReplantLotInput(lot_code="L0", available_seeds=20, cost_per_seed=1.0),
            ReplantLotInput(lot_code="L1", available_seeds=20, cost_per_seed=1.5),
            ReplantLotInput(lot_code="L2", available_seeds=20, cost_per_seed=2.0),
        ],
    )


def test_create_persists_and_freezes_snapshot(db):
    run, user = make_run(db)
    payload = plan_payload(run)
    record = create_plan(db, payload, user)

    saved = db.get(ReplantPlan, record["id"])
    assert saved.status in ("optimal", "open", "infeasible")
    assert saved.frozen_snapshot["fit_hash"] == "fit123"
    assert saved.frozen_snapshot["snapshot_hash"]
    assert len(saved.frozen_snapshot["lots"]) == 3
    # 结果含逐批分配与实际概率、界限
    res = saved.result
    assert "allocation" in res and len(res["allocation"]) == 3
    assert "actual_probability" in res and "lower_bound" in res and "upper_bound" in res


def test_saved_plan_does_not_change_when_source_changes(db):
    run, user = make_run(db)
    record = create_plan(db, plan_payload(run), user)
    before = db.get(ReplantPlan, record["id"])
    before_hash = before.frozen_snapshot["snapshot_hash"]
    before_result = dict(before.result)

    # 来源计数被后续修改（模拟研究数据变化）
    count = db.query(LotCount).first()
    count.k = 5
    db.commit()

    after = db.get(ReplantPlan, record["id"])
    assert after.frozen_snapshot["snapshot_hash"] == before_hash
    assert after.result["total_cost"] == before_result["total_cost"]
    assert after.result["allocation"] == before_result["allocation"]


def test_missing_lot_rejected(db):
    run, user = make_run(db)
    payload = plan_payload(run)
    payload.lots = payload.lots[:2]  # 缺一批
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        create_plan(db, payload, user)
    assert exc.value.status_code == 422


def test_unknown_lot_rejected(db):
    run, user = make_run(db)
    payload = plan_payload(run)
    payload.lots[0].lot_code = "NOPE"
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        create_plan(db, payload, user)
    assert exc.value.status_code == 422


def test_non_completed_run_rejected(db):
    run, user = make_run(db)
    run.status = "failed"
    db.commit()
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        create_plan(db, plan_payload(run), user)
    assert exc.value.status_code == 409


def test_cap_and_cost_validation(db):
    run, user = make_run(db)
    from fastapi import HTTPException
    bad = plan_payload(run)
    bad.lots[0].available_seeds = 201
    with pytest.raises(HTTPException) as exc:
        create_plan(db, bad, user)
    assert exc.value.status_code == 422
    bad = plan_payload(run)
    bad.lots[0].cost_per_seed = 0
    with pytest.raises(HTTPException) as exc:
        create_plan(db, bad, user)
    assert exc.value.status_code == 422


# ---------- HTTP 层：鉴权与端到端 ----------

def override_db():
    with Session() as session:
        yield session


app.dependency_overrides[get_db] = override_db


class _Client:
    def __init__(self, user):
        self.user = user

    def __call__(self):
        # 覆盖依赖必须是零参数可调用对象；带 *args/**kwargs 会被 FastAPI 当作查询参数解析
        return self.user


from src.controllers.dependencies import current_user


def test_http_create_and_viewer_forbidden(db):
    run, analyst = make_run(db)
    viewer = User(email=f"v-{run.id}@x.demo", name="阅读者", role="viewer", password_hash="x")
    db.add(viewer)
    db.commit()

    app.dependency_overrides[current_user] = _Client(analyst)
    client = TestClient(app)
    payload = plan_payload(run).model_dump()
    resp = client.post("/replant-plans", json=payload)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["run_id"] == run.id
    pid = body["id"]

    # 列表与详情可读
    assert client.get("/replant-plans").status_code == 200
    assert client.get(f"/replant-plans/{pid}").status_code == 200

    # viewer 写入被拒
    app.dependency_overrides[current_user] = _Client(viewer)
    resp = client.post("/replant-plans", json=payload)
    assert resp.status_code == 403

    # 未认证 401
    app.dependency_overrides.pop(current_user)
    resp = client.post("/replant-plans", json=payload)
    assert resp.status_code == 401
