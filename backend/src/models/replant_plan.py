from sqlalchemy import JSON,ForeignKey
from sqlalchemy.orm import Mapped,mapped_column
from ..config.database import Base
from ..utils.clock import now
class ReplantPlan(Base):
    __tablename__="replant_plans"
    id:Mapped[int]=mapped_column(primary_key=True)
    name:Mapped[str]
    run_id:Mapped[int]=mapped_column(ForeignKey("bayes_runs.id"))
    target:Mapped[int]
    required_probability:Mapped[float]
    time_limit:Mapped[float]
    status:Mapped[str]
    frozen_snapshot:Mapped[dict]=mapped_column(JSON)
    result:Mapped[dict]=mapped_column(JSON,default=dict)
    owner:Mapped[str]
    created_at:Mapped[str]=mapped_column(default=now)
