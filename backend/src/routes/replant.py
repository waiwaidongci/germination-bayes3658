from fastapi import APIRouter,Depends
from sqlalchemy.orm import Session
from ..config.database import get_db
from ..controllers.dependencies import current_user,roles
from ..constants.roles import Role
from ..schemas.replant import ReplantPlanInput
from ..services.replant_plan_service import list_plans,get_plan,create_plan
from ..utils.serialize import record
router=APIRouter(prefix='/replant-plans',tags=['replant-plans'])
@router.get('')
def index(db:Session=Depends(get_db),user=Depends(current_user)):return list_plans(db)
@router.get('/{pid}')
def show(pid:int,db:Session=Depends(get_db),user=Depends(current_user)):return record(get_plan(db,pid))
@router.post('',status_code=201)
def create(data:ReplantPlanInput,db:Session=Depends(get_db),user=Depends(roles(Role.ADMIN,Role.ANALYST))):
    return create_plan(db,data,user)
