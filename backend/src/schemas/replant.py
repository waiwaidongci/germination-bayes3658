from pydantic import BaseModel,Field,ConfigDict

class ReplantLotInput(BaseModel):
    model_config=ConfigDict(extra="forbid")
    lot_code:str=Field(min_length=1,max_length=40)
    available_seeds:int=Field(ge=0,le=200,strict=True)
    cost_per_seed:float=Field(gt=0,allow_inf_nan=False,strict=True)

class ReplantPlanInput(BaseModel):
    model_config=ConfigDict(extra="forbid")
    name:str=Field(min_length=2,max_length=160)
    run_id:int=Field(gt=0,strict=True)
    target:int=Field(ge=0,le=5600,strict=True)
    required_probability:float=Field(gt=0,lt=1,allow_inf_nan=False,strict=True)
    time_limit:float=Field(default=5.0,gt=0,le=60.0,allow_inf_nan=False,strict=True)
    lots:list[ReplantLotInput]=Field(min_length=1,max_length=28)
