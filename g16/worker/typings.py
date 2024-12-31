import sys
from pathlib import Path
from pydantic import BaseModel,Field
from datetime import datetime
from typing import Literal,Union,List,Dict,Optional


TaskState = Literal["queuing","staging","canceled","running","finished","failed","killed","holding"]
AppId = Literal['Funmg','QC_Calculator','OLED_spectrum','hongshuai_unimol_sky','normal']
InputType = Literal['smi','sdf','xyz']
JobType = Literal['s0','s1','t1','absorption_spec','emission_spec']


class MolData(BaseModel):
    name:str
    type:InputType
    content:str

class TaskInCfg(BaseModel):
    file:MolData
    cfg:Dict

class TaskIn(BaseModel):
    name: str
    cfg: TaskInCfg
    app_id: AppId = 'QC_Calculator'

class TaskOut(TaskIn):
    owner: str
    id: str = ''
    state: TaskState = "queuing"
    progress: Union[float, None] = None
    is_favorite: bool = False
    ctime: float = Field(default_factory=lambda :datetime.now().timestamp())
    exec_time: Union[float, None] = None
    end_time: Union[float, None] = None
    worker_id: Union[str, None] = None


class WorkerInfo(BaseModel):
    worker_id:str
    app_ids: List[AppId]=['normal']
    token:str

## 任务更新
class TaskUpdateData(BaseModel):
    name: Union[str, None] = None
    progress: Union[float, None] = None
    exec_time: Union[float, None] = None
    end_time: Union[float, None] = None
    state: Union[TaskState, None] = None
    remove_flag: Union[bool, None] = None
    is_favorite: Union[bool, None] = None

class TaskUpdate(BaseModel):
    id:str
    data:TaskUpdateData

class WorkerUpdateTaskData(WorkerInfo):
    data:TaskUpdate

