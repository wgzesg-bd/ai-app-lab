from enum import Enum

from pydantic import BaseModel


class MemoryUpdateSetting(str, Enum):
    NO_AUTO_UPDATE = "NO_AUTO_UPDATE"
    BLOCKING = "BLOCKING"
    NON_BLOCKING = "NON_BLOCKING"


class RunnerConfig(BaseModel):
    memory_update_behavior: MemoryUpdateSetting = MemoryUpdateSetting.BLOCKING
