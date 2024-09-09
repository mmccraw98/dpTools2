from pydantic import BaseModel
from typing import Union, Tuple, Any

class Result(BaseModel):
    data: Tuple[Any, ...]
    time: Union[float, int]
