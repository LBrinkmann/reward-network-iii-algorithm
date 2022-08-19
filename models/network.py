from typing import Optional,List,Dict,Any
from pydantic import BaseModel,validator,parse_obj_as,ValidationError,root_validator
from collections import Counter

class node(BaseModel):
    
    node_num: int
    display_name: str
    node_size: int
    level: int
    x: float
    y: float

    @validator('node_num')
    def max_ten_nodes(cls, n):
        if n<0 or n>9:
            raise ValueError('must be a number between 0 and 9')
        return n
    @validator('level')
    def max_four_levels(cls, n):
        if n<0 or n>3:
            raise ValueError('must be a number between 0 and 3')
        return n

    class Config:
        schema_extra = {
            "example": [
                {
                    'node_num':0,
                    'display_name': 'A',
                    'node_size':3,
                    'level':0,
                    'x':-10.394,
                    'y':3.2020
                }
            ]
        }

class edge(BaseModel):
    source_id: int
    target_id: int
    reward: int

    @validator('source_id')
    def check_source(cls, n):
        if n<0 or n>9:
            raise ValueError('must be a number between 0 and 9')
        return n
    @validator('target_id')
    def check_target(cls, n):
        if n<0 or n>9:
            raise ValueError('must be a number between 0 and 9')
        return n
    @validator('reward')
    def check_reward(cls, n):
        possible_rewards = [-100,-20,0,20,140]
        if n not in possible_rewards:
            raise ValueError(f'reward must be a value in {possible_rewards}')
        return n
    @root_validator()
    def validate_no_self_connection(cls, values: Dict[str, Any]) -> Dict[str, Any]:
         print(Counter(values.get("source_id")))
         print(Counter(values.get("target_id")))
         if values.get("source_id") == values.get("target_id"):
              raise ValueError("source_id must be different from target_id")
         return values

    class Config:
        schema_extra = {
            "example": [
                {
                    'source_id': 0,
                    'target_id': 2,
                    'reward':20
                }
            ]
        }


class network(BaseModel):
    network_id: str
    nodes: List[node]
    edges: List[edge]
    starting_node: int
    max_reward: int