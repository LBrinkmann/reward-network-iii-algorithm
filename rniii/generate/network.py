import pydantic

pydantic.class_validators._FUNCS.clear()
from typing import Dict, Any, Optional
from pydantic import BaseModel, validator, root_validator


class Node(BaseModel):
    node_num: int
    display_name: str
    node_size: int
    starting_node: Optional[bool] = False
    level: int
    x: float
    y: float

    @validator("node_num")
    def max_ten_nodes(cls, n):
        if n < 0 or n > 9:
            raise ValueError("node number must be a number between 0 and 9")
        return n

    @validator("level")
    def max_four_levels(cls, n):
        if n < 0 or n > 4:
            raise ValueError("level must be a number between 0 and 3")
        return n

    class Config:
        schema_extra = {
            "example": [
                {
                    "node_num": 0,
                    "display_name": "A",
                    "node_size": 3,
                    "level": 0,
                    "x": -10.394,
                    "y": 3.2020,
                }
            ]
        }


class Edge(BaseModel):
    source_num: int
    target_num: int
    reward: int
    arc_type: str
    source_x: float
    source_y: float
    arc_x: float
    arc_y: float
    target_x: float
    target_y: float

    @validator("source_num")
    def check_source(cls, n):
        if n < 0 or n > 9:
            raise ValueError("source node number must be a number between 0 and 9")
        return n

    @validator("target_num")
    def check_target(cls, n):
        if n < 0 or n > 9:
            raise ValueError("target node number must be a number between 0 and 9")
        return n

    # @validator('reward')
    # def check_reward(cls, n):
    #    possible_rewards = [-100, -20, 0, 20, 140]
    #    if n not in possible_rewards:
    #        raise ValueError(f'reward must be a value in {possible_rewards}')
    #    return n

    @validator("arc_type")
    def check_arc_type(cls, n):
        if n not in ["straight", "curved"]:
            raise ValueError(
                "arc type must be chosen among possible options between straight or curved"
            )
        return n

    @root_validator()
    def validate_no_self_connection(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("source_num") == values.get("target_num"):
            raise ValueError("source_num must be different from target_num")
        return values

    class Config:
        schema_extra = {
            "example": [
                {
                    "source_num": 0,
                    "target_num": 2,
                    "reward": 20,
                    "arc_type": "straight",
                    "source_x": 89.84720834968113,
                    "source_y": 7.376435889674515,
                    "arc_x": 34.54971253634034,
                    "arc_y": 47.55241560766308,
                    "target_x": -20.02973382010107,
                    "target_y": 87.20670189283723,
                }
            ]
        }


class Network(BaseModel):
    network_id: str
    nodes: list[Node]
    edges: list[Edge]
    starting_node: int
    max_reward: Optional[int]
