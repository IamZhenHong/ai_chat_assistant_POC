import uuid
from pydantic import BaseModel
from typing import Optional

# ✅ Fix target_id: int -> target_id: uuid.UUID
class LoveAnalysisCreate(BaseModel):
    convo: str
    target_id: uuid.UUID  # Changed from int to UUID

class LoveAnalysisOut(BaseModel):
    content: str

    class Config:
        from_attributes = True

class PersonaCreate(BaseModel):
    name : str
    description : str
    gender : str


class PersonaOut(BaseModel):
    id : uuid.UUID
    name : str
    description : str
    gender : str


class StyleCreate(BaseModel):
    convo: str
    target_id: uuid.UUID  # Changed from int to UUID

class Style(StyleCreate):
    content: str


class ChatStrategyCreate(BaseModel):
    target_id: uuid.UUID  # Changed from int to UUID

class ChatStrategyOut(BaseModel):
    content: str


class ReplyOptionsCreate(BaseModel):
    target_id: uuid.UUID  # Changed from int to UUID
    persona_id: uuid.UUID  # Changed from int to UUID
    

class ReplyOptionsOut(BaseModel):
    option1: str
    option2: str
    option3: str
    option4: str


class TargetBase(BaseModel):
    name: str
    gender: Optional[str] = None
    relationship_context: Optional[str] = None
    relationship_perception: Optional[str] = None
    relationship_goals: Optional[str] = None
    relationship_goals_long: Optional[str] = None
    personality: Optional[str] = None
    language: Optional[str] = None


class TargetCreate(TargetBase):
    pass


class TargetOut(TargetBase):
    id: uuid.UUID  # Changed from int to UUID

    class Config:
        from_attributes = True
