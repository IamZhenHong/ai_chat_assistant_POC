import uuid
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# ✅ Fix Recipient_id: int -> Recipient_id: uuid.UUID
class UserCreate(BaseModel):
    name: str
    gender: Optional[str] = None
    age: Optional[int] = None  # Assuming age should be an integer
    language: Optional[str] = None
    about_me: Optional[str] = None

class UserOut(BaseModel):
    id: uuid.UUID  # Ensuring consistency with UUID
    name: str
    gender: Optional[str] = None
    age: Optional[int] = None
    language: Optional[str] = None
    about_me: Optional[str] = None
    created_at: datetime  # Added for better response tracking
    updated_at: Optional[datetime] = None  # Useful for tracking modifications

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    name: Optional[str] = None  # Allow partial updates
    gender: Optional[str] = None
    age: Optional[int] = None
    language: Optional[str] = None
    about_me: Optional[str] = None

class UserUpdateOut(UserOut): 
    name: str

from pydantic import BaseModel
from typing import Optional
import uuid

class RecipientBase(BaseModel):
    """Base model for recipient data."""
    name: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[str] = None
    about_me: Optional[str] = None
    language: Optional[str] = None

class RecipientCreate(RecipientBase):
    """Model for creating a new recipient."""
    user_id: uuid.UUID  # Changed from int to UUID

class RecipientUpdate(RecipientBase):
    """Model for updating recipient data."""
    pass

class RecipientUpdateOut(BaseModel):
    """Model for response after updating a recipient."""
    pass

class RecipientOut(RecipientBase):
    """Model for retrieving recipient details."""
    relationship_id: Optional[uuid.UUID] = None  # Changed from int to UUID
    id: Optional[uuid.UUID] = None  # Changed from int to UUID

    class Config:
        from_attributes = True

class ConversationSnippetCreate(BaseModel):
    conversation_id: Optional[str] = None  # If None, create a new conversation
    relationship_id: str
    sequence_id: Optional[int] = None
    content: str
    image_url: Optional[str] = None
    uploaded_at: Optional[datetime] = None

class ConversationSnippetOut(BaseModel):
    conversation_id: str
    sequence_id: Optional[int] = None
    image_url: Optional[str] = None
    uploaded_at: Optional[datetime] = None

class ConversationCreate(BaseModel):
    relationship_id: str
    topic: Optional[str] = "New Conversation"
    conversation_history: Optional[str] = ""
    conversation_summary: Optional[str] = ""
    content: str
    last_updated: Optional[datetime] = datetime.utcnow()

class ConversationOut(BaseModel):
    id: str
    relationship_id: str
    topic: str
    conversation_history: str
    conversation_summary: str
    last_updated: datetime

class PersonaCreate(BaseModel):
    name : str
    description : str
    gender : str

class PersonaOut(BaseModel):
    id : uuid.UUID
    name : str
    description : str
    gender : str

class ReplySuggestionCreate(BaseModel):
    option : int
    relationship_id: uuid.UUID  # ✅ Ensures relationship exists
    conversation_id: uuid.UUID  # ✅ Ensures conversation exists
    persona_id: Optional[uuid.UUID] = None  # ✅ Ensures persona exists

class ReplySuggestionOut(BaseModel):

    reply_1: str
    reply_2: str
    reply_3: str
    reply_4: str


class ConversationAnalysisCreate(BaseModel):
    conversation_id: str
    relationship_id: str

class ConversationAnalysisOut(BaseModel):
    user_communication_style: str
    user_personality: str
    recipient_communication_style: str
    recipient_personality: str
    relationship_stage: str
    relationship_trend: str


class ConversationSummaryOut(BaseModel):
    conversation_id: str
    conversation_summary: str
    topic: str

class RelationshipOverviewOut(BaseModel):
    relationship_stage_overview: str
    relationship_trend_overview: str
    user_personality_overview: str
    user_communication_style_overview: str
    recipient_personality_overview: str
    recipient_communication_style_overview: str


class UserTextingStyleOut(BaseModel):
    user_texting_style: str