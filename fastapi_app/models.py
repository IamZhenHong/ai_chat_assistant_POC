from sqlalchemy import Column, ForeignKey, String, Text, Integer, TIMESTAMP, JSON, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    phone_number = Column(String, unique=True, nullable=True)
    email = Column(String, unique=True, nullable=True)
    name = Column(String, nullable=False)
    language = Column(String)
    gender = Column(String)
    age = Column(String)
    about_me = Column(Text)
    created_at = Column(TIMESTAMP)

class Recipient(Base):
    __tablename__ = 'recipients'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    name = Column(String, nullable=False)
    gender = Column(String)
    age = Column(String)
    language = Column(String)
    about_me = Column(Text)
    created_at = Column(TIMESTAMP)

class Relationship(Base):
    __tablename__ = 'relationships'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    recipient_id = Column(UUID(as_uuid=True), ForeignKey('recipients.id'), nullable=False)
    relationship_stage_overview = Column(String)
    relationship_goal = Column(String)
    user_personality_overview = Column(String)
    user_communication_style_overview = Column(String)
    recipient_personality_overview = Column(String)
    recipient_communication_style_overview = Column(String)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    relationship_id = Column(UUID(as_uuid=True), ForeignKey('relationships.id'), nullable=False)
    topic = Column(String)
    conversation_history = Column(Text)
    conversation_summary = Column(Text)
    last_updated = Column(TIMESTAMP)

class ConversationSnippet(Base):
    __tablename__ = 'conversation_snippets'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.id'), nullable=False)
    sequence_id = Column(Integer, nullable=False)
    content = Column(Text)
    image_url = Column(String, nullable=True)
    uploaded_at = Column(TIMESTAMP)

class ConversationAnalysis(Base):
    __tablename__ = 'conversation_analyses'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.id'), nullable=False)
    user_communication_style = Column(String)
    user_personality = Column(String)
    recipient_communication_style = Column(String)
    recipient_personality = Column(String)
    relationship_stage = Column(String)
    relationship_trend = Column(String)
    generated_at = Column(TIMESTAMP)

class ReplySuggestion(Base):
    __tablename__ = 'reply_suggestions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.id'), nullable=False)
    persona_id = Column(UUID(as_uuid=True), ForeignKey('personas.id'), nullable=True)
    reply_1 = Column(Text)
    reply_2 = Column(Text)
    reply_3 = Column(Text)
    reply_4 = Column(Text)
    created_at = Column(TIMESTAMP)

class Persona(Base):
    __tablename__ = 'personas'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text)
    language = Column(String)
    texting_style = Column(JSON)
    created_at = Column(TIMESTAMP)

# Unique constraint to ensure correct order of snippets per conversation
UniqueConstraint('conversation_id', 'sequence_id', name='uq_conversation_snippet_order')
