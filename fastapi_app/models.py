import uuid
from sqlalchemy import Column, String, ForeignKey, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from .database import Base

class Target(Base):
    __tablename__ = "targets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    gender = Column(String, nullable=True)
    relationship_context = Column(String, nullable=True)
    relationship_perception = Column(String, nullable=True)
    relationship_goals = Column(String, nullable=True)
    relationship_goals_long = Column(String, nullable=True)
    personality = Column(String, nullable=True)
    language = Column(String, nullable=True)

    # Relationships
    conversation_snippets = relationship("ConversationSnippet", back_populates="target")
    love_analyses = relationship("LoveAnalysis", back_populates="target")
    styles = relationship("Style", back_populates="target")
    chat_strategies = relationship("ChatStrategy", back_populates="target")
    reply_options_flows = relationship("ReplyOptionsFlow", back_populates="target")


class ConversationSnippet(Base):
    __tablename__ = "conversation_snippets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())

    # Foreign key and relationship
    target_id = Column(UUID(as_uuid=True), ForeignKey("targets.id"), nullable=False)
    target = relationship("Target", back_populates="conversation_snippets")


class LoveAnalysis(Base):
    __tablename__ = "love_analysis"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    convo = Column(String, nullable=False)
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())

    # Foreign key and relationship
    target_id = Column(UUID(as_uuid=True), ForeignKey("targets.id"), nullable=False)
    target = relationship("Target", back_populates="love_analyses")


class Style(Base):
    __tablename__ = "styles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    convo = Column(String, nullable=False)
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())

    # Foreign key and relationship
    target_id = Column(UUID(as_uuid=True), ForeignKey("targets.id"), nullable=False)
    target = relationship("Target", back_populates="styles")


class ChatStrategy(Base):
    __tablename__ = "chat_strategies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    convo = Column(String, nullable=False)
    love_analysis = Column(String, nullable=False)
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())

    # Foreign key and relationship
    target_id = Column(UUID(as_uuid=True), ForeignKey("targets.id"), nullable=False)
    target = relationship("Target", back_populates="chat_strategies")


class ReplyOptionsFlow(Base):
    __tablename__ = "reply_options_flows"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_strategy = Column(String, nullable=False)
    convo = Column(String, nullable=False)
    option1 = Column(String, nullable=False)
    option2 = Column(String, nullable=False)
    option3 = Column(String, nullable=False)
    option4 = Column(String, nullable=False)

    # Foreign key and relationship
    target_id = Column(UUID(as_uuid=True), ForeignKey("targets.id"), nullable=False)
    target = relationship("Target", back_populates="reply_options_flows")

class Persona(Base):
    __tablename__ = "personas"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    gender = Column(String, nullable=False)
    description = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
                        