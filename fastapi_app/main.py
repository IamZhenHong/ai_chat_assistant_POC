from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from . import models, schemas
from openai import OpenAI
import os
from dotenv import load_dotenv
from sqlalchemy import desc
import json
from typing import List
import os
from supabase import create_client, Client
import uuid
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import urllib.parse
import psycopg2
# Load environment variables
load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
if not url or not key:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment variables.")
supabase: Client = create_client(url, key)
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()
# ✅ Extract host and set port
host = f"db.{url.split('//')[1]}"
port = 5432  # Supabase default

# ✅ URL Encode the Password
password = os.environ.get("SUPABASE_PASSWORD")
encoded_password = urllib.parse.quote(password, safe="")

# ✅ Construct SQLAlchemy connection URL
DATABASE_URL = f"postgresql://postgres.rgwighbhilwimfpmrcak:{encoded_password}@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres"
engine = create_engine(DATABASE_URL, echo=True)

try:
    with engine.connect() as connection:
        print("✅ Successfully connected to Supabase PostgreSQL!")
except Exception as e:
    print(f"❌ SQLAlchemy Connection Error: {e}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ✅ Define the Base model class
Base = declarative_base()

Base.metadata.create_all(bind=engine)

app = FastAPI()

# ✅ Create Target (Supabase)
@app.post("/targets/", response_model=schemas.TargetOut)
def create_target(target: schemas.TargetCreate):
    target_data = target.dict()
    target_data["id"] = str(uuid.uuid4())  # Generate UUID

    response = supabase.table("targets").insert(target_data).execute()

    return response.data[0]

# ✅ Get All Targets (Supabase)
@app.get("/targets/", response_model=List[schemas.TargetOut])
def get_all_targets():
    response = supabase.table("targets").select("*").execute()

    return response.data

# ✅ Update Target (Supabase)
@app.put("/targets/{target_id}", response_model=schemas.TargetOut)
def update_target(target_id: uuid.UUID, updated_target: schemas.TargetCreate):
    response = (
    supabase.table("targets")
    .update(updated_target.dict())
    .eq("id", str(target_id))  # ✅ Convert to string
    .execute()
    )

    return response.data[0]

# ✅ Add Conversation Snippet (Supabase)
@app.post("/conversation_snippets/", response_model=schemas.ConversationSnippetOut)
def create_conversation_snippet(conversation_snippet: schemas.ConversationSnippetCreate):
    # ✅ Validate `target_id`
    if not conversation_snippet.target_id:
        raise HTTPException(status_code=400, detail="❌ Target ID is required.")

    # ✅ Insert conversation snippet with error handling
    try:
        convo_snippet = {
            "id": str(uuid.uuid4()),
            "content": conversation_snippet.convo,
            "target_id": str(conversation_snippet.target_id),
        }
        supabase.table("conversation_snippets").insert(convo_snippet).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Database error inserting conversation snippet: {str(e)}")

    return convo_snippet

@app.post("/persona/", response_model=schemas.PersonaOut)
def create_persona(persona: schemas.PersonaCreate):
    if not persona.name:
        raise HTTPException(status_code=400, detail="Name is required")

    if not persona.description:
        raise HTTPException(status_code=400, detail="Description is required")

    if not persona.gender:
        raise HTTPException(status_code=400, detail="Gender is required")
    
    response = supabase.table("personas").insert(persona.dict()).execute()

    return response.data[0]

@app.get("/personas/", response_model=List[schemas.PersonaOut])
def get_all_personas():
    response = supabase.table("personas").select("*").execute()

    return response.data


@app.post("/love_analysis/", response_model=schemas.LoveAnalysisOut)
def create_love_analysis(love_analysis: schemas.LoveAnalysisCreate):
    # ✅ Validate `target_id`
    if not love_analysis.target_id:
        raise HTTPException(status_code=400, detail="❌ Target ID is required.")

    # ✅ Insert conversation snippet with error handling
    try:
        convo_snippet = {
            "id": str(uuid.uuid4()),
            "content": love_analysis.convo,
            "target_id": str(love_analysis.target_id),
        }
        supabase.table("conversation_snippets").insert(convo_snippet).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Database error inserting conversation snippet: {str(e)}")

    # ✅ Fetch target and check if exists
    target_data = (
        supabase.table("targets")
        .select("*")
        .eq("id", str(love_analysis.target_id))
        .execute()
    ).data

    if not target_data:
        raise HTTPException(status_code=404, detail="❌ Target not found")

    target = target_data[0]  # ✅ Extract first result safely

    # ✅ Fetch previous analysis (handle missing data)
    prev_analysis_data = (
        supabase.table("love_analysis")
        .select("content")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    ).data
    prev_content = prev_analysis_data[0]["content"] if prev_analysis_data else "No previous analysis available."

    # ✅ System prompt for GPT-4o
    prompt = f"""
    You are a love coach who is very good at analyzing the relationship dynamics, personalities, and latent feelings of both parties. I'm your client seeking your advice. 
    ###
    Analyze the previous love analysis content and chat history provided, and output the following analysis:
    1. General relationship dynamic
    2. How I present myself in front of the other party
    3. How the other party most likely sees me and feels about me
    4. What the other party most likely needs from our interaction or relationship
    5. My personalities shown in the conversation
    6. The other party's personality shown in the conversation
    7. What the other party is likely to do next in our interactions
    8. Overall advice if I want to achieve my relationship goals
    9. How have the relationship dynamics changed since the last conversation
    ###
    Previous Love Analysis:
    {prev_content}

    Current Conversation:
    {love_analysis.convo}

    New Love Analysis:
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            store=True,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"""Output in {target.get('language', 'English')}: """},
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ GPT-4o API error: {str(e)}")

    # ✅ Ensure GPT response is valid
    if not completion.choices or not completion.choices[0].message.content:
        raise HTTPException(status_code=500, detail="❌ GPT-4o returned an empty response")

    analysis_content = completion.choices[0].message.content

    # ✅ Insert new analysis with error handling
    new_analysis = {
        "id": str(uuid.uuid4()),
        "convo": love_analysis.convo,
        "content": analysis_content,
        "target_id": str(love_analysis.target_id),
    }

    try:
        supabase.table("love_analysis").insert(new_analysis).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Database error inserting love analysis: {str(e)}")

    # ✅ Return response
    return schemas.LoveAnalysisOut(content=analysis_content)

# ✅ Create Chat Strategy (Supabase)
@app.post("/chat_strategies/", response_model=schemas.ChatStrategyOut)
def create_chat_strategy(chat_strategy: schemas.ChatStrategyCreate):
    target = (
        supabase.table("targets")
        .select("*")
        .eq("id", str(chat_strategy.target_id))
        .execute()
    ).data[0]

    latest_love_analysis = (
        supabase.table("love_analysis")
        .select("content")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    ).data

    latest_convo = (
        supabase.table("conversation_snippets")
        .select("content")
        .eq("target_id", str(chat_strategy.target_id))
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    ).data

    latest_chat_strategy = (
        supabase.table("chat_strategies")
        .select("content")
        .eq("target_id", str(chat_strategy.target_id))
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    ).data


    system_prompt = f"""
        You are a love coach who is very good at helping clients come up with the right strategy and exact reply in communication to reach their short-term and long-term relationship goals. I'm your client seeking your advice.

        Come up with a communication strategy that is brief, easy to follow, and actionable for me to talk to {target["name"]} based on the context below.
        Output in {target["language"]}:
        Context: 
        """
    system_prompt += f"""
        my gender: {target["gender"]}
        I'm talking to {target["name"]} online
        {target["name"]}'s gender: {target["gender"]}
        {target["name"]}'s personality: {target["personality"]}
        relationship context: {target["relationship_context"]}
        my feelings about our relationship: {target["relationship_perception"]}
        my short-term goal with {target["name"]}: {target["relationship_goals"]}
        my long-term goal with {target["name"]}: {target["relationship_goals_long"]}
        relationship dynamics:
        {latest_love_analysis}
        Last conversation snippet: {latest_convo}
        Last chat strategy: {latest_chat_strategy}
        """


    completion = client.chat.completions.create(
        model="gpt-4o",
        store=True,
        messages=[{"role": "system", "content": system_prompt}],
    )
    chat_strategy_content = completion.choices[0].message.content

    new_strategy = {
        "id": str(uuid.uuid4()),
        "convo": latest_convo[0]['content'] if latest_convo else 'None',
        "love_analysis": latest_love_analysis[0]['content'] if latest_love_analysis else 'None',
        "content": chat_strategy_content,
        "target_id": str(chat_strategy.target_id),
    }
    supabase.table("chat_strategies").insert(new_strategy).execute()

    return new_strategy

from fastapi import HTTPException

@app.post("/reply_options_flow/", response_model=schemas.ReplyOptionsOut)
def create_reply_options_flow(reply_options: schemas.ReplyOptionsCreate):
    # ✅ Validate `target_id`
    if not reply_options.target_id:
        raise HTTPException(status_code=400, detail="❌ target_id is required")
    
    if not reply_options.persona_id:
        raise HTTPException(status_code=400, detail="❌ persona_id is required")

    # ✅ Fetch target & check if exists
    target_data = (
        supabase.table("targets")
        .select("*")
        .eq("id", str(reply_options.target_id))
        .execute()
    ).data

    persona_data = (
        supabase.table("personas")
        .select("*")
        .eq("id", str(reply_options.persona_id))
        .execute()
    ).data

    if not persona_data:
        raise HTTPException(status_code=404, detail="❌ Persona not found")

    if not target_data:
        raise HTTPException(status_code=404, detail="❌ Target not found")

    target = target_data[0]  # ✅ Extract first result

    # ✅ Fetch latest conversation snippet (handle missing data)
    latest_convo_data = (
    supabase.table("conversation_snippets")
    .select("content")
    .eq("target_id", reply_options.target_id)  # Only get rows where target_id matches
    .order("created_at", desc=True)
    .limit(1)
    .execute()
).data
    
    latest_convo = latest_convo_data[0]['content'] if latest_convo_data else "No recent conversation."

    # ✅ Fetch latest chat strategy (handle missing data)
    latest_chat_strategy_data = (
        supabase.table("chat_strategies")
        .select("content")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    ).data
    latest_chat_strategy = latest_chat_strategy_data[0]['content'] if latest_chat_strategy_data else "No chat strategy available."

    # ✅ System prompt with error handling
    system_prompt = f"""
    Generate 4 response options based on:
    - Last conversation: {latest_convo}
    - Chat strategy: {latest_chat_strategy}
    - Context: {target.get('name', 'Unknown')}, {target.get('personality', 'Unknown')}
    - Persona: {persona_data[0].get('name', 'Unknown')}, {persona_data[0].get('description', 'Unknown')}
    """

    print(system_prompt)

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            store=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content":f"""Output in {target.get('language', 'English')}: """},
                ],
            response_format = schemas.ReplyOptionsOut
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ GPT-4o API error: {str(e)}")

    # ✅ Ensure GPT response is valid
    if not completion.choices or not completion.choices[0].message.content:
        raise HTTPException(status_code=500, detail="❌ GPT-4o returned an empty response")

    options = completion.choices[0].message.parsed
    print(options)
    print("option1",options.option1)
    print("option2",options.option2)
    print("option3",options.option3)
    # ✅ Ensure at least 4 options are available

    # ✅ Create new reply options entry
    new_reply_options_flow = {
        "id": str(uuid.uuid4()),
        "convo": latest_convo,
        "chat_strategy": latest_chat_strategy,
        "option1": options.option1,
        "option2": options.option2,
        "option3": options.option3,
        "option4": options.option4,
        "target_id": str(reply_options.target_id),
    }

    # ✅ Insert into Supabase with error handling
    try:
        supabase.table("reply_options_flows").insert(new_reply_options_flow).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Database error: {str(e)}")

    # ✅ Return response
    return schemas.ReplyOptionsOut(
        option1=options.option1,
        option2=options.option2,
        option3=options.option3,
        option4=options.option4,
    )
