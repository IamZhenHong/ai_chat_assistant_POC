from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, desc
from sqlalchemy.ext.declarative import declarative_base
from typing import List
import urllib.parse
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
from celery import Celery
from celery.result import AsyncResult
from fastapi.middleware.cors import CORSMiddleware
from . import models, schemas
from .worker import summarize_conversation, update_relationship_overview, update_user_texting_style
import requests
from openai import AzureOpenAI
import json
import uvicorn
import time
import re
import json
from fastapi_app.worker import celery_app  
# Load environment variables
load_dotenv()

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
  api_key= os.getenv("AZURE_OPENAI_API_KEY"),
  api_version="2024-08-01-preview"
)

MEM0_URL = "http://192.168.11.4:18669"
MEM0_APP_ID = 800

# Database Configuration
encoded_password = urllib.parse.quote(SUPABASE_PASSWORD, safe="")
DATABASE_URL = f"postgresql://postgres.rgwighbhilwimfpmrcak:{encoded_password}@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres"
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Test Database Connection
try:
    with engine.connect() as connection:
        print("✅ Successfully connected to Supabase PostgreSQL!")
except Exception as e:
    print(f"❌ SQLAlchemy Connection Error: {e}")

# Define Base Model
Base = declarative_base()
Base.metadata.create_all(bind=engine)

REDIS_URL = os.getenv("REDIS_URL")

# Celery Configuration

# FastAPI App Setup
app = FastAPI()

# CORS Configuration
origins = [
    "https://b886-60-50-200-107.ngrok-free.app",  # ✅ Deployed Streamlit app
    "https://91d4-128-106-187-4.ngrok-free.app ",  # ✅ Ngrok FastAPI URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/users/", response_model=schemas.UserOut)
def create_user(user: schemas.UserCreate):
    user_data = user.dict()
    user_data["id"] = str(uuid.uuid4())  # Generate UUID

    response = supabase.table("users").insert(user_data).execute()

    return response.data[0]

@app.get("/users/", response_model=List[schemas.UserOut])
def get_all_users():
    response = supabase.table("users").select("*").execute()

    return response.data

@app.put("/users/{user_id}", response_model=schemas.UserUpdateOut)
def update_user(user_id: uuid.UUID, updated_user: schemas.UserUpdate):
    user_id_str = str(user_id)

    response = (
        supabase.table("users")
        .update(updated_user.dict())
        .eq("id", user_id_str)
        .execute()
    )

    if not response.data:
        raise HTTPException(status_code=404, detail="User not found or no changes made.")

    return response.data[0]

# ✅ Create Target (Supabase)
@app.post("/recipients/", response_model=schemas.RecipientOut)
def create_recipient(target: schemas.RecipientCreate):
    recipient_data = target.dict()
    recipient_data["id"] = str(uuid.uuid4())  # Generate UUID
    recipient_data["user_id"] = str(target.user_id)  # Convert to string
    # recipient_data["relationship_id"] = str(uuid.uuid4())  # Generate Relationship UUID

    # ✅ Insert recipient first
    response = supabase.table("recipients").insert(recipient_data).execute()
    if not response.data:
        raise HTTPException(status_code=500, detail="❌ Failed to create recipient.")
    new_recipient = response.data[0]
    relationship_id = str(uuid.uuid4())
    # ✅ Create a new relationship object
    relationship_data = {
        "id": relationship_id,
        "user_id": str(target.user_id),
        "recipient_id": new_recipient["id"],
        "relationship_stage_overview": "New Connection",
        "relationship_goal": "",
        "user_personality_overview": "",
        "user_communication_style_overview": "",
        "recipient_personality_overview": "",
        "recipient_communication_style_overview": "",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "language": new_recipient.get("language"),
    }

    # ✅ Insert relationship
    relationship_response = supabase.table("relationships").insert(relationship_data).execute()
    if not relationship_response.data:
        raise HTTPException(status_code=500, detail="❌ Failed to create relationship.")
    
    update_recipient_data = {
        "relationship_id": relationship_id
    }
    # ✅ Update recipient with relationship_id
    response = (
        supabase.table("recipients")
        .update(update_recipient_data)
        .eq("id", new_recipient["id"])
        .execute()
    )
    
    return {
        "id": new_recipient["id"],
        "user_id": new_recipient["user_id"],
        "name": new_recipient["name"],
        "gender": new_recipient.get("gender"),
        "language": new_recipient.get("language"),
        "age": new_recipient.get("age"),
        "about_me": new_recipient.get("about_me"),
        "relationship_id": relationship_data["id"],
    }


# ✅ Get All Recipients for a Specific User
@app.get("/recipients/{user_id}", response_model=List[schemas.RecipientOut])
def get_all_recipients(user_id: uuid.UUID):
    response = (
        supabase.table("recipients")
        .select("*")
        .eq("user_id", str(user_id))  # Ensure it's properly converted to a string
        .execute()
    )

    return response.data if response.data else []

# ✅ Update Target (Supabase)
@app.put("/recipients/{recipient_id}", response_model=schemas.RecipientUpdateOut)
def update_recipient(recipient_id: uuid.UUID, updated_recipient: schemas.RecipientUpdate):
    # Convert recipient_id to string for Supabase
    recipient_id_str = str(recipient_id)

    response = (
        supabase.table("recipients")
        .update(updated_recipient.dict())
        .eq("id", recipient_id_str)
        .execute()
    )

    if not response.data:
        raise HTTPException(status_code=404, detail="Recipient not found or no changes made.")

    return response.data[0]  # Ensure returning a single object, not a list


# ✅ Add Conversation Snippet (Supabase)
@app.post("/conversation_snippets/", response_model=schemas.ConversationSnippetOut)
def create_conversation_snippets(conversation_snippets: List[schemas.ConversationSnippetCreate]):
    if not conversation_snippets:
        raise HTTPException(status_code=400, detail="❌ No snippets provided.")
    
    conversation_id = str(uuid.uuid4())
    
    conversation_history = "\n".join(snippet.content for snippet in conversation_snippets)

    prompt = f"""
    You are an expert at structuring chat histories into JSON format.

    Given the following chat history:

    {conversation_history}

    Extract the structured format as a JSON array where each message follows this format:
    [
        {{"name": "sender_name", "content": "message_content"}},
        {{"name": "recipient_name", "content": "message_content"}}
    ]

    Ensure:
    - The output is valid JSON.
    - No extra explanations, only return raw JSON.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant that formats chat logs into structured data."},
                  {"role": "user", "content": prompt}],
        temperature=0
    )

    response_text = response.choices[0].message.content.strip()

    # Remove Markdown-style triple backticks
    response_text = re.sub(r"```json\n(.*?)\n```", r"\1", response_text, flags=re.DOTALL)

    # Ensure valid JSON formatting
    try:
        formatted_conversation_history = json.loads(response_text)
    except json.JSONDecodeError:
        print("Error: GPT response was not valid JSON. Raw response:")
        print(response_text)
        return []

  
    new_conversation = {
        "id": conversation_id,
        "relationship_id": str(conversation_snippets[0].relationship_id),
        "topic": "New Conversation",
        "conversation_history": [formatted_conversation_history],
        "conversation_summary": "",
        "last_updated": datetime.utcnow().isoformat(),
    }
    
    supabase.table("conversations").insert(new_conversation).execute()
    
    summarize_conversation.apply_async(args=[conversation_history, conversation_id], countdown=10)
    update_user_texting_style.apply_async(args=[conversation_history, conversation_snippets[0].relationship_id], countdown=10)
    
    snippets_data = [
        {
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "sequence_id": snippet.sequence_id,
            "content": snippet.content,
            "image_url": snippet.image_url,
            "uploaded_at": datetime.utcnow().isoformat(),
        }
        for snippet in conversation_snippets
    ]





    recipient_name = supabase.table("recipients").select("name").eq("id", conversation_snippets[0].recipient_id).execute().data[0].get("name", "Unknown")

    user_name = supabase.table("users").select("name").eq("id", conversation_snippets[0].user_id).execute().data[0].get("name", "Unknown")


    # Define the request payload
    payload = {
        "user_id": user_name,
        "app_id": str(MEM0_APP_ID),
        "memory": formatted_conversation_history,
        "character_name": recipient_name
    }

    print(payload)
    # Define headers
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    # Make the POST request
    response = requests.post(f"{MEM0_URL}/api/add_memory", json=payload, headers=headers)

    # Print the response
    print(response.status_code)
    print(response.json())  # Assuming the response is in JSON format

    if(response.status_code != 200):
        raise HTTPException(status_code=500, detail="❌ Failed to upload conversation history to Mem0.")
    
    
    response = supabase.table("conversation_snippets").insert(snippets_data).execute()
    if not response.data:
        raise HTTPException(status_code=500, detail="❌ Failed to insert conversation snippets.")
    return {"conversation_id": conversation_id}

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

@app.post("/conversation_analysis/", response_model=schemas.ConversationAnalysisOut)
def create_conversation_analysis(conversation_analysis: schemas.ConversationAnalysisCreate):
    if not conversation_analysis.conversation_id:
        raise HTTPException(status_code=400, detail="Conversation ID is required")
    
    # Fetch relationship data
    relationship_data = supabase.table("relationships").select("*").eq("id", str(conversation_analysis.relationship_id)).execute().data
    if not relationship_data:
        raise HTTPException(status_code=404, detail="Relationship not found")
    
    relationship = relationship_data[0]
    
    # Fetch conversation history
    conversation_history = supabase.table("conversations").select("conversation_history").eq("id", str(conversation_analysis.conversation_id)).execute().data
    if not conversation_history:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation_text = conversation_history[0].get("conversation_history", "Unknown")

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You analyze conversation history and generate only relevant database queries in natural language. "
                    "Do not include explanations, just the queries."
                ),
            },
            {
                "role": "user",
                "content": f"""
                Based on the following conversation, generate one natural language query
                to retrieve useful context from the database. Answer in {relationship.get('language', 'English')}.
                
                Conversation history:
                {conversation_text}


                """
            }
        ],
        response_format=schemas.GptMemoryQueryOut
    )


    query = response.choices[0].message.parsed

    print("Wuries",query)


    user_id = supabase.table("users").select("name").eq("id", relationship.get("user_id")).execute().data[0].get("name", "Unknown")
    recipient_id = supabase.table("recipients").select("name").eq("id", relationship.get("recipient_id")).execute().data[0].get("name", "Unknown")
    print(user_id)
    print(recipient_id)
        # user_name = supabase.table("users").select("name").eq("id", conversation_snippets[0].user_id).execute().data[0].get("name", "Unknown")


    payload = {
        "user_id": user_id,
        "app_id": str(MEM0_APP_ID),
        "query": query.query,
        "character_name": recipient_id
    }

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    response = requests.post(f"{MEM0_URL}/api/search_memory", json=payload, headers=headers)

    print(response.json()['data']['memory'])
    retrieved_memory = response.json()['data']['memory']
    compiled_memory = " ".join(retrieved_memory)  # Join list elements into a single string

    print("Compiled retrieved memory",compiled_memory)  # Output the final result

    recipient_personality = relationship.get("recipient_personality_overview", "Unknown")
    recipient_communication_style = relationship.get("recipient_communication_style_overview", "Unknown")
    
    # Fetch recipient language
    recipient_language_data = supabase.table("recipients").select("language").eq("id", relationship.get("recipient_id")).execute().data
    recipient_language = recipient_language_data[0].get("language", "English") if recipient_language_data else "English"
    
    system_prompt = f"""
    Analyze the conversation based on:
    - Relationship Goal: {relationship.get('relationship_goal', 'Unknown')}
    - Recipient Personality: {recipient_personality}
    - Recipient Communication Style: {recipient_communication_style}
    - Conversation History: {conversation_text}
    
    Additional Context (If relevant):
    Memory related to conversation ; {compiled_memory}
    Be comprehensive and detailed.
    """
    
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            store=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Output in {recipient_language}: "},
            ],
            response_format=schemas.ConversationAnalysisOut
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ GPT-4o API error: {str(e)}")
    
    analysis_results = completion.choices[0].message.parsed
    
    new_conversation_analysis = {
        "id": str(uuid.uuid4()),
        "conversation_id": str(conversation_analysis.conversation_id),
        "user_communication_style": analysis_results.user_communication_style,
        "user_personality": analysis_results.user_personality,
        "recipient_communication_style": analysis_results.recipient_communication_style,
        "recipient_personality": analysis_results.recipient_personality,
        "relationship_stage": analysis_results.relationship_stage,
        "relationship_trend": analysis_results.relationship_trend,
        "generated_at": datetime.utcnow().isoformat(),
    }
    
    conversation_analysis_update = {
        "relationship_id": str(conversation_analysis.relationship_id),
        "relationship_stage": analysis_results.relationship_stage,
        "relationship_trend": analysis_results.relationship_trend,
        "user_personality": analysis_results.user_personality,
        "user_communication_style": analysis_results.user_communication_style,
        "recipient_personality": analysis_results.recipient_personality,
        "recipient_communication_style": analysis_results.recipient_communication_style,
        "updated_at": datetime.utcnow().isoformat(),
    }
    
    update_relationship_overview.apply_async(args=[conversation_analysis_update], countdown=10)
    supabase.table("conversation_analyses").insert(new_conversation_analysis).execute()
    
    return analysis_results

import time
from datetime import datetime

@app.post("/reply_suggestions/", response_model=schemas.ReplySuggestionOut)
def create_reply_suggestion(reply_suggestion: schemas.ReplySuggestionCreate):
    """Generates AI-powered reply suggestions based on conversation context and persona."""
    
    start_time = time.time()  # Track total execution time
    step_times = {}

    print(f"🔹 Received request: {reply_suggestion.dict()}")
    
    step_times["request_received"] = time.time() - start_time  # Track time taken

    # ✅ Validate required fields
    if not reply_suggestion.conversation_id:
        raise HTTPException(status_code=400, detail="❌ conversation_id is required")

    # ✅ Fetch relationship data
    print(f"🔹 Fetching relationship data for ID: {reply_suggestion.relationship_id}")
    relationship_start = time.time()  # Track step time
    relationship_data = (
        supabase.table("relationships")
        .select("*")
        .eq("id", str(reply_suggestion.relationship_id))
        .execute()
    ).data
    step_times["fetch_relationship"] = time.time() - relationship_start

    if not relationship_data:
        raise HTTPException(status_code=404, detail="❌ Relationship not found")

    relationship = relationship_data[0]
    print("✅ Relationship data found", relationship)

    # ✅ Fetch conversation data
    print(f"🔹 Fetching conversation data for ID: {reply_suggestion.conversation_id}")
    conversation_start = time.time()
    conversation_data = (
        supabase.table("conversations")
        .select("*")
        .eq("id", str(reply_suggestion.conversation_id))
        .execute()
    ).data
    step_times["fetch_conversation"] = time.time() - conversation_start

    if not conversation_data:
        raise HTTPException(status_code=404, detail="❌ Conversation not found")

    conversation = conversation_data[0]

    compiled_memory =""

    if (reply_suggestion.use_memory == True):
    # ✅ Generate query for memory search
        query_start = time.time()
        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze conversation history and generate only relevant database queries in natural language. "
                        "Do not include explanations, just the queries."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
                    Based on the following conversation, generate one natural language query
                    to retrieve useful context from the database. Answer in {relationship.get('language', 'English')}.
                    
                    Conversation history:
                    {conversation.get('conversation_history')}
                    """
                }
            ],
            response_format=schemas.GptMemoryQueryOut
        )
        step_times["generate_query"] = time.time() - query_start

        query = response.choices[0].message.parsed
        print("🔹 Generated query:", query)

        # ✅ Memory search request
        memory_search_start = time.time()
        payload = {
            "user_id": relationship.get("user_id"),
            "app_id": str(MEM0_APP_ID),
            "query": query.query,
            "character_name": relationship.get("recipient_id")
        }
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }

        memory_response = requests.post(f"{MEM0_URL}/api/search_memory", json=payload, headers=headers)
        step_times["memory_search"] = time.time() - memory_search_start

        retrieved_memory = memory_response.json().get('data', {}).get('memory', [])
        compiled_memory = " ".join(retrieved_memory)

        print("🔹 Compiled retrieved memory:", compiled_memory)
    
    # ✅ Construct system prompt
    system_prompt_start = time.time()
    system_prompt = f"""
    Suggest replies based on:
    - Relationship Goal: {relationship.get('relationship_goal', 'Unknown')}
    - Recipient Personality: {relationship.get('recipient_personality_overview', 'Unknown')}
    - Recipient Communication Style: {relationship.get('recipient_communication_style_overview', 'Unknown')}
    - Conversation History: {conversation.get('conversation_history', 'Unknown')}
    - User Texting Style: {relationship.get('user_texting_style', 'Normal')}

    Additional context (if relevant):
    - Retrieved Memory: {compiled_memory}
    """
    step_times["construct_prompt"] = time.time() - system_prompt_start

    # ✅ Send request to OpenAI
    openai_start = time.time()
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Output in {relationship.get('language', 'English')}: "},
            ],
            response_format=schemas.ReplySuggestionOut
        )
        print("✅ OpenAI response received")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ GPT-4o API error: {str(e)}")
    step_times["openai_request"] = time.time() - openai_start

    # ✅ Parse AI response
    response_parse_start = time.time()
    options = completion.choices[0].message.parsed
    step_times["parse_response"] = time.time() - response_parse_start

    print(f"✅ Parsed reply suggestions: {options}")

    # ✅ Insert reply into database
    db_insert_start = time.time()
    new_reply_suggestion = {
        "id": str(uuid.uuid4()),
        "conversation_id": str(reply_suggestion.conversation_id),
        "persona_id": str(reply_suggestion.persona_id) if reply_suggestion.persona_id else None,
        "reply_1": options.reply_1,
        "reply_2": options.reply_2,
        "reply_3": options.reply_3,
        "reply_4": options.reply_4,
        "created_at": datetime.utcnow().isoformat(),
    }

    try:
        response = supabase.table("reply_suggestions").insert(new_reply_suggestion).execute()
        print("✅ Supabase insertion successful")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Database error: {str(e)}")
    step_times["insert_db"] = time.time() - db_insert_start

    # ✅ Calculate total execution time
    step_times["total_execution"] = time.time() - start_time

    # ✅ Print step times
    print("\n⏳ Execution Time Breakdown:")
    for step, duration in step_times.items():
        print(f"🕒 {step}: {duration:.4f} seconds")

    # ✅ Return response
    return schemas.ReplySuggestionOut(
        reply_1=options.reply_1,
        reply_2=options.reply_2,
        reply_3=options.reply_3,
        reply_4=options.reply_4,
    )

@app.get("/user_texting_style/{relationship_id}")
def get_user_texting_style(relationship_id: uuid.UUID):
    relationship_data = (
        supabase.table("relationships")
        .select("*")
        .eq("id", str(relationship_id))
        .execute()
    ).data

    if not relationship_data:
        raise HTTPException(status_code=404, detail="❌ Relationship not found")
    
    return relationship_data[0].get("user_texting_style")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  