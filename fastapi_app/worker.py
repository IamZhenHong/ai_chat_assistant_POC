from celery import Celery
import time
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import openai
from fastapi_app import schemas
from openai import OpenAI
from . import schemas
import uuid
from datetime import datetime

load_dotenv()
# Celery setup with Redis
celery_app = Celery(
    "worker",
    broker="redis://red-cuu1fk23esus73ee6ahg:6379",  # Redis as the message broker
    backend="https://dashboard.render.com/web/srv-cuu1enij1k6c738h9t40/deploys/dep-cuu1j68gph6c73b7k960",  # Redis to store results
)
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)

openai_api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI()

@celery_app.task
def update_user_texting_style(conversation_history:str, relationship_id:uuid.UUID):
    """Asynchronous task to update the user's texting style in Supabase."""
    print("📌 Data received for update:", conversation_history, relationship_id)
    relationship = supabase.from_("relationships").select("*").eq("id", str(relationship_id)).execute().data

    if not relationship:
        return {"status": "error", "message": "Relationship not found."}

    system_prompt = f"""
    Given the conversation history, update the user's texting style:

    preivous texting style:
      {relationship[0].get('user_texting_style', 'Unknown')}

    📌 **Conversation History:**
    {conversation_history}

    ⚡ **Update the user's texting style accordingly.**
    """
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Output in the text's language."},
            ],
            response_format= schemas.UserTextingStyleOut

        )
    
    except Exception as e:
        return f"Error: {e}"
    response = completion.choices[0].message.parsed

    update_response = supabase.from_("relationships").update(
        {
            "user_texting_style": response.user_texting_style
        }
    ).eq("id", str(relationship_id)).execute()


@celery_app.task
def update_relationship_overview(data: dict):
    """Asynchronous task to update the relationship overview in Supabase."""
    
    print("📌 Data received for update:", data)

    # Ensure the required data fields exist
    if not data.get("relationship_id"):
        print("❌ Error: Missing relationship_id in update_relationship_overview task.")
        return {"status": "error", "message": "Missing relationship_id"}
    
    relationship_id = data.get("relationship_id")

    relationship = supabase.from_("relationships").select("*").eq("id", str(relationship_id)).execute()

    system_prompt = f"""
    Given the previous relationship overview, update the relationship overview with the new conversation analysis:

    📌 **Previous Relationship Overview:**
    - **Relationship Stage:** {relationship.data[0].get('relationship_stage_overview', 'Unknown')}
    - **Relationship Trend:** {relationship.data[0].get('relationship_trend', 'Unknown')}
    - **User Personality Overview:** {relationship.data[0].get('user_personality_overview', 'Unknown')}
    - **User Communication Style Overview:** {relationship.data[0].get('user_communication_style_overview', 'Unknown')}
    - **Recipient Personality Overview:** {relationship.data[0].get('recipient_personality_overview', 'Unknown')}
    - **Recipient Communication Style Overview:** {relationship.data[0].get('recipient_communication_style_overview', 'Unknown')}

    📌 **Latest Conversation Analysis:**
    - **Relationship Stage:** {data.get('relationship_stage', 'Unknown')}
    - **Relationship Trend:** {data.get('relationship_trend', 'Unknown')}
    - **User Personality:** {data.get('user_personality', 'Unknown')}
    - **User Communication Style:** {data.get('user_communication_style', 'Unknown')}
    - **Recipient Personality:** {data.get('recipient_personality', 'Unknown')}
    - **Recipient Communication Style:** {data.get('recipient_communication_style', 'Unknown')}

    ⚡ **Update the relationship overview accordingly.**
    """

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Output in the text's language."},
            ],
            response_format= schemas.RelationshipOverviewOut

        )
    
    except Exception as e:
        return f"Error: {e}"
    
    response = completion.choices[0].message.parsed

    try:
        # Perform the update in Supabase
        update_response = (
            supabase.from_("relationships")
            .update({
                "relationship_stage_overview": response.relationship_stage_overview,
                "user_personality_overview": response.user_personality_overview,
                "user_communication_style_overview": response.user_communication_style_overview,
                "recipient_personality_overview":response.recipient_personality_overview,
                "recipient_communication_style_overview": response.recipient_communication_style_overview,
                "relationship_trend": response.relationship_trend_overview,
                "updated_at": datetime.utcnow().isoformat()
            })
            .eq("id", str(data["relationship_id"]))  # Ensure ID is correctly formatted
            .execute()
        )

        # ✅ Correctly check if there's an error
        if hasattr(update_response, "error") and update_response.error:
            print("❌ Supabase Error:", update_response.error)
            return {"status": "failed", "error": update_response.error}

        print("✅ Relationship updated successfully:", update_response.data)
        return {
            "status": "success",
            "relationship_id": data["relationship_id"],
            "updated_data": update_response.data,
        }

    except Exception as e:
        print("❌ Exception during update:", str(e))
        return {"status": "error", "message": str(e)}


@celery_app.task
def summarize_conversation(text: str, conversation_id: uuid.UUID):
    """Simulated text summarization task (asynchronous)."""

    system_prompt = f"Summarize the following conversation: {text}"

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            store=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Output in the text's language."},
            ],
            response_format= schemas.ConversationSummaryOut

        )
    
    except Exception as e:
        return f"Error: {e}"
    response = completion.choices[0].message.parsed

    update_response = supabase.from_("conversations").update(
        {
            "topic": response.topic,
            "conversation_summary": response.conversation_summary
        }
    ).eq("id", str(conversation_id)).execute()  # Ensures only the row with matching id is updated

    print(update_response)

    time.sleep(5)  # Simulating processing delay
    return {"status": "success", "conversation_id": str(conversation_id)}
