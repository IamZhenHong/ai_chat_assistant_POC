import streamlit as st
import requests
import asyncio
import aiohttp
import time
import uuid

# Define the base URL for the API
# BASE_URL = "http://192.168.69.28:8000"  # Use your local IP

BASE_URL = "https://526b-60-50-200-107.ngrok-free.app"
st.title("恋爱分析")

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "recipients_list" not in st.session_state:
    st.session_state.recipients_list = []
if "selected_recipient" not in st.session_state:
    st.session_state.selected_recipient = None
if "personas_list" not in st.session_state:
    st.session_state.personas_list = []
if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "reply_options_result" not in st.session_state:
    st.session_state.reply_options_result = None
if "users_list" not in st.session_state:
    st.session_state.users_list = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "reply_options_method" not in st.session_state:
    st.session_state.reply_options_method= None



def fetch_users():
    try:
        response = requests.get(f"{BASE_URL}/users/")
        if response.status_code == 200:
            st.session_state.users_list = response.json()
        else:
            st.error(f"无法获取用户列表: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"请求失败: {e}")

def fetch_recipients():
    try:
        user_id = str(st.session_state.user_id)
        uuid.UUID(user_id)
        response = requests.get(f"{BASE_URL}/recipients/{user_id}/")
        st.markdown(response.json())
        st.session_state.recipients_list = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"请求失败: {e}")

def fetch_personas():
    try:
        response = requests.get(f"{BASE_URL}/personas/")
        if response.status_code == 200:
            st.session_state.personas_list = response.json()
        else:
            st.error(f"无法获取目标列表: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"请求失败: {e}")

fetch_personas()

st.markdown("### 选择用户")
if not st.session_state.users_list:
    fetch_users()

user_options = ["创建新用户"] + [f"{user['id']}: {user['name']}" for user in st.session_state.users_list]
selected_user = st.selectbox("选择用户", user_options)

is_new_user = selected_user == "创建新用户"

if selected_user != "创建新用户":
    user_id = selected_user.split(":")[0]
    if st.session_state.user_id != user_id:
        st.session_state.user_id = user_id
        st.session_state.recipients_list = []  # Reset recipients when user changes
        fetch_recipients()

if is_new_user:
    st.markdown("#### 创建用户")
    form_data = {
        "name": "",
    }
else:
    user_id = selected_user.split(":")[0]
    st.session_state.user_id = user_id
    st.markdown("编辑用户")
    selected_user = next((user for user in st.session_state.users_list if user["id"] == user_id), None)

    if selected_user == None:
        st.error("用户不存在！")

    form_data = {
        "name": selected_user["name"],
    }


# Streamlit UI components
name = st.text_input("姓名", value=form_data.get("name", ""))
gender = st.text_input("性别", value=form_data.get("gender", ""))
language = st.text_input("语言", value=form_data.get("language", ""))
age = st.text_input("年龄", value=form_data.get("age", ""))
about_me = st.text_area("关于我", value=form_data.get("about_me", ""))


if is_new_user:
    create_user_button = st.button("创建用户")
    if create_user_button:
        if name:
            try:
                response = requests.post(f"{BASE_URL}/users/", json={"name": name,"gender":gender, "language":language, "age":age, "about_me":about_me  })
                if response.status_code == 200:
                    st.session_state.user_id = response.json()["id"]
                    st.markdown(st.session_state.user_id)
                    st.success("用户创建成功！")
                    fetch_users()
                    fetch_recipients()
                else:
                    st.error(f"错误: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"请求失败: {e}")
        else:
            st.warning("请提供姓名。")
else:
    update_user_button = st.button("更新用户")
    if update_user_button:
        if name:
            try:
                user_id = selected_user["id"]
                response = requests.put(f"{BASE_URL}/users/{user_id}/", json={"name": name})
                if response.status_code == 200:
                    st.success("用户更新成功！")
                    fetch_users()
                else:
                    st.error(f"错误: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"请求失败: {e}")
        else:
            st.warning("请提供姓名。")

if st.session_state.user_id:
    st.markdown("### 选择或创建联系人")

    recipient_options = ["创建新联系人"] + [f"{rec['id']}: {rec['name']}" for rec in st.session_state.recipients_list]
    selected_option = st.selectbox("选择联系人", recipient_options)

    is_new_recipient = selected_option == "创建新联系人"

    if is_new_recipient:
        st.markdown("#### 创建联系人")
        form_data = {
            "recipient_name": "",
            "gender": "",
            "language": "",
            "age": "",
            "about_me": "",
        }

    else:
        recipient_id = selected_option.split(":")[0]
        # st.session_state.selected_recipient = recipient_id
        st.markdown("编辑联系人")
        selected_recipient = next((recipient for recipient in st.session_state.recipients_list if recipient["id"] == recipient_id), None)
        st.session_state.selected_recipient = selected_recipient
        if selected_recipient == None:
            st.error("联系人不存在！")

        form_data = {
            "recipient_name": selected_recipient["name"],
            "gender": selected_recipient["gender"],
            "language": selected_recipient["language"],
            "age": selected_recipient["age"],
            "about_me": selected_recipient["about_me"],
        }

    recipient_name = st.text_input("姓名", value=form_data["recipient_name"],key="update_recipient_name")
    recipient_gender = st.text_input("性别", value=form_data["gender"],key="update_recipient_gender")
    recipient_language = st.text_input("语言", value=form_data["language"],key="update_recipient_language")
    recipient_age = st.text_input("年龄", value=form_data["age"],key="update_recipient_age")
    recipient_about_me = st.text_area("关于我", value=form_data["about_me"],key="update_recipient_about_me")

    if is_new_recipient:
        create_recipient_button = st.button("创建联系人")
        if create_recipient_button:
            if name:
                try:
                    response = requests.post(
                        f"{BASE_URL}/recipients/",
                        json={
                            "user_id": str(st.session_state.user_id),
                            "name": recipient_name,
                            "gender": recipient_gender,
                            "language": recipient_language,
                            "age": recipient_age,
                            "about_me": recipient_about_me,
                        },
                    )
                    if response.status_code == 200:
                        recipient_data = response.json()
                        fetch_recipients()
                        st.session_state.selected_recipient = next(
    (recipient for recipient in st.session_state.recipients_list if recipient["id"] == recipient_data["id"]),
    None  # ✅ Returns None if no match is found
)

                        st.success("联系人创建成功！")
    
                    else:
                        st.error(f"错误: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"请求失败: {e}")
            else:
                st.warning("请提供姓名。")
    else:
        update_recipient_button = st.button("更新联系人")
        if update_recipient_button:
            if name:
                try:
                    response = requests.put(
                        f"{BASE_URL}/recipients/{recipient_id}/",
                        json={
                            "name": recipient_name,
                            "gender": recipient_gender,
                            "language": recipient_language,
                            "age": recipient_age,
                            "about_me": recipient_about_me,
                        },
                    )
                    if response.status_code == 200:
                        st.success("联系人更新成功！")
                        fetch_recipients()
                    else:
                        st.error(f"错误: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"请求失败: {e}")

st.markdown(st.session_state.selected_recipient)

st.markdown("### 输入当前对话")
current_convo = st.text_area("当前对话", placeholder="输入当前的对话内容...")

if st.button("生成分析"): 
    if current_convo and st.session_state.selected_recipient:
        st.info("生成中...")
        try:
            recipient = st.session_state.selected_recipient
            if "relationship_id" in recipient and recipient["relationship_id"] is not None:
                relationship_id = recipient["relationship_id"]
            else:
                st.error("Error: 'relationship_id' is missing or invalid.")
                relationship_id = None  # Handle gracefully

            response = requests.post(
                f"{BASE_URL}/conversation_snippets/",
                json=[{
                    "relationship_id": st.session_state.selected_recipient["relationship_id"],
                    "sequence_id": 1,
                    "content": current_convo,
                }],
            )

            if response.status_code == 200:
                st.success("对话上传成功！")
                conversation_id = response.json()["conversation_id"]
                st.session_state.conversation_id = conversation_id

            response = requests.post(
                f"{BASE_URL}/conversation_analysis/",
                json={"conversation_id": st.session_state.conversation_id, "relationship_id": st.session_state.selected_recipient["relationship_id"]},
            )


            # Check if the request was successful
            if response.status_code == 200:
                st.session_state.analysis_result = response.json()
                
                st.success("✅ 对话分析生成成功！")

                analysis = st.session_state.analysis_result  # Store JSON response

                # 📌 Display Analysis Results with Icons & Formatting
                st.subheader("📊 对话分析结果")

                st.markdown(f"""
                - **🗣️ 用户沟通风格:** {analysis.get('user_communication_style', '未提供')}
                - **😊 用户性格特征:** {analysis.get('user_personality', '未提供')}
                - **💬 对方沟通风格:** {analysis.get('recipient_communication_style', '未提供')}
                - **🎭 对方性格:** {analysis.get('recipient_personality', '未提供')}
                - **🔄 关系阶段:** {analysis.get('relationship_stage', '未提供')}
                - **📉 关系趋势:** {analysis.get('relationship_trend', '未提供')}
                """, unsafe_allow_html=True)


            else:
                st.error(f"错误: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"请求失败: {e}")
    else:
        st.warning("请填写所有必要信息！")


def format_persona(persona):
    if isinstance(persona, dict) and "name" in persona and "gender" in persona:
        return persona["name"] + " " + "(" + persona["gender"] + ")"  # Format for personas
    return str(persona)  # Just return the string for extra options

st.markdown("### 选择性格")

# Define extra options
extra_options = ["Your Persona", "Normal Persona"]

# Create the selectbox with both personas and extra options
selected_persona = st.selectbox(
    "选择性格",
    extra_options + st.session_state.personas_list,  # Combine both lists
    format_func=format_persona,  # Use the updated format function
    key="selected_persona"
)

st.write(f"你选择了: {selected_persona}")



if selected_persona and selected_persona!="Your Persona" and selected_persona!="Normal Persona":
    # Using Markdown for a simple nicely formatted output
    st.markdown("### Persona Details")
    st.markdown(f"**Name:** {selected_persona['name']}")
    st.markdown(f"**Gender:** {selected_persona['gender']}")
    st.markdown(f"**Description:** {selected_persona['description']}")


if selected_persona:
    if selected_persona == "Your Persona":
        st.session_state.reply_options_method = 1
    elif selected_persona == "Normal Persona":
        st.session_state.reply_options_method = 2
    else:
        st.session_state.reply_options_method = 3
        st.session_state.persona_id = selected_persona["id"]

if st.button("生成回复选项"):
    
    if current_convo and st.session_state.selected_recipient:
        st.info("生成中...")
        try:
            if st.session_state.conversation_id is None:
                response = requests.post(
                    f"{BASE_URL}/conversation_snippets/",
                    json=[{
                        "relationship_id": st.session_state.selected_recipient["relationship_id"],
                        "sequence_id": 1,
                        "content": current_convo,
                    }],
                )
            
                if response.status_code == 200:
                    st.success("对话上传成功！")
                    conversation_id = response.json()["conversation_id"]
                    st.session_state.conversation_id = conversation_id
            
            response = requests.post(
                f"{BASE_URL}/reply_suggestions/",
                json={"option": st.session_state.reply_options_method, "conversation_id": st.session_state.conversation_id, "persona_id": st.session_state.persona_id, "relationship_id": st.session_state.selected_recipient["relationship_id"]},
            )

            # Assuming response contains reply_1, reply_2, reply_3, reply_4
            if response.status_code == 200:
                st.session_state.reply_options_result = response.json()
                
                st.success("✅ 回复选项生成成功！")
                
                replies = st.session_state.reply_options_result  # Store JSON response
                
                # Display the responses in a structured way
                st.subheader("💬回复选项：")
                
                # Use `st.markdown` with bullet points for better readability
                st.markdown(f"""
                - **💡 选项 1:** {replies.get('reply_1', '未提供')}
                - **💡 选项 2:** {replies.get('reply_2', '未提供')}
                - **💡 选项 3:** {replies.get('reply_3', '未提供')}
                - **💡 选项 4:** {replies.get('reply_4', '未提供')}
                """, unsafe_allow_html=True)
        except requests.exceptions.RequestException as e:
            st.error(f"请求失败: {e}")
# st.markdown("### 选择性格")

# # Create the selectbox using the personas list directly
# selected_persona = st.selectbox(
#     "选择性格",
#     st.session_state.personas_list,
#     format_func=format_persona,
#     key="selected_persona"
# )



# st.markdown("---")
# st.markdown(st.session_state.user_id)

# # User Profile Creation
# st.markdown("### 创建用户")
# if st.session_state.user_id is None:
#     username = st.text_input("输入您的姓名")
#     if st.button("创建用户"):
#         response = requests.post(f"{BASE_URL}/users/", json={"name": username})
#         if response.status_code == 200:
#             st.session_state.user_id = response.json()["id"]
#             st.success("用户创建成功！")
#         else:
#             st.error("用户创建失败！")
# else:
#     st.success("用户已登录")

# # Fetch existing recipients
# def fetch_recipients():
#     try:
#         response = requests.get(f"{BASE_URL}/recipients/")
#         if response.status_code == 200:
#             st.session_state.recipients_list = response.json()
#         else:
#             st.error(f"无法获取目标列表: {response.status_code} - {response.text}")
#     except requests.exceptions.RequestException as e:
#         st.error(f"请求失败: {e}")

# def fetch_personas():
#     try:
#         response = requests.get(f"{BASE_URL}/personas/")
#         if response.status_code == 200:
#             st.session_state.personas_list = response.json()
#         else:
#             st.error(f"无法获取性格列表: {response.status_code} - {response.text}")
#     except requests.exceptions.RequestException as e:
#         st.error(f"请求失败: {e}")

# if not st.session_state.recipients_list:
#     fetch_recipients()
# if not st.session_state.personas_list:
#     fetch_personas()

# st.markdown("### 选择或创建联系人")
# recipient_options = ["创建新联系人"] + [f"{rec['id']}: {rec['name']}" for rec in st.session_state.recipients_list]
# selected_option = st.selectbox("选择联系人", recipient_options)

# is_new_recipient = selected_option == "创建新联系人"

# if is_new_recipient:
#     st.markdown("#### 创建联系人")
#     name = st.text_input("姓名")
#     gender = st.text_input("性别")
#     language = st.text_input("语言")
    
#     if st.button("创建联系人"):
#         if name:
#             try:
#                 response = requests.post(
#                     f"{BASE_URL}/recipients/",
#                     json={
#                         "user_id": st.session_state.user_id,
#                         "name": name,
#                         "gender": gender,
#                         "language": language,
#                     },
#                 )
#                 if response.status_code == 200:
#                     recipient_data = response.json()
#                     st.session_state.selected_recipient = recipient_data["id"]
#                     st.success("联系人创建成功！")
#                     fetch_recipients()
#                 else:
#                     st.error(f"错误: {response.status_code} - {response.text}")
#             except requests.exceptions.RequestException as e:
#                 st.error(f"请求失败: {e}")
#         else:
#             st.warning("请提供姓名。")
# else:
#     recipient_id = selected_option.split(":")[0]
#     st.session_state.selected_recipient = recipient_id
    
#     st.markdown("### 输入当前对话")
#     current_convo = st.text_area("当前对话", placeholder="输入当前的对话内容...")

#     st.markdown("### 选择性格")
#     selected_persona = st.selectbox("选择性格", st.session_state.personas_list, format_func=lambda p: f"{p['name']} ({p['gender']})")
#     if selected_persona:
#         st.session_state.selected_persona = selected_persona["id"]

#     async def fetch_reply_options(session, recipient_id, persona_id, conversation):
#         async with session.post(
#             f"{BASE_URL}/reply_suggestions/",
#             json={"conversation_id": str(uuid.uuid4()), "persona_id": persona_id, "relationship_id": recipient_id},
#         ) as response:
#             return await response.json()

#     async def fetch_analysis(session, recipient_id, conversation):
#         async with session.post(
#             f"{BASE_URL}/conversation_analysis/",
#             json={"conversation_id": str(uuid.uuid4()), "relationship_id": recipient_id},
#         ) as response:
#             return await response.json()

#     async def fetch_all_data(recipient_id, conversation, persona_id):
#         async with aiohttp.ClientSession() as session:
#             reply_task = fetch_reply_options(session, recipient_id, persona_id, conversation)
#             analysis_task = fetch_analysis(session, recipient_id, conversation)
#             results = await asyncio.gather(reply_task, analysis_task, return_exceptions=True)
#             return results

#     if st.button("生成分析和回复建议"):
#         if current_convo and st.session_state.selected_recipient and st.session_state.selected_persona:
#             st.info("并行生成中...")
#             results = asyncio.run(fetch_all_data(st.session_state.selected_recipient, current_convo, st.session_state.selected_persona))
            
#             if isinstance(results[0], dict):
#                 st.session_state.reply_options_result = results[0]
#                 st.success("回复选项生成成功！")
#             if isinstance(results[1], dict):
#                 st.session_state.analysis_result = results[1]
#                 st.success("对话分析生成成功！")
#         else:
#             st.warning("请填写所有必要信息！")

# if st.session_state.analysis_result:
#     st.subheader("对话分析结果")
#     st.markdown(st.session_state.analysis_result)

# if st.session_state.reply_options_result:
#     st.subheader("回复选项")
#     for i, option in enumerate(st.session_state.reply_options_result.values(), 1):
#         st.markdown(f"**选项 {i}:** {option}")

# st.markdown("---")
# st.caption("由 FastAPI 和 Streamlit 提供支持")
