from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
AWS_REGION = "us-west-2"

# Number of conversation turns to retain in the window (1 turn = 1 human + 1 AI message)
WINDOW_SIZE = 2

llm = ChatBedrockConverse(
    model=MODEL_ID,
    region_name=AWS_REGION,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support assistant. Be concise and professional."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def get_windowed_messages(history: InMemoryChatMessageHistory) -> list:
    """Return only the last WINDOW_SIZE turns (each turn = 2 messages)."""
    return history.messages[-(WINDOW_SIZE * 2):]

def chat(session_id: str, user_input: str) -> str:
    history = get_session_history(session_id)
    windowed = get_windowed_messages(history)

    response = chain.invoke({
        "history": windowed,
        "input": user_input,
    })

    history.add_user_message(user_input)
    history.add_ai_message(response.content)
    return response.content

window_config = {"configurable": {"session_id": "support_window"}}

turns = [
    "Hi, I'm having trouble accessing my AWS S3 bucket.",
    "I get an 'Access Denied' error when I try to list objects.",
    "My IAM user has the S3ReadAccess policy attached.",
    "How do I check if my bucket policy is blocking access?",
    "What if the bucket policy is empty but I still get the error?",
    "Can you summarize the steps I should take to resolve this?",
]

if __name__ == "__main__":
    session_id = window_config["configurable"]["session_id"]
    print("=== Multi-Turn Support Conversation With Window Memory ===")
    print(f"Window size: {WINDOW_SIZE} turns | Session: {session_id}\n")

    for i, turn in enumerate(turns, 1):
        history = get_session_history(session_id)
        msgs_in_window = len(get_windowed_messages(history))

        print(f"Turn {i} (messages in window: {msgs_in_window})")
        print(f"User: {turn}")
        response = chat(session_id, turn)
        print(f"Assistant: {response}\n")
