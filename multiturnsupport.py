from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
AWS_REGION = "us-west-2"

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

conversation_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

buffer_config = {"configurable": {"session_id": "support_buffer"}}

turns = [
    "Hi, I'm having trouble accessing my AWS S3 bucket.",
    "I get an 'Access Denied' error when I try to list objects.",
    "My IAM user has the S3ReadAccess policy attached.",
    "How do I check if my bucket policy is blocking access?",
]

if __name__ == "__main__":
    print("=== Multi-Turn Support Conversation ===\n")

    for i, turn in enumerate(turns, 1):
        print(f"Turn {i}")
        print(f"User: {turn}")
        response = conversation_chain.invoke({"input": turn}, config=buffer_config)
        print(f"Assistant: {response.content}\n")
