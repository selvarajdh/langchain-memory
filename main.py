import uuid

from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
AWS_REGION = "us-west-2"

llm = ChatBedrockConverse(
    model=MODEL_ID,
    region_name=AWS_REGION,
)

# Step 2: Prompt template that includes conversation history
prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{human_prompt}"),
])

chain = prompt | llm

# Step 1: Session history store (buffer memory per session)
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Step 3: Memory-aware invoke — reads history, calls LLM, updates buffer
def chat(session_id: str, system_prompt: str, human_prompt: str) -> str:
    history = get_session_history(session_id)
    response = chain.invoke({
        "system_prompt": system_prompt,
        "history": history.messages,
        "human_prompt": human_prompt,
    })
    history.add_user_message(human_prompt)
    history.add_ai_message(response.content)
    return response.content

# Step 4: Test the conversation flow
if __name__ == "__main__":
    system_prompt = (
        "You are a helpful AI assistant with expertise in cloud computing. "
        "Provide clear, concise, and accurate answers."
    )
    session_id = str(uuid.uuid4())

    print(f"Session ID: {session_id}")
    print("Chat started (buffer memory active). Type 'quit' to exit.\n")

    while True:
        human_prompt = input("You: ").strip()
        if human_prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        reply = chat(session_id, system_prompt, human_prompt)
        print(f"\nAssistant: {reply}\n")
