import uuid

from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
AWS_REGION = "us-west-2"

llm = ChatBedrockConverse(
    model=MODEL_ID,
    region_name=AWS_REGION,
)

# Prompt for the main conversation — injects a running summary + recent turns
prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}\n\nConversation summary so far:\n{summary}"),
    MessagesPlaceholder(variable_name="recent_history"),
    ("human", "{human_prompt}"),
])

chain = prompt | llm

# Prompt used to compress the summary after each turn
summarize_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a concise summarizer. Given the previous summary and a new "
        "exchange, produce an updated summary that captures all key facts, "
        "decisions, and context. Be brief.",
    ),
    (
        "human",
        "Previous summary:\n{previous_summary}\n\n"
        "New exchange:\nHuman: {human_message}\nAI: {ai_message}\n\n"
        "Updated summary:",
    ),
])

summarize_chain = summarize_prompt | llm

# Session store: each session keeps a rolling summary + a short recent-turns window
store: dict[str, dict] = {}

RECENT_WINDOW = 4  # number of most-recent messages (human+AI pairs) kept verbatim


def get_session(session_id: str) -> dict:
    if session_id not in store:
        store[session_id] = {"summary": "", "recent": []}
    return store[session_id]


def _update_summary(session: dict, human_message: str, ai_message: str) -> None:
    result = summarize_chain.invoke({
        "previous_summary": session["summary"],
        "human_message": human_message,
        "ai_message": ai_message,
    })
    session["summary"] = result.content.strip()

    # Keep only the most-recent RECENT_WINDOW messages in the verbatim buffer
    session["recent"].append(HumanMessage(content=human_message))
    session["recent"].append(AIMessage(content=ai_message))
    if len(session["recent"]) > RECENT_WINDOW * 2:
        session["recent"] = session["recent"][-(RECENT_WINDOW * 2):]


def chat(session_id: str, system_prompt: str, human_prompt: str) -> str:
    session = get_session(session_id)

    response = chain.invoke({
        "system_prompt": system_prompt,
        "summary": session["summary"] or "No conversation yet.",
        "recent_history": session["recent"],
        "human_prompt": human_prompt,
    })

    ai_reply = response.content
    _update_summary(session, human_prompt, ai_reply)
    return ai_reply


if __name__ == "__main__":
    system_prompt = (
        "You are a helpful AI assistant with expertise in cloud computing. "
        "Provide clear, concise, and accurate answers."
    )
    session_id = str(uuid.uuid4())

    print(f"Session ID: {session_id}")
    print("Chat started (summary memory active). Type 'quit' to exit.\n")

    while True:
        human_prompt = input("You: ").strip()
        if human_prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        reply = chat(session_id, system_prompt, human_prompt)
        print(f"\nAssistant: {reply}\n")

        # Show the live summary so you can observe it being compressed
        session = get_session(session_id)
        print(f"[Summary] {session['summary']}\n")
