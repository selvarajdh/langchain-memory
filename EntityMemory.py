import json
import uuid

from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
AWS_REGION = "us-west-2"

llm = ChatBedrockConverse(
    model=MODEL_ID,
    region_name=AWS_REGION,
)

# ---------------------------------------------------------------------------
# Main conversation chain
# Injects: system persona, known entity facts, recent turns, current input
# ---------------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "{system_prompt}\n\n"
        "Known facts about entities mentioned in this conversation:\n{entity_context}",
    ),
    MessagesPlaceholder(variable_name="recent_history"),
    ("human", "{human_prompt}"),
])

chain = prompt | llm

# ---------------------------------------------------------------------------
# Entity extraction chain
# After each turn, ask the LLM what entities appeared and what was learned.
# Returns JSON so we can merge facts into the store programmatically.
# ---------------------------------------------------------------------------
extract_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an entity extractor. Given a single conversation exchange, "
        "identify every named entity (person, place, product, organisation, "
        "technology, concept, etc.) and the key facts stated about it.\n"
        "Reply with ONLY a JSON object whose keys are entity names and whose "
        "values are short fact strings. Example:\n"
        '{{"AWS": "Amazon cloud provider", "Lambda": "serverless compute service from AWS"}}\n'
        "If no entities are present, reply with {{}}.",
    ),
    (
        "human",
        "Human: {human_message}\nAI: {ai_message}",
    ),
])

extract_chain = extract_prompt | llm

# ---------------------------------------------------------------------------
# Entity update chain
# Merges a new fact with whatever we already know about an entity.
# ---------------------------------------------------------------------------
merge_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a knowledge curator. Merge two pieces of information about "
        "the same entity into a single, concise fact string. Remove duplicates. "
        "Reply with ONLY the merged fact — no preamble.",
    ),
    (
        "human",
        "Existing: {existing}\nNew: {new_fact}",
    ),
])

merge_chain = merge_prompt | llm

# ---------------------------------------------------------------------------
# Session store
# Each session holds:
#   "entities"  — dict[str, str]  entity name → accumulated fact string
#   "recent"    — list[BaseMessage]  rolling verbatim window
# ---------------------------------------------------------------------------
store: dict[str, dict] = {}

RECENT_WINDOW = 4  # human+AI pairs kept verbatim


def get_session(session_id: str) -> dict:
    if session_id not in store:
        store[session_id] = {"entities": {}, "recent": []}
    return store[session_id]


def _extract_entities(human_message: str, ai_message: str) -> dict[str, str]:
    result = extract_chain.invoke({
        "human_message": human_message,
        "ai_message": ai_message,
    })
    raw = result.content.strip()
    # Strip markdown fences if the model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw)
        return {k: str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _merge_fact(existing: str, new_fact: str) -> str:
    if not existing:
        return new_fact
    result = merge_chain.invoke({"existing": existing, "new_fact": new_fact})
    return result.content.strip()


def _update_entities(session: dict, human_message: str, ai_message: str) -> None:
    new_entities = _extract_entities(human_message, ai_message)
    for entity, fact in new_entities.items():
        session["entities"][entity] = _merge_fact(
            session["entities"].get(entity, ""), fact
        )

    # Maintain verbatim recent window
    session["recent"].append(HumanMessage(content=human_message))
    session["recent"].append(AIMessage(content=ai_message))
    if len(session["recent"]) > RECENT_WINDOW * 2:
        session["recent"] = session["recent"][-(RECENT_WINDOW * 2):]


def _build_entity_context(entities: dict[str, str]) -> str:
    print("\n[DEBUG _build_entity_context] Raw entity dict (JSON):")
    print(json.dumps(entities, indent=2))
    if not entities:
        return "None yet."
    context = "\n".join(f"- {name}: {fact}" for name, fact in entities.items())
    print("[DEBUG _build_entity_context] Formatted context string:")
    print(context)
    print()
    return context


def chat(session_id: str, system_prompt: str, human_prompt: str) -> str:
    session = get_session(session_id)

    response = chain.invoke({
        "system_prompt": system_prompt,
        "entity_context": _build_entity_context(session["entities"]),
        "recent_history": session["recent"],
        "human_prompt": human_prompt,
    })

    ai_reply = response.content
    _update_entities(session, human_prompt, ai_reply)
    return ai_reply


PRESET_PROMPTS = {
    "1": (
        "Cloud Computing Expert",
        "You are a helpful AI assistant with expertise in cloud computing. "
        "Provide clear, concise, and accurate answers.",
    ),
    "2": (
        "Python Developer",
        "You are an expert Python developer. Help with code, debugging, best "
        "practices, and library recommendations.",
    ),
    "3": (
        "DevOps Engineer",
        "You are a senior DevOps engineer. Assist with CI/CD pipelines, "
        "containerisation, infrastructure-as-code, and monitoring.",
    ),
    "4": (
        "General Assistant",
        "You are a helpful, friendly AI assistant. Answer questions clearly "
        "and concisely on any topic.",
    ),
}


def choose_system_prompt() -> str:
    print("=" * 55)
    print("  EntityMemory Conversational Assistant")
    print("=" * 55)
    print("\nChoose an assistant persona:\n")
    for key, (name, _) in PRESET_PROMPTS.items():
        print(f"  [{key}] {name}")
    print("  [5] Enter a custom system prompt")
    print()

    while True:
        choice = input("Your choice (1-5): ").strip()
        if choice in PRESET_PROMPTS:
            name, prompt = PRESET_PROMPTS[choice]
            print(f"\nPersona selected: {name}")
            return prompt
        if choice == "5":
            custom = input("Enter your custom system prompt: ").strip()
            if custom:
                print("\nCustom persona set.")
                return custom
            print("Prompt cannot be empty. Try again.")
        else:
            print("Invalid choice. Enter 1-5.")

if __name__ == "__main__":
    system_prompt = choose_system_prompt()
    session_id = str(uuid.uuid4())

    print(f"\nSession ID: {session_id}")
    print("Chat started (entity memory active). Type 'quit' to exit.\n")

    while True:
        human_prompt = input("You: ").strip()
        if not human_prompt:
            continue
        if human_prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        reply = chat(session_id, system_prompt, human_prompt)
        print(f"\nAssistant: {reply}\n")

        # Show the live entity store so you can watch facts accumulate
        session = get_session(session_id)
        if session["entities"]:
            print("[Entities]")
            for name, fact in session["entities"].items():
                print(f"  {name}: {fact}")
            print()
