Add buffer memory to the application, we’ll:

Create a session history function.
Build a prompt template that includes conversation history.
Wrap the chain with memory management.
Test the conversation flow.

Implementing Your First Memory:
Now that we understand how memory operates, 
we’ll build a personal assistant that preserves conversation history. 
We’ll use buffer memory to get started.

Buffer memory is the simplest memory strategy — 
it stores every user message and model response exactly as they occur, 
without summarizing, filtering, or removing. 
This makes it an ideal starting point for understanding how memory integrates into a LangChain application.

Implementing buffer memory requires handling two responsibilities:

Storing conversation messages
attaching those messages to each model request automatically
LangChain handles these responsibilities using two components.

InMemoryChatMessageHistory: This component stores conversation messages in memory. It keeps a sequenced list of user messages and model responses for an active conversation.

RunnableWithMessageHistory: This component tracks chat history during each model request. It automatically stores and retrieves messages before passing them to the model.

Together, these components enable session-based memory management, where each conversation is identified by a unique session ID that serves as a label to keep messages from different conversations separate.

How Session-Based Memory Works
Note: Buffer memory does not scale well for long conversations. 
Since every message is retained, token usage increases with each turn. 
For extended conversations, other memory strategies, such as window or summary memory, are more efficient. 

To add buffer memory to the application, we’ll:

1) Create a session history function.
2) Build a prompt template that includes conversation history.
3) Wrap the chain with memory management.
4) Test the conversation flow.

1. Creating the Session History Function
    -> A dictionary named store and a function named get_session_history() are defined to manage chat history across different sessions.
       The function takes a session_id parameter and returns the corresponding chat history from the store.
    -> Check if the session_id exists in the store dictionary. 
       If it doesn’t exist, 
            create a new InMemoryChatMessageHistory object and add it to the store with the session_id as the key.

2. Creating a Conversational Prompt Template
    Use the ChatPromptTemplate class to create a prompt template named prompt that includes conversation history. This template includes three components:

    Add a system message that defines the assistant’s role and behavior.
    Add a MessagesPlaceholder with variable_name set to "history" for storing past conversation messages.
    Add a human message template with an input placeholder for the current user message.
    Create a base_chain that connects the prompt template, the language model, and a string output parser so each component passes its output to the next.

    Checkpoint 3 Step instruction is unavailable until previous steps are completed
3. Wrapping the Chain With Message History
    At this point, we have message storage in place and a prompt that accepts conversation history. 
    Now, use RunnableWithMessageHistory to wrap the conversation chain and enable automatic history handling. 
    This ensures that stored messages are retrieved before each request and updated after each response.

    Save the wrapped chain in a variable named conversation_chain.

Pass the following arguments:
    base_chain as the runnable to wrap
    get_session_history to retrieve or create chat history for each session
    input_messages_key set to "input" to indicate where the user’s current message is provided
    history_messages_key set to "history" to indicate where past conversation messages are stored
    Next, define a session configuration by creating a nested dictionary with a "configurable" key that contains a "session_id" set to "user-101". Assign this configuration to a variable named session_config. This determines which conversation history is used when invoking the chain.

Checkpoint 4 Step instruction is unavailable until previous steps are completed
4. Displaying the Conversation History

Retrieve the chat history for the session by calling the get_session_history() function with "user-101" as the session ID and store the result in a variable named session_history.

The provided loop iterates through all messages in the session history, assigns a role label "User" or "Assistant" based on the message type, and prints each message with its number, role, and content.