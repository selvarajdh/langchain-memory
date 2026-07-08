Diagnosing Token Limit Problems
Buffer memory stores every message from a conversation and sends the entire message history to the model with every new request, as part of the input.

As a conversation grows, each request sent to the model becomes larger because more message history is included in the input context. This growth is measured in tokens — the units of text the model uses to process a request. A short word may correspond to one token, while a longer or compound word may be split into multiple tokens.

Token Growth With Buffer Memory
A conversation is made up of multiple turns. Each conversation turn is a single completed exchange that adds one user message and one model response. A single turn typically contributes 50 – 150 tokens to the conversation.

Every request to the model includes:

System instructions (~50 – 200 tokens)
Conversation history from memory (grows with each turn)
Current user input (~20 – 100 tokens)
Space reserved for the model’s response (~100 – 500 tokens)
Note: The token values here are estimated and not fixed. Actual token usage depends on the content of the messages. 

The example table below illustrates how token usage increases as more turns are added to the conversation history included in a single request.

Conversation Length	Approximate Token Usage	What the Single Request Contains
1 turn	~50 – 150 tokens	System instructions + 1 completed turn (1 user message + 1 model response) + space for the model’s reply
5 turns	~250 – 750 tokens	System instructions + 5 completed turns (5 user messages + 5 model responses) + space for the model’s reply
20 turns	~1,000 – 3,000 tokens	System instructions + 20 completed turns (20 user messages + 20 model responses) + space for the model’s reply
50 turns	~2,500 – 7,500 tokens	System instructions + 50 completed turns (50 user messages + 50 model responses) + space for the model’s reply
100 turns	~5,000 – 15,000 tokens	System instructions + 100 completed turns (100 user messages + 100 model responses) + space for the model’s reply
Note: Each request includes the accumulated conversation history and the user message being processed. This increasing request size is not unlimited. 

As requests grow in token count, token limits become an important constraint to diagnose.

The token limit constraint is enforced by the model’s context window.

A context window defines a limit on the maximum number of tokens the model can process in a single request. If the total number of tokens in a single request exceeds the defined limit, the model cannot process the request and fails to generate a response.

Context Window Size
Context window sizes vary by model and may change over time.
The following values are examples:

GPT-3.5-turbo: approximately 16,000 tokens
GPT-4: approximately 128,000 tokens
GPT-4o-mini: approximately 128,000 tokens
Note: These limits are not fixed. Context window sizes can change as models are updated or replaced. Always check the OpenAI Models documentation for the model used.

The problem is not the buffer memory itself. The problem is that requests keep getting larger. Larger requests exceed the context window and cause token overflow.

Fixing the Problem With Window Memory
=====================================

One way to prevent token overflow is to use window memory.
Instead of keeping the entire conversation, window memory stores only the most recent k exchanges. The value of k is chosen based on how much recent context is needed in the conversation.
When a new message arrives, and the window is full, older exchanges are removed. This maintains a constant request size, even as the conversation progresses.
Trade-off: Older context is lost.
Benefit: Requests stay within token limits, and conversations can continue indefinitely.
In this exercise, we will start by loading a preconfigured LangChain conversation with buffer memory and session-based message storage. This setup will allow us to observe how request size grows as the conversation history accumulates, without having to build the chain manually.
We will then:
    Measure token growth with buffer memory.
    Identify when requests exceed the context window.
    Apply window memory.
    Verify that request size remains stable.

This workflow will mirror how token limit issues are diagnosed and fixed in real production systems.

Instructions
Checkpoint 1 Passed
1. Running a Multi-Turn Support Conversation With Buffer Memory

Run through a complete support conversation to see buffer memory in action as it retains every message exchanged.

Define a session configuration dictionary named buffer_config that specifies the "session_id" as "support_buffer" to create a unique session identifier that isolates the memory storage for this conversation.
A loop structure with print() statements is provided to run through the conversation and display each turn.
Inside the loop, invoke the conversation_chain with the current turn as "input" and the session configuration, storing the result in the response variable so each message is sent to the chain while maintaining session context for conversation continuity.
Make sure to nest the session_id inside both "configurable" and "session_id" keys in the buffer_config dictionary.
Make sure the config parameter is separate from the input dictionary, and that you pass it as a second argument to the .invoke() method.

Checkpoint 2 Passed
2. Analyzing Buffer Memory Usage
    Take a closer look at what’s stored in buffer memory and calculate how much it grows with each conversation turn.
    Retrieve the chat history for the "support_buffer" session and store the messages in a variable named buffer_messages to access the complete conversation history stored in memory for analysis.Join all message contents into a single string using spaces and store it in a variable named buffer_content to create a single text representation that enables token estimation.
    Estimate the token count by multiplying the number of words in buffer_content by 1.3 to get an approximate token count, since tokens don’t map one-to-one with words. Store the result in a variable named buffer_tokens.
    The print() statements are provided to display the buffer memory analysis results.

Checkpoint 3 Passed
3. Implementing the Window Memory Trim Function
    Create a utility function that keeps conversations manageable by automatically trimming older messages beyond a set limit.
    A constant WINDOW_SIZE is set so the chat keeps only the last 5 exchanges (10 messages total).
    A function called trim_to_window is also defined, which takes two parameters:

    history (the chat history object)
    k (the number of exchanges to keep, defaulting to WINDOW_SIZE)
Inside the function:
    Check if the total number of messages in the history exceeds the window limit.
    If it does, extract only the most recent messages within the limit and store them temporarily in a variable named recent_messages to preserve the conversations that should be retained.
    Use the .clear() method to clear the entire history.
    Create a nested loop to add back only the extracted recent_messages, maintaining the sliding window of conversation context.
Checkpoint 4 Passed
4. Running a Multi-Turn Support Conversation With Window Memory
    Run the same support conversation using window memory to see how it keeps only recent exchanges while discarding older ones.
    Define a session configuration dictionary named window_config that specifies the "session_id" as "support_window" to create a unique session identifier that isolates the memory storage for this conversation.
    A loop structure with print() statements is provided to run through the conversation and display each turn.
    Inside the loop, invoke the conversation_chain with the current turn as "input" and the session configuration, storing the result in the response variable so each message is sent to the chain while maintaining session context.
    After each invocation, call the trim_to_window() function with the session’s history from the store and WINDOW_SIZE to automatically remove older messages and maintain the sliding window of recent conversations.

Checkpoint 5 Passed
5. Analyzing Window Memory Usage
    Check what’s left in window memory after trimming to see how it maintains stable token usage throughout the conversation.
    Retrieve the chat history for the "support_window" session and store the messages in a variable named window_messages to access the limited conversation history maintained by the sliding window.
    Join all message contents into a single string using spaces and store it in a variable named window_content to create a single text representation that enables token estimation.
    Estimate the token count by multiplying the number of words in window_content by 1.3 to get an approximate token count since tokens don’t map one-to-one with words. Store the result in a variable named window_tokens.
    The print() statements are provided to display the window memory analysis results.