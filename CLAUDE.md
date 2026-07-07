# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```cmd
set PYTHONIOENCODING=utf-8 && python main.py
```

The app is fully interactive — type messages at the `You:` prompt and press Enter. Type `quit`, `exit`, or `q` to stop.

## Installing Dependencies

```cmd
pip install -r requirements.txt
```

## AWS Configuration

The app calls Amazon Bedrock (`us-west-2`) using credentials from `~/.aws/credentials` (shared credentials file). If you hit an `ExpiredTokenException`, refresh credentials before running:

```cmd
aws sso login
```

Verify credentials are active:

```cmd
aws sts get-caller-identity
```

## Architecture

`main.py` is the entire application — a multi-turn conversational CLI built with LangChain and Amazon Bedrock.

**Model**: `us.anthropic.claude-haiku-4-5-20251001-v1:0` via `ChatBedrockConverse` (cross-region inference profile, `us-west-2`).

**Memory pattern**: Manual buffer memory using `InMemoryChatMessageHistory`. Each session is keyed by `session_id` in a module-level `store` dict. The `chat()` function reads the history buffer, formats it into the prompt via `MessagesPlaceholder`, invokes the chain, then appends both the user message and AI reply back to the buffer.

**Prompt structure**:
1. System message — sets assistant persona
2. `MessagesPlaceholder("history")` — injects full conversation buffer
3. Human message — current user input

**Limitations**: Buffer memory retains every message verbatim, so token usage grows linearly with conversation length. For long sessions, consider switching to window or summary memory strategies.
