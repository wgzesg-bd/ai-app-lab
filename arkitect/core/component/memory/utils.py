from openai.types.responses import Response
from volcenginesdkarkruntime.types.chat.chat_completion_message import (
    ChatCompletionMessage,
)

from arkitect.types.llm.model import ArkMessage


def _ark_message_to_string(messages: list[ArkMessage | dict]) -> str:
    content = ""
    for message in messages:
        if isinstance(message, ArkMessage):
            content += f"{message.role}: {message.content}\n"
        elif isinstance(message, dict):
            content += f"{message['role']}: {message['content']}\n"
    return content


def format_ark_message_as_string(message: ArkMessage | dict | Response | ChatCompletionMessage) -> str:
    if isinstance(message, ArkMessage):
        return f"{message.role}: {message.content}\n"
    elif isinstance(message, dict):
        return f"{message['role']}: {message['content']}\n"
    elif isinstance(message, Response):
        return f"assistant: {message.choices[0].message.content}"
    elif isinstance(message, ChatCompletionMessage):
        return f"assistant: {message.content}"
    else:
        raise ValueError("Invalid message type")
    

def format_ark_message_as_dict(message: ArkMessage | dict | Response | ChatCompletionMessage) -> dict:
    if isinstance(message, ArkMessage):
        return message.model_dump()
    elif isinstance(message, dict):
        return message
    elif isinstance(message, Response):
        return {
            "role": "assistant",
            "content": message.output_text
        }
    elif isinstance(message, ChatCompletionMessage):
        return {
            "role": "assistant",
            "content": message.content
        }
    else:
        raise ValueError("Invalid message type")