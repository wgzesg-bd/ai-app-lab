# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from openai.types.responses import Response
from typing_extensions import override
from volcenginesdkarkruntime import AsyncArk
from volcenginesdkarkruntime.types.chat.chat_completion_message import (
    ChatCompletionMessage,
)

from arkitect.core.component.memory.base_memory_service import (
    BaseMemoryService,
    Memory,
    SearchMemoryResponse,
)
from arkitect.core.component.memory.utils import format_message_as_string
from arkitect.types.llm.model import Message
from arkitect.utils.common import Singleton

DEFAULT_SEARCH_MEM_PROMPT = """
You have obtained a series of interactions between a user and an AI assistant.
Please identify the user’s profile and other key information from
past interactions to help answer the user’s new question.
"""

DEFAULT_SEARCH_LLM_MODEL = "doubao-1-5-pro-32k-250115"


class InMemoryMemoryService(BaseMemoryService):
    def __init__(
        self,
        default_search_model: str = DEFAULT_SEARCH_LLM_MODEL,
        default_search_prompt: str = DEFAULT_SEARCH_MEM_PROMPT,
    ) -> None:
        self.default_search_model = default_search_model
        self.default_search_prompt = default_search_prompt

        self.memory: dict = {}
        self._cached_query: dict = {}
        self._llm = AsyncArk()

    @override
    async def update_memory(
        self,
        user_id: str,
        new_messages: list[Message | dict | Response | ChatCompletionMessage],
        **kwargs: Any,
    ) -> None:
        if user_id not in self.memory:
            self.memory[user_id] = []
        self.memory[user_id].extend(new_messages)
        # invalidate cache
        self._cached_query[user_id] = {}

    @override
    async def search_memory(
        self,
        user_id: str,
        query: str,
        **kwargs: Any,
    ) -> SearchMemoryResponse:
        if user_id not in self.memory:
            return SearchMemoryResponse(
                memories=[
                    Memory(
                        memory_content="no memory found for this user",
                        reference=None,
                    )
                ]
            )
        if self._cached_query.get(user_id, {}).get(query, None) is not None:
            return self._cached_query[user_id][query]
        memories = self.memory[user_id]
        results = "用户过去的交互记录\n\n"
        for memory in memories:
            content = format_message_as_string(memory)
            results += content
        summary = await self._llm.chat.completions.create(
            model=self.default_search_model,
            messages=[
                {
                    "role": "system",
                    "content": self.default_search_prompt,
                },
                {
                    "role": "user",
                    "content": results,
                },
            ],
            stream=False,
        )
        memory_response = SearchMemoryResponse(
            memories=[
                Memory(
                    memory_content=summary.choices[0].message.content,
                    reference=None,
                )
            ]
        )
        if user_id not in self._cached_query:
            self._cached_query[user_id] = {}
        self._cached_query[user_id][query] = memory_response
        return memory_response

    @override
    async def delete_user(self, user_id: str) -> None:
        if user_id in self.memory:
            del self.memory[user_id]


class InMemoryMemoryServiceSingleton(InMemoryMemoryService, Singleton):
    pass
