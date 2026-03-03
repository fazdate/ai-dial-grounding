import asyncio
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient


BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }


llm_client = AzureChatOpenAI(
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version="",
    azure_deployment="gpt-4o",
)

token_tracker = TokenTracker()

def join_context(context: list[dict[str, Any]]) -> str:
    formatted_users = []
    for user in context:
        user_str = "User:"
        for key, value in user.items():
            user_str += f"\n  {key}: {value}"
        formatted_users.append(user_str)
    return "\n\n".join(formatted_users)

async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    response = await llm_client.ainvoke(messages)
    usage = response.response_metadata.get('token_usage', {})
    total_tokens = usage.get('total_tokens', 0)
    token_tracker.add_tokens(total_tokens)
    print(f"Response: {response.content}")
    print(f"Total tokens: {total_tokens}")
    return response.content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        user_client = UserClient()
        all_users = user_client.get_all_users()

        batch_size = 100
        user_batches = [all_users[i:i + batch_size] for i in range(0, len(all_users), batch_size)]

        tasks = []
        for batch in user_batches:
            context = join_context(batch)
            user_prompt = USER_PROMPT.format(context=context, query=user_question)
            task = generate_response(BATCH_SYSTEM_PROMPT, user_prompt)
            tasks.append(task)

        batch_results = await asyncio.gather(*tasks)

        filtered_results = [result for result in batch_results if result != "NO_MATCHES_FOUND"]

        if filtered_results:
            combined_results = "\n\n".join(filtered_results)
            final_user_prompt = f"Retrieved results:\n{combined_results}\n\nUser question: {user_question}"
            final_response = await generate_response(FINAL_SYSTEM_PROMPT, final_user_prompt)
            print(f"Final answer: {final_response}")
        else:
            print("No users found matching the query.")

        summary = token_tracker.get_summary()
        print(f"Token usage summary: {summary}")


if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation