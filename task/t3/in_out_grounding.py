import asyncio
from typing import Any

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient


# Define Pydantic models for output parsing
class HobbyUsers(BaseModel):
    hobbies: dict[str, list[int]] = Field(description="Dictionary mapping hobby names to lists of user IDs")


# System prompts
NEE_SYSTEM_PROMPT = """You are a Named Entity Extraction (NEE) system specialized in identifying hobbies and interests from user queries.

Your task is to analyze the user's query and extract all mentioned hobbies or interests that people might have.

Instructions:
1. Identify all hobbies, activities, or interests mentioned in the query
2. For each hobby, find users from the provided context who are likely to have that interest
3. Return a JSON object where keys are hobby names and values are lists of user IDs

Context format:
Each document contains a user ID and their "about_me" text describing their interests.

Be precise and only include hobbies that are clearly mentioned or strongly implied in the query.

Response format:
{format_instructions}
"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides user information grouped by hobbies.

Based on the extracted hobby-user mappings, fetch and return the full user information for each user ID, grouped by hobby.

Ensure all user IDs exist and return valid user data.
"""


# Create clients
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    azure_deployment="text-embedding-3-small-1",
    dimensions=384
)

llm_client = AzureChatOpenAI(
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version="",
    azure_deployment="gpt-4o",
)

user_client = UserClient()


class HobbiesWizard:
    def __init__(self):
        self.vectorstore = Chroma(
            collection_name="user_hobbies",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        self.current_user_ids = set()

    async def initialize_vectorstore(self):
        """Cold start: load all users and create vectorstore"""
        print("🔄 Initializing vectorstore with all users...")
        all_users = user_client.get_all_users()

        documents = []
        for user in all_users:
            # Only embed id and about_me
            content = f"User ID: {user['id']}\nAbout me: {user.get('about_me', '')}"
            doc = Document(
                page_content=content,
                metadata={"user_id": user["id"]}
            )
            documents.append(doc)

        # Recreate vectorstore
        self.vectorstore = Chroma(
            collection_name="user_hobbies",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        await self.vectorstore.aadd_documents(documents)

        self.current_user_ids = {user["id"] for user in all_users}
        print(f"✅ Vectorstore initialized with {len(all_users)} users")

    async def update_vectorstore(self):
        """Update vectorstore with new/deleted users"""
        print("🔄 Updating vectorstore...")
        all_users = user_client.get_all_users()
        current_ids = {user["id"] for user in all_users}

        # Find deleted users
        deleted_ids = self.current_user_ids - current_ids
        if deleted_ids:
            # Delete from vectorstore
            ids_to_delete = [str(uid) for uid in deleted_ids]
            self.vectorstore.delete(ids=ids_to_delete)
            print(f"🗑️ Removed {len(deleted_ids)} users from vectorstore")

        # Find new users
        new_users = [user for user in all_users if user["id"] not in self.current_user_ids]
        if new_users:
            documents = []
            for user in new_users:
                content = f"User ID: {user['id']}\nAbout me: {user.get('about_me', '')}"
                doc = Document(
                    page_content=content,
                    metadata={"user_id": user["id"]}
                )
                documents.append(doc)
            await self.vectorstore.aadd_documents(documents)
            print(f"➕ Added {len(new_users)} new users to vectorstore")

        self.current_user_ids = current_ids

    async def search_hobbies(self, query: str, k: int = 50) -> dict[str, list[int]]:
        """Search for users by hobbies using vector similarity"""
        # Update vectorstore first
        await self.update_vectorstore()

        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(query, k=k)

        # Prepare context for NEE
        context_parts = []
        for doc in docs:
            context_parts.append(doc.page_content)
        context = "\n\n".join(context_parts)

        # Use LLM for Named Entity Extraction
        parser = PydanticOutputParser(pydantic_object=HobbyUsers)
        messages = [
            SystemMessagePromptTemplate.from_template(NEE_SYSTEM_PROMPT).format(format_instructions=parser.get_format_instructions()),
            HumanMessage(content=f"Query: {query}\n\nContext:\n{context}")
        ]
        prompt = ChatPromptTemplate.from_messages(messages).partial(format_instructions=parser.get_format_instructions())

        hobby_users: HobbyUsers = (prompt | llm_client | parser).invoke({})

        return hobby_users.hobbies

    async def get_full_user_info(self, hobby_user_map: dict[str, list[int]]) -> dict[str, list[dict[str, Any]]]:
        """Fetch full user info and verify user IDs exist"""
        result = {}

        for hobby, user_ids in hobby_user_map.items():
            valid_users = []
            for uid in user_ids:
                try:
                    user = await user_client.get_user(uid)
                    valid_users.append(user)
                except Exception:
                    print(f"⚠️ User ID {uid} not found, skipping")
            if valid_users:
                result[hobby] = valid_users

        return result


async def main():
    wizard = HobbiesWizard()
    await wizard.initialize_vectorstore()

    print("🎯 Hobbies Searching Wizard")
    print("Example: I need people who love to go to mountains")

    while True:
        user_query = input("> ").strip()
        if user_query.lower() in ['quit', 'exit']:
            break

        print("🔍 Searching for hobbies...")
        hobby_user_map = await wizard.search_hobbies(user_query)

        if hobby_user_map:
            print("📋 Fetching user information...")
            full_result = await wizard.get_full_user_info(hobby_user_map)

            import json
            print(json.dumps(full_result, indent=2))
        else:
            print("❌ No hobbies found matching the query")


if __name__ == "__main__":
    asyncio.run(main())
