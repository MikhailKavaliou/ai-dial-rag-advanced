from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


#TODO:
# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """You are a helpful RAG-powered assistant specialized in microwave usage and operation.

Your responses should be based on the provided RAG Context and conversation history. The user messages will be structured with:
1. RAG Context - relevant information retrieved from the microwave manual
2. User Question - the actual question being asked

Instructions:
- Use the RAG Context provided to answer the User Question accurately
- Only answer questions related to microwave usage, operation, safety, and maintenance
- If the question is not related to microwave usage or cannot be answered from the context, politely decline
- Do not answer questions that are outside the scope of the provided context or conversation history
- Be concise and helpful in your responses
- If information is not available in the context, clearly state that you don't have that information
"""

#TODO:
# Provide structured system prompt, with RAG Context and User Question sections.
USER_PROMPT = """RAG Context:
{context}

User Question:
{question}
"""


#TODO:
# - create embeddings client with 'text-embedding-3-small-1' model
# - create chat completion client
# - create text processor, DB config: {'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'}
# ---
# Create method that will run console chat with such steps:
# - get user input from console
# - retrieve context
# - perform augmentation
# - perform generation
# - it should run in `while` loop (since it is console chat)


def main():
    """Main function to run the RAG-powered console chat"""
    print("🤖 Initializing RAG-powered Microwave Assistant...\n")

    # Create embeddings client with 'text-embedding-3-small-1' model
    embeddings_client = DialEmbeddingsClient(
        deployment_name='text-embedding-3-small-1',
        api_key=API_KEY
    )

    # Create chat completion client
    chat_client = DialChatCompletionClient(
        deployment_name='gpt-4o',
        api_key=API_KEY
    )

    # Create text processor with DB config
    db_config = {
        'host': 'localhost',
        'port': 5433,
        'database': 'vectordb',
        'user': 'postgres',
        'password': 'postgres'
    }
    text_processor = TextProcessor(embeddings_client, db_config)

    # Process the microwave manual (only on first run or if needed)
    import os
    manual_path = os.path.join(os.path.dirname(__file__), 'embeddings', 'microwave_manual.txt')

    # Ask if user wants to reprocess the manual
    print("Do you want to process/reprocess the microwave manual? (y/n): ", end='')
    response = input().strip().lower()
    if response == 'y':
        print("\n📚 Processing microwave manual...")
        text_processor.process_text_file(
            file_path=manual_path,
            chunk_size=300,
            overlap=40,
            dimensions=1536,
            truncate=True
        )
        print("✅ Manual processed and stored in database\n")

    # Initialize conversation
    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))

    print("=" * 80)
    print("🎤 Microwave Assistant Chat - Ask me anything about microwave usage!")
    print("Type 'exit', 'quit', or 'q' to end the conversation")
    print("=" * 80)
    print()

    # Main chat loop
    while True:
        # Get user input
        user_question = input("You: ").strip()

        # Check for exit commands
        if user_question.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Thank you for using the Microwave Assistant! Goodbye!")
            break

        if not user_question:
            continue

        # Retrieve context using semantic search
        print("🔍 Retrieving relevant context...", end='', flush=True)
        relevant_chunks = text_processor.search(
            query=user_question,
            search_mode=SearchMode.COSINE_DISTANCE,
            top_k=3,
            min_score=0.3,
            dimensions=1536
        )
        print(" Done!")

        # Perform augmentation - combine context with user question
        context_text = "\n\n".join(relevant_chunks) if relevant_chunks else "No relevant context found."
        augmented_prompt = USER_PROMPT.format(
            context=context_text,
            question=user_question
        )

        # Add user message to conversation
        user_message = Message(Role.USER, augmented_prompt)
        conversation.add_message(user_message)

        # Perform generation - get LLM response
        print("🤔 Thinking...", end='', flush=True)
        try:
            ai_response = chat_client.get_completion(
                messages=conversation.get_messages(),
                temperature=0.7,
                max_tokens=500
            )
            print(" Done!\n")

            # Add AI response to conversation
            conversation.add_message(ai_response)

            # Display response
            print(f"Assistant: {ai_response.content}\n")
            print("-" * 80)
            print()

        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            # Remove the failed user message from conversation
            conversation.messages.pop()


if __name__ == "__main__":
    main()
