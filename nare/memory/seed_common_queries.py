"""Seed memory with common queries to enable FAST route."""

COMMON_QUERIES = [
    ("привет", "Привет! Чем могу помочь?"),
    ("hello", "Hello! How can I help you?"),
    ("hi", "Hi! What can I do for you?"),
    ("help", "I can help you with code editing, questions, and exploration. What do you need?"),
    ("спасибо", "Пожалуйста!"),
    ("thanks", "You're welcome!"),
    ("что ты умеешь", "Я могу редактировать код, отвечать на вопросы о проекте, и исследовать кодовую базу."),
    ("what can you do", "I can edit code, answer questions about the project, and explore the codebase."),
]

def seed_memory(memory_system):
    """Add common queries to memory for instant FAST route."""
    from ...reasoning import llm
    import numpy as np
    
    for query, answer in COMMON_QUERIES:
        try:
            emb = llm.get_embedding(query)
            episode = {
                "query": query,
                "solution": answer,
                "reasoning_trace": "Common query seed",
                "score": 1.0,
                "metadata": {"source": "seed", "common": True}
            }
            memory_system.add_episode(episode, np.array([emb], dtype=np.float32))
        except Exception as e:
            print(f"Failed to seed '{query}': {e}")
