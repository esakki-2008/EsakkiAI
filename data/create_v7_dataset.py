import random
from pathlib import Path


OUTPUT = Path("data/conversation_v7.txt")

random.seed(42)


qa_pairs = [

    # AI / ML
    (
        "What is artificial intelligence?",
        "Artificial intelligence is a field of computer science focused on building systems that can perform tasks that normally require human intelligence, such as understanding language, recognizing patterns, reasoning, and making decisions."
    ),
    (
        "What is machine learning?",
        "Machine learning is a branch of artificial intelligence where computers learn patterns from data and use those patterns to make predictions or decisions without being explicitly programmed for every individual case."
    ),
    (
        "What is deep learning?",
        "Deep learning is a type of machine learning that uses neural networks with multiple layers to learn complex patterns from large amounts of data."
    ),
    (
        "What is supervised learning?",
        "Supervised learning trains a model using labeled examples. The model learns a relationship between input data and the expected output."
    ),
    (
        "What is unsupervised learning?",
        "Unsupervised learning finds patterns or structures in data without using labeled output examples."
    ),
    (
        "What is reinforcement learning?",
        "Reinforcement learning trains an agent to make decisions by interacting with an environment and receiving rewards or penalties for its actions."
    ),

    # Neural networks
    (
        "What is a neural network?",
        "A neural network is a computational model made of interconnected layers that transform input data through mathematical operations and learn useful patterns by adjusting parameters during training."
    ),
    (
        "What is a neuron in a neural network?",
        "A neuron receives input values, applies learned weights and a bias, and passes the result through an activation function."
    ),
    (
        "What is an activation function?",
        "An activation function introduces nonlinearity into a neural network, allowing the model to learn complex relationships."
    ),
    (
        "What is ReLU?",
        "ReLU, or Rectified Linear Unit, is an activation function that returns zero for negative values and the input value for positive values."
    ),

    # Transformers
    (
        "What is a transformer?",
        "A transformer is a neural network architecture that uses attention mechanisms to process relationships between tokens in a sequence."
    ),
    (
        "What is self-attention?",
        "Self-attention allows each token in a sequence to examine other tokens and determine which ones are important for understanding the current token."
    ),
    (
        "What is an attention head?",
        "An attention head is an individual attention mechanism that learns relationships between tokens. Multiple heads allow a transformer to learn different relationships at the same time."
    ),
    (
        "What is a language model?",
        "A language model is a machine learning model that learns patterns in language and predicts likely tokens or words based on previous context."
    ),
    (
        "What is GPT?",
        "GPT stands for Generative Pre-trained Transformer. It is a language-model architecture based on the transformer and is designed to generate text from previous context."
    ),

    # Python
    (
        "What is Python?",
        "Python is a high-level programming language known for its readable syntax and wide use in software development, data science, automation, and artificial intelligence."
    ),
    (
        "What is a Python variable?",
        "A Python variable is a name that refers to a value or object stored during program execution."
    ),
    (
        "What is a Python function?",
        "A Python function is a reusable block of code that performs a specific task and can accept inputs and return outputs."
    ),
    (
        "What is a Python list?",
        "A Python list is an ordered and mutable collection that can store multiple values."
    ),

    # Programming
    (
        "What is a programming language?",
        "A programming language is a formal language used to write instructions that a computer can execute."
    ),
    (
        "What is an algorithm?",
        "An algorithm is a finite sequence of well-defined steps used to solve a problem or perform a computation."
    ),
    (
        "What is a data structure?",
        "A data structure is a way of organizing and storing data so that it can be accessed and modified efficiently."
    ),
    (
        "What is object-oriented programming?",
        "Object-oriented programming is a programming approach that organizes software around objects containing data and behavior."
    ),

    # Databases
    (
        "What is a database?",
        "A database is an organized collection of data that can be stored, searched, updated, and managed efficiently."
    ),
    (
        "What is SQL?",
        "SQL, or Structured Query Language, is used to create, read, update, and manage data in relational databases."
    ),
    (
        "What is a primary key?",
        "A primary key is a column or group of columns that uniquely identifies each record in a database table."
    ),

    # Git / GitHub
    (
        "What is Git?",
        "Git is a distributed version control system used to track changes in files and collaborate on software projects."
    ),
    (
        "What is GitHub?",
        "GitHub is a platform for hosting Git repositories and collaborating on software projects."
    ),
    (
        "What is a Git repository?",
        "A Git repository is a project directory containing files and the history of changes tracked by Git."
    ),

    # Software engineering
    (
        "What is debugging?",
        "Debugging is the process of finding, understanding, and fixing errors in software."
    ),
    (
        "What is an API?",
        "An API, or Application Programming Interface, is a defined way for different software components to communicate with each other."
    ),
    (
        "What is latency?",
        "Latency is the amount of time between sending a request and receiving a response from a system."
    ),
    (
        "What is caching?",
        "Caching stores frequently accessed data in a faster storage layer so that future requests can be served more quickly and with lower latency."
    ),

    # EsakkiAI
    (
        "What is EsakkiAI?",
        "EsakkiAI is an educational GPT-style language model project built from scratch using a transformer architecture, a custom tokenizer, a training pipeline, and conversational data."
    ),
    (
        "How does EsakkiAI work?",
        "EsakkiAI tokenizes text, converts tokens into embeddings, processes them through transformer layers, and predicts the next token using the learned model parameters."
    ),
    (
        "What architecture does EsakkiAI use?",
        "EsakkiAI uses a GPT-style decoder-only transformer architecture with token embeddings, positional embeddings, causal self-attention, feed-forward layers, normalization, and an output projection."
    ),
]


def create_dataset():

    examples = []

    # Repeat with controlled variation in ordering.
    for _ in range(30):

        shuffled = qa_pairs.copy()
        random.shuffle(shuffled)

        for question, answer in shuffled:

            examples.append(
                "<USER>\n"
                + question
                + "\n<ASSISTANT>\n"
                + answer
                + "\n<END>\n"
            )

    random.shuffle(examples)

    text = "".join(examples)

    OUTPUT.write_text(
        text,
        encoding="utf-8"
    )

    print("=" * 60)
    print("EsakkiAI V7 Dataset Created")
    print("=" * 60)
    print(f"Examples: {len(examples):,}")
    print(f"Characters: {len(text):,}")
    print(f"Output: {OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    create_dataset()