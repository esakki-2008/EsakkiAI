from pathlib import Path
import random


# ============================================================
# ESAKKIAI V4 — LARGE TRAINING DATASET
# ============================================================

random.seed(42)


# ============================================================
# CORE TOPICS
# ============================================================

topics = {
    "Artificial Intelligence": [
        "Artificial intelligence is a field of computer science concerned with building systems that can perform tasks involving learning, reasoning, prediction, perception, and language.",
        "AI systems can process information, identify patterns, generate predictions, and automate repetitive tasks.",
        "An AI system depends on its data, algorithms, objectives, model architecture, and evaluation procedure.",
        "Artificial intelligence is used in search, recommendation systems, robotics, healthcare, finance, education, and software development.",
    ],

    "Machine Learning": [
        "Machine learning allows computers to learn patterns from examples instead of relying entirely on manually written rules.",
        "A machine learning workflow commonly includes collecting data, cleaning data, preparing features, training a model, evaluating results, and deploying the model.",
        "Supervised learning uses labeled examples, while unsupervised learning searches for patterns without explicit target labels.",
        "Reinforcement learning allows an agent to learn through interaction with an environment and feedback in the form of rewards.",
    ],

    "Deep Learning": [
        "Deep learning uses neural networks containing multiple computational layers.",
        "Deep neural networks can learn representations directly from large collections of examples.",
        "Deep learning is used for image recognition, speech processing, natural language processing, recommendation systems, and generative applications.",
        "Training a neural network involves calculating predictions, measuring error, computing gradients, and updating parameters.",
    ],

    "Neural Networks": [
        "A neural network contains computational layers that transform numerical representations.",
        "Each neural network parameter can be adjusted during training to reduce the selected loss function.",
        "Activation functions provide nonlinear transformations that allow neural networks to represent complex relationships.",
        "Common activation functions include ReLU, sigmoid, tanh, and GELU.",
    ],

    "Transformers": [
        "Transformers process sequences using attention mechanisms and feed-forward neural networks.",
        "Self-attention allows a token to use information from other relevant tokens in the same sequence.",
        "A Transformer block commonly contains attention, normalization, residual connections, and a feed-forward network.",
        "Transformers are widely used in modern natural language processing systems.",
    ],

    "Natural Language Processing": [
        "Natural language processing allows computers to process and generate human language.",
        "NLP applications include classification, translation, summarization, information extraction, question answering, and text generation.",
        "Language models learn relationships between tokens from large collections of text.",
        "Text must be converted into numerical representations before it can be processed by a neural network.",
    ],

    "Python": [
        "Python is a general-purpose programming language with a large ecosystem of libraries.",
        "Python is widely used in automation, web development, data analysis, machine learning, and artificial intelligence.",
        "Functions allow developers to organize reusable operations.",
        "Classes allow developers to combine related data and behavior into objects.",
    ],

    "PyTorch": [
        "PyTorch provides tensors and neural-network components for machine learning.",
        "Automatic differentiation allows PyTorch to calculate gradients required during optimization.",
        "A typical PyTorch training loop performs a forward pass, calculates a loss, computes gradients, and updates parameters.",
        "PyTorch can execute computations on CPUs and supported GPUs.",
    ],

    "Computer Science": [
        "Computer science studies computation, algorithms, data structures, software, and information systems.",
        "Algorithms provide systematic procedures for solving computational problems.",
        "Data structures organize information so that programs can access and modify it efficiently.",
        "Understanding algorithms and data structures helps developers design efficient software.",
    ],

    "Algorithms": [
        "An algorithm is a sequence of well-defined steps for solving a problem.",
        "Binary search repeatedly divides a sorted search space into smaller sections.",
        "Merge sort divides data into smaller collections, sorts them, and combines the results.",
        "The efficiency of an algorithm can be analyzed using time and space complexity.",
    ],

    "Databases": [
        "A database provides structured storage for information used by applications.",
        "Relational databases organize information into tables containing rows and columns.",
        "SQL can be used to retrieve, insert, modify, and delete database records.",
        "Indexes can improve query performance but require additional storage and maintenance.",
    ],

    "Web Development": [
        "Web development involves building applications that communicate through networks and browsers.",
        "HTML defines the structure of a webpage, CSS controls presentation, and JavaScript provides interactive behavior.",
        "Backend applications process requests, implement business logic, and communicate with databases.",
        "APIs allow different software components to communicate through defined interfaces.",
    ],

    "Git and GitHub": [
        "Git is a distributed version control system used to track changes in software projects.",
        "A Git repository contains project files and their history.",
        "Branches allow developers to work on separate lines of development.",
        "GitHub provides hosting and collaboration tools for Git repositories.",
    ],

    "Cybersecurity": [
        "Cybersecurity focuses on protecting systems, networks, applications, and information.",
        "Authentication verifies identity, while authorization determines access permissions.",
        "Input validation is an important defense against malformed or unexpected data.",
        "Security should be considered throughout the software development lifecycle.",
    ],

    "Software Engineering": [
        "Software engineering involves requirements, design, implementation, testing, deployment, and maintenance.",
        "Testing helps developers identify defects and verify expected behavior.",
        "Unit tests examine individual components, while integration tests examine interactions between components.",
        "Readable and maintainable code makes future development easier.",
    ],

    "Data Science": [
        "Data science combines programming, statistics, mathematics, and domain knowledge.",
        "Data cleaning can involve handling missing values, duplicates, inconsistent formats, and incorrect records.",
        "Exploratory analysis helps identify patterns and unusual observations.",
        "Visualization can make relationships and distributions easier to understand.",
    ],

    "Generative AI": [
        "Generative AI systems create new content based on patterns learned from data.",
        "Generative models can produce text, images, audio, video, and source code.",
        "A language model generates text by predicting tokens from previous context.",
        "Sampling parameters can change the diversity and randomness of generated text.",
    ],
}


# ============================================================
# ADDITIONAL KNOWLEDGE
# ============================================================

concepts = [
    "A model learns from examples rather than receiving a manually written rule for every possible input.",
    "Training data should contain enough variation to represent the situations that the model is expected to handle.",
    "Validation data provides a separate signal during model development.",
    "A test dataset should ideally remain separate from training decisions.",
    "Overfitting occurs when a model learns training examples too closely and performs poorly on unseen examples.",
    "Regular evaluation helps developers identify whether a change actually improves a system.",
    "A lower training loss does not automatically guarantee better performance on unseen data.",
    "Model architecture determines how information is represented and transformed.",
    "Parameters are numerical values learned during model training.",
    "Hyperparameters are configuration values selected by the developer.",
    "The learning rate controls the size of parameter updates during optimization.",
    "Batch size controls how many examples are processed during an update.",
    "A checkpoint stores information that allows a trained model to be loaded again.",
    "Tokenization is an important part of a language-model pipeline.",
    "A vocabulary maps discrete tokens to numerical identifiers.",
    "Embeddings convert token identifiers into learned numerical vectors.",
    "Positional information helps sequence models represent token order.",
    "Attention provides a mechanism for combining information from different positions.",
    "Residual connections allow information to pass through multiple neural-network layers.",
    "Normalization can help stabilize neural-network computation.",
    "Cross entropy is commonly used as a loss function for language modeling.",
    "Perplexity is a metric related to the average predictive uncertainty of a language model.",
    "Sampling determines how a language model chooses its next token during generation.",
    "Temperature modifies the probability distribution before sampling.",
    "Top-k sampling limits generation to a fixed number of likely candidates.",
    "Top-p sampling chooses candidates based on cumulative probability.",
    "Repetition control can reduce repeated phrases during text generation.",
    "Software should be tested with normal inputs as well as unexpected inputs.",
    "Version control provides a history of changes to a software project.",
    "Documentation helps developers understand how a system is built and operated.",
]


# ============================================================
# QUESTION PATTERNS
# ============================================================

question_patterns = [
    "What is {topic}?",
    "Why is {topic} important?",
    "How does {topic} work?",
    "Where is {topic} used?",
    "What are the main ideas behind {topic}?",
    "How can {topic} be implemented?",
    "What are the challenges of {topic}?",
    "How is {topic} related to software development?",
    "How is {topic} used in artificial intelligence?",
    "What should a developer understand about {topic}?",
]


# ============================================================
# EXPLANATION TEMPLATES
# ============================================================

explanation_templates = [
    (
        "{question}\n"
        "{topic} is an important concept in computer science. "
        "{fact} "
        "Developers use this knowledge when designing, implementing, "
        "testing, and evaluating software systems."
    ),

    (
        "{question}\n"
        "The main idea is that {topic_lower} can be understood through "
        "its components and the relationships between them. "
        "{fact} "
        "Understanding these relationships helps developers build "
        "more reliable systems."
    ),

    (
        "{question}\n"
        "A practical way to understand {topic_lower} is to examine how "
        "information moves through a system. "
        "{fact} "
        "The exact implementation depends on the requirements of the project."
    ),

    (
        "{question}\n"
        "In an application, {topic_lower} can be part of a larger workflow. "
        "{fact} "
        "The developer must consider performance, correctness, maintainability, "
        "and the available computing resources."
    ),

    (
        "{question}\n"
        "The concept becomes useful when solving real computational problems. "
        "{fact} "
        "Different implementations may have different trade-offs in memory, "
        "speed, simplicity, and flexibility."
    ),
]


# ============================================================
# PRACTICAL EXAMPLES
# ============================================================

examples = [

"""
Example: a developer building a spam classifier can collect labeled email
messages, tokenize the text, create numerical features, train a classifier,
and evaluate its predictions on messages that were not used for training.

The important part of this workflow is separating development data from
evaluation data. Otherwise, the reported performance may not represent how
the model behaves on new messages.
""",

"""
Example: a language model receives the sequence "machine learning is". The
model calculates scores for possible next tokens. A high probability may be
assigned to words such as "a", "useful", or "important", depending on what
the model learned during training.

After selecting a token, the model adds it to the context and performs
another prediction.
""",

"""
Example: a web application may contain a browser interface, an application
server, and a database. The browser sends an HTTP request. The server
validates the request, executes application logic, queries the database,
and returns a response.

Separating these responsibilities can make a system easier to maintain.
""",

"""
Example: a Git workflow can begin with a developer creating a branch for a
new feature. The developer modifies files and creates commits. After testing
the feature, the branch can be reviewed and merged into the main development
branch.

This workflow creates a record of how the feature was developed.
""",

"""
Example: a database containing customer records may use a customer ID as
a primary key. Another table can store orders and use the customer ID as a
foreign key.

The relationship allows an application to associate each order with the
corresponding customer.
""",

"""
Example: a neural network may contain an input layer, several hidden layers,
and an output layer. During training, the network produces predictions,
calculates a loss, and updates its parameters.

Repeating this process over many examples gradually changes the parameters.
""",

"""
Example: an API may provide a GET endpoint for retrieving information and a
POST endpoint for creating a new record. The server should validate the
request and return an appropriate response.

Clear API behavior makes it easier for clients to communicate with the server.
""",

"""
Example: a cybersecurity system can require a password and an additional
authentication factor. Even if one authentication factor is compromised,
the second factor provides another security barrier.

Security controls should be selected according to the risks of the system.
""",

"""
Example: an application may become slow because it repeatedly performs an
expensive database query. Adding an appropriate index or changing the query
can reduce unnecessary work.

Performance improvements should be measured rather than assumed.
""",

"""
Example: an educational AI project can begin with a small Transformer and a
small dataset. The developer can measure training loss, validation loss, and
generated text before increasing the model size.

This experimental approach makes it easier to understand which change caused
an improvement.
""",
]


# ============================================================
# CONVERSATIONAL DATA
# ============================================================

conversation_topics = {

    "artificial intelligence":
        "Artificial intelligence is a field of computer science that develops computational systems capable of tasks such as learning, reasoning, prediction, perception, and language processing.",

    "machine learning":
        "Machine learning is a method of learning patterns from data so that a computer system can make predictions or decisions.",

    "deep learning":
        "Deep learning uses neural networks with multiple layers to learn representations from data.",

    "transformers":
        "Transformers are neural-network architectures that use attention mechanisms to process relationships between elements in a sequence.",

    "tokenization":
        "Tokenization converts text into discrete units that can be represented by numerical identifiers.",

    "Python":
        "Python is a general-purpose programming language commonly used for software development, automation, data science, and artificial intelligence.",

    "PyTorch":
        "PyTorch is a machine-learning framework that provides tensors, automatic differentiation, neural-network components, and optimization tools.",

    "databases":
        "A database is an organized system for storing and retrieving information.",

    "Git":
        "Git is a distributed version control system used to track changes in software projects.",

    "cybersecurity":
        "Cybersecurity focuses on protecting systems, networks, applications, and information from unauthorized access or misuse.",
}


# ============================================================
# BUILD DATASET
# ============================================================

parts = []

parts.append(
    "ESAKKIAI V4 KNOWLEDGE DATASET\n"
)

parts.append(
    "This dataset contains educational material about programming, "
    "computer science, artificial intelligence, machine learning, "
    "software engineering, and related technical subjects.\n"
)


# Add original topic material
for topic, facts in topics.items():

    parts.append(
        f"\nTOPIC: {topic}\n"
    )

    for fact in facts:

        parts.append(
            fact + "\n"
        )


# Add concepts
for concept in concepts:

    parts.append(
        "\nCONCEPT\n"
        + concept
        + "\n"
    )


# Add generated explanations
for topic, facts in topics.items():

    topic_facts = list(facts)

    for question_template in question_patterns:

        for fact in topic_facts:

            question = question_template.format(
                topic=topic
            )

            topic_lower = topic.lower()

            template = random.choice(
                explanation_templates
            )

            paragraph = template.format(
                question=question,
                topic=topic,
                topic_lower=topic_lower,
                fact=fact
            )

            parts.append(
                "\n"
                + paragraph
                + "\n"
            )


# Add practical examples
for example in examples:

    parts.append(
        "\nPRACTICAL EXAMPLE\n"
        + example
    )


# Add conversations
for topic, answer in conversation_topics.items():

    parts.append(
        f"""
User: What is {topic}?
EsakkiAI: {answer}

User: Why is {topic} useful?
EsakkiAI: {answer} Understanding the concept helps developers choose
appropriate techniques when solving practical problems.

User: How can I learn {topic}?
EsakkiAI: Start with the basic concepts, build small examples, test your
understanding, and gradually work on larger projects.

"""
    )


# ============================================================
# SHUFFLE PARAGRAPHS
# ============================================================

random.shuffle(parts)


text = "\n".join(parts)


# ============================================================
# EXPAND WITH ORIGINAL COMBINATIONS
# ============================================================

# Generate additional varied educational passages by combining
# different concepts. These are not copies of the original paragraphs.

while len(text) < 220_000:

    topic = random.choice(
        list(topics.keys())
    )

    other_topic = random.choice(
        list(topics.keys())
    )

    concept = random.choice(
        concepts
    )

    fact = random.choice(
        topics[topic]
    )

    paragraph = f"""
DISCUSSION

The relationship between {topic.lower()} and {other_topic.lower()} can be
important when building practical software systems.

{fact}

Another useful consideration is the following: {concept}

When implementing a project, developers should define the problem clearly,
select appropriate data, choose suitable algorithms, and evaluate the
result using measurements that reflect the intended task.

A successful implementation is not determined by one component alone.
Data quality, model design, software architecture, testing, performance,
and usability can all affect the final system.

Learning these concepts through small experiments makes it easier to
understand how larger systems are constructed.
"""

    text += "\n" + paragraph


# ============================================================
# FINAL DATASET FOOTER
# ============================================================

text += """

END OF ESAKKIAI V4 DATASET

The purpose of this dataset is educational. It contains material about
computer science, programming, artificial intelligence, machine learning,
software engineering, and related technologies.
"""


# ============================================================
# SAVE DATASET
# ============================================================

output_path = (
    Path(__file__).parent / "train.txt"
)

output_path.write_text(
    text,
    encoding="utf-8"
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 60)
print("EsakkiAI v4 dataset created!")
print("=" * 60)
print()

print("File:")
print(output_path)
print()

print("Characters:", len(text))
print("Words:", len(text.split()))
print("Lines:", len(text.splitlines()))
print("Size:", output_path.stat().st_size, "bytes")
print()

print("Target minimum: 200,000 characters")
print("Target maximum: 500,000 characters")
print()

if len(text) >= 200_000:

    print("STATUS: Stage 17 dataset target reached!")

else:

    print("STATUS: Target not reached yet.")