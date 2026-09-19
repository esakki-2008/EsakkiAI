import os
import random


OUTPUT_PATH = "data/conversation_train.txt"

TARGET_CHARS = 250_000

random.seed(42)


# ============================================================
# CONVERSATION DATA
# ============================================================

conversations = [

    (
        "What is artificial intelligence?",
        "Artificial intelligence is a field of computer science that focuses on building systems that can perform tasks that normally require human intelligence, such as reasoning, learning, understanding language, and recognizing patterns."
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
        "What is a neural network?",
        "A neural network is a computational model made of interconnected layers of mathematical operations. It learns relationships between inputs and outputs by adjusting its parameters during training."
    ),

    (
        "What is a transformer?",
        "A transformer is a neural network architecture that uses attention mechanisms to process relationships between tokens in a sequence. Transformers are widely used in language models."
    ),

    (
        "What is self attention?",
        "Self attention allows a model to examine different tokens in a sequence and determine which tokens are important to each other when producing a representation."
    ),

    (
        "What is natural language processing?",
        "Natural language processing is a field of artificial intelligence that focuses on enabling computers to process, understand, and generate human language."
    ),

    (
        "What is tokenization?",
        "Tokenization converts text into smaller units called tokens. A token may represent a character, word, subword, punctuation mark, or another piece of text."
    ),

    (
        "What is BPE?",
        "BPE stands for Byte Pair Encoding. It is a subword tokenization technique that repeatedly combines frequently occurring symbol pairs to create useful vocabulary units."
    ),

    (
        "What is a language model?",
        "A language model learns patterns in text and estimates which tokens are likely to appear given previous tokens. This allows the model to generate text one token at a time."
    ),

    (
        "How does a language model generate text?",
        "A language model receives a sequence of tokens, calculates probabilities for possible next tokens, selects a token according to its generation strategy, and repeats the process."
    ),

    (
        "What is PyTorch?",
        "PyTorch is a machine learning framework used to build and train neural networks. It provides tensors, automatic differentiation, neural network modules, and optimization tools."
    ),

    (
        "What is Python?",
        "Python is a general purpose programming language known for readable syntax and a large ecosystem of libraries. It is widely used for web development, automation, data science, and artificial intelligence."
    ),

    (
        "What is supervised learning?",
        "Supervised learning trains a model using labeled examples. The model learns a relationship between input data and known target outputs."
    ),

    (
        "What is unsupervised learning?",
        "Unsupervised learning works with data that does not have explicit target labels. The model attempts to discover useful patterns, structures, or relationships in the data."
    ),

    (
        "What is reinforcement learning?",
        "Reinforcement learning is a machine learning approach where an agent interacts with an environment and learns actions using rewards and penalties."
    ),

    (
        "What is a dataset?",
        "A dataset is a collection of examples used for analysis or machine learning. A dataset may contain text, images, numbers, labels, or other types of information."
    ),

    (
        "What is training data?",
        "Training data is the portion of a dataset used by a machine learning model to learn its parameters."
    ),

    (
        "What is validation data?",
        "Validation data is used to measure how a model performs on examples that were not directly used to update its parameters during training."
    ),

    (
        "What is overfitting?",
        "Overfitting happens when a model learns the training data too closely and performs poorly on new examples. Validation data and regularization can help detect or reduce overfitting."
    ),

    (
        "What is an epoch?",
        "An epoch is one complete pass through a training dataset. Neural networks are often trained for multiple epochs."
    ),

    (
        "What is batch size?",
        "Batch size is the number of training examples processed before the model updates its parameters. Larger batches can improve computational efficiency but require more memory."
    ),

    (
        "What is a learning rate?",
        "The learning rate controls how strongly the model parameters are changed during optimization. A learning rate that is too large can make training unstable, while one that is too small can make training slow."
    ),

    (
        "What is cross entropy loss?",
        "Cross entropy loss measures the difference between the predicted probability distribution and the expected target distribution. It is commonly used for classification and language modeling."
    ),

    (
        "What is perplexity?",
        "Perplexity is a metric commonly used to evaluate language models. It is related exponentially to average cross entropy loss, with lower values generally indicating better predictive performance on the evaluation data."
    ),

    (
        "What is an optimizer?",
        "An optimizer updates model parameters using gradients calculated during training. Adam and AdamW are examples of commonly used optimizers."
    ),

    (
        "What is AdamW?",
        "AdamW is an optimization algorithm based on Adam that separates weight decay from the gradient update. It is commonly used when training neural networks."
    ),

    (
        "What is Git?",
        "Git is a distributed version control system used to track changes in source code and collaborate on software projects."
    ),

    (
        "What is GitHub?",
        "GitHub is a platform for hosting Git repositories and collaborating on software projects. Developers can use it for code management, issues, pull requests, and project documentation."
    ),

    (
        "What is a database?",
        "A database is an organized system for storing and retrieving information. Databases can be used by applications to persist and query structured or unstructured data."
    ),

    (
        "What is cybersecurity?",
        "Cybersecurity is the practice of protecting systems, networks, applications, and data from unauthorized access, misuse, disruption, or other security threats."
    ),

    (
        "What is an API?",
        "An API is an interface that allows different software components to communicate with each other using defined operations, requests, and responses."
    ),

    (
        "What is a REST API?",
        "A REST API is a web API style that commonly uses HTTP methods such as GET, POST, PUT, and DELETE to interact with resources."
    ),

    (
        "What is software engineering?",
        "Software engineering is the systematic process of designing, developing, testing, deploying, and maintaining software systems."
    ),

    (
        "What is debugging?",
        "Debugging is the process of identifying, understanding, and fixing problems in software."
    ),

    (
        "What is an algorithm?",
        "An algorithm is a defined sequence of steps used to solve a problem or perform a computation."
    ),

    (
        "What is a data structure?",
        "A data structure is a way of organizing and storing data so that operations such as accessing, searching, inserting, and deleting can be performed efficiently."
    ),

    (
        "What is generative AI?",
        "Generative AI refers to models that can create new content such as text, images, audio, video, or code based on learned patterns."
    ),

    (
        "What is an embedding?",
        "An embedding is a numerical representation of an object such as a word, sentence, image, or document. Embeddings can capture useful relationships between objects."
    ),

    (
        "What is vector search?",
        "Vector search retrieves items by comparing numerical representations called embeddings. It is useful when similarity in meaning is more important than exact keyword matching."
    ),

    (
        "What is RAG?",
        "RAG stands for Retrieval Augmented Generation. It combines information retrieval with text generation so a language model can use relevant external information when producing an answer."
    ),

    (
        "What is caching?",
        "Caching stores frequently used data in a faster storage layer so future requests can be served more quickly. Caching can reduce latency and decrease repeated computation."
    ),

    (
        "Why is latency important in AI systems?",
        "Latency is the time between a request and the corresponding response. Lower latency can make an AI application feel more responsive, especially during interactive use."
    ),

    (
        "What is model inference?",
        "Model inference is the process of using a trained machine learning model to produce predictions or generate outputs from new input data."
    ),

    (
        "What is model training?",
        "Model training is the process of adjusting a model's parameters using data and an optimization algorithm so that the model learns useful patterns."
    ),

    (
        "What is a checkpoint?",
        "A checkpoint is a saved copy of a model's parameters and sometimes its training state. Checkpoints allow training to be resumed or a particular model version to be used later."
    ),

    (
        "What is early stopping?",
        "Early stopping is a training technique that stops optimization when validation performance stops improving for a specified period."
    ),

    (
        "What is temperature in text generation?",
        "Temperature controls the randomness of token sampling. Lower temperature generally makes generation more focused, while higher temperature allows more varied choices."
    ),

    (
        "What is top k sampling?",
        "Top k sampling restricts token selection to the k most probable candidate tokens before sampling the next token."
    ),

    (
        "What is top p sampling?",
        "Top p sampling selects a dynamic group of candidate tokens whose cumulative probability reaches a specified threshold, then samples from that group."
    ),

    (
        "What is repetition penalty?",
        "A repetition penalty modifies token probabilities to reduce the likelihood of repeatedly generating tokens that have already appeared in the context."
    ),

    (
        "What is EsakkiAI?",
        "EsakkiAI is an educational GPT-style language model project built from scratch to explore tokenization, transformers, training, evaluation, and text generation."
    ),

    (
        "How was EsakkiAI built?",
        "EsakkiAI was built using Python and PyTorch with a custom transformer architecture, a custom BPE-style tokenizer, a technical text dataset, and a CPU-based training pipeline."
    ),

    (
        "What tokenizer does EsakkiAI use?",
        "EsakkiAI uses a custom BPE-style subword tokenizer. The tokenizer learns frequent symbol combinations from the training text and uses them as vocabulary units."
    ),

    (
        "How many parameters does EsakkiAI have?",
        "The current EsakkiAI model has approximately 480 thousand trainable parameters."
    ),

    (
        "How is EsakkiAI trained?",
        "EsakkiAI is trained using next-token prediction. The model receives a sequence of tokens and learns to predict the following token using cross entropy loss."
    ),

    (
        "Why is EsakkiAI small?",
        "EsakkiAI is intentionally small so that it can be trained and experimented with on a normal laptop without requiring a large GPU or expensive cloud infrastructure."
    ),

    (
        "Can EsakkiAI replace a large language model?",
        "No. EsakkiAI is a small educational language model and does not have the scale, training data, or capabilities of large production language models."
    ),

    (
        "What is the goal of EsakkiAI?",
        "The goal of EsakkiAI is to provide a practical learning project for understanding how language models work from data preparation and tokenization through transformer training, evaluation, and generation."
    ),

    (
        "Why use a transformer for EsakkiAI?",
        "Transformers are well suited to language modeling because attention mechanisms allow the model to learn relationships between tokens in a sequence."
    ),

    (
        "What does next token prediction mean?",
        "Next token prediction means training a model to estimate the token that should appear after the previous tokens in a sequence."
    ),

    (
        "How does EsakkiAI handle text?",
        "EsakkiAI first converts text into tokens using its tokenizer. The token IDs are passed into the transformer, which produces probabilities for the next token."
    ),

    (
        "What happens during inference?",
        "During inference, the trained model receives a prompt and repeatedly predicts the next token until the requested generation length is reached."
    ),

    (
        "What is a transformer attention head?",
        "An attention head is one independent attention mechanism inside a transformer layer. Multiple heads allow the model to learn different relationships between tokens."
    ),

    (
        "What is a feed forward network?",
        "A feed forward network is a neural network component that transforms each token representation through a sequence of linear layers and a nonlinear activation function."
    ),

    (
        "What is layer normalization?",
        "Layer normalization normalizes activations within a neural network layer and can help stabilize and improve the training process."
    ),

    (
        "What is an embedding layer?",
        "An embedding layer maps token IDs to dense numerical vectors that the transformer can process."
    ),

    (
        "What is positional information?",
        "Positional information tells a transformer where tokens occur in a sequence because attention alone does not inherently provide token order."
    ),

    (
        "What is a causal language model?",
        "A causal language model predicts future tokens using only the current and previous tokens. It prevents the model from seeing future tokens during training."
    ),

    (
        "Why does training loss decrease?",
        "Training loss usually decreases when the model becomes better at predicting the target tokens in the training data. The amount of improvement depends on the model, data, optimization process, and training duration."
    ),

    (
        "Why is validation loss important?",
        "Validation loss measures model performance on data that was not directly used for parameter updates. Comparing training and validation loss can provide information about generalization."
    ),

    (
        "What is overtraining?",
        "Overtraining can occur when a model continues adapting closely to the training data while its ability to generalize to unseen data does not improve."
    ),

    (
        "How can a language model be improved?",
        "A language model can be improved through better training data, tokenizer design, model architecture, optimization, regularization, evaluation, and generation strategies."
    ),

    (
        "What is batch training?",
        "Batch training processes multiple sequences together before an optimization update. This can improve computational efficiency compared with processing one sequence at a time."
    ),

    (
        "What is gradient clipping?",
        "Gradient clipping limits the magnitude of gradients during training. It can help prevent unstable parameter updates caused by unusually large gradients."
    ),

    (
        "What is a learning rate scheduler?",
        "A learning rate scheduler changes the learning rate during training according to a predefined strategy. This can help optimization progress from larger updates toward smaller updates."
    ),

    (
        "What is CPU training?",
        "CPU training uses the computer's central processing unit to perform model calculations. It is slower than suitable GPU training for large models but can be practical for small educational models."
    ),

    (
        "Why was EsakkiAI trained on a CPU?",
        "EsakkiAI was trained on a CPU because its small architecture allows experimentation without requiring a dedicated high-memory GPU."
    ),

    (
        "What is a model parameter?",
        "A model parameter is a learned numerical value inside a neural network. During training, optimization updates these values to reduce the model's loss."
    ),

    (
        "What is a vocabulary?",
        "A vocabulary is the collection of token units that a tokenizer can represent. Each token is associated with an integer ID used by the model."
    ),

    (
        "What is subword tokenization?",
        "Subword tokenization represents text using units that can be smaller or larger than complete words. It can reduce unknown-word problems while keeping vocabulary size manageable."
    ),

    (
        "Why use BPE instead of character tokenization?",
        "BPE can represent frequent groups of characters as single tokens, allowing the model to process common words and word parts more efficiently than pure character tokenization."
    ),

    (
        "What is a checkpoint file?",
        "A checkpoint file stores model information so a trained model can be loaded later without training it again."
    ),

    (
        "What is model evaluation?",
        "Model evaluation is the process of measuring how well a trained model performs using metrics and test examples that provide information about its behavior."
    ),

    (
        "What is a test prompt?",
        "A test prompt is an input selected to examine how a language model responds to a particular type of question or instruction."
    ),

    (
        "What makes a good training dataset?",
        "A good training dataset should contain relevant, diverse, consistent, and sufficiently large examples for the behavior the model is expected to learn."
    ),

    (
        "Why does a small language model make mistakes?",
        "A small language model has limited parameters and usually learns from less data than large models. These limitations can cause incomplete knowledge, repetition, malformed text, and weaker contextual consistency."
    ),

    (
        "What is a token ID?",
        "A token ID is an integer that identifies a token in a tokenizer vocabulary. Neural networks operate on these numerical IDs rather than raw text."
    ),

    (
        "What is softmax?",
        "Softmax converts a collection of numerical logits into a probability distribution whose values sum to one."
    ),

    (
        "What is a logit?",
        "A logit is an unnormalized numerical score produced by a model before values are converted into probabilities."
    ),

    (
        "What is sampling?",
        "Sampling selects a token from a probability distribution rather than always choosing the single highest probability token."
    ),

    (
        "What is greedy decoding?",
        "Greedy decoding selects the token with the highest probability at every generation step. It is simple but can produce repetitive or less varied text."
    ),

    (
        "What is context length?",
        "Context length is the maximum number of tokens a model can process as its current input context."
    ),

    (
        "What is a training step?",
        "A training step usually consists of processing a batch, calculating the loss, computing gradients, and updating the model parameters."
    ),

    (
        "What is an epoch in language model training?",
        "An epoch represents one complete pass over the available training examples. In random sequence sampling, training is often described using optimization steps instead of strict epochs."
    ),

    (
        "What is AI model inference latency?",
        "Inference latency is the time required for an AI model to produce an output after receiving an input. It can depend on model size, hardware, sequence length, and generation settings."
    ),

    (
        "How does caching improve AI applications?",
        "Caching can store previously calculated or frequently requested results. When the same information is requested again, the application may return the cached result instead of repeating expensive computation."
    ),

    (
        "What is software testing?",
        "Software testing evaluates whether a program behaves according to its expected requirements and helps identify defects before or after deployment."
    ),

    (
        "What is deployment?",
        "Deployment is the process of making a software application or model available in an environment where users or other systems can access it."
    ),

    (
        "What is version control?",
        "Version control records changes to files over time so developers can review history, collaborate, and restore previous versions when necessary."
    ),

    (
        "What is open source software?",
        "Open source software is software whose source code is made available under a license that permits specified forms of use, modification, and redistribution."
    ),

    (
        "What is an AI application?",
        "An AI application is software that uses artificial intelligence techniques to perform tasks such as prediction, classification, recommendation, generation, or analysis."
    ),

    (
        "How can AI help developers?",
        "AI can assist developers with tasks such as code generation, debugging, documentation, testing, analysis, and natural language interfaces."
    ),

    (
        "What is machine learning evaluation?",
        "Machine learning evaluation measures how well a trained model performs on data or tasks that represent its intended use."
    ),

    (
        "Why separate training and validation data?",
        "Separating training and validation data provides a way to measure performance on examples that were not directly used to update model parameters."
    ),

    (
        "What is a hyperparameter?",
        "A hyperparameter is a configuration value selected before or during training rather than learned directly as a model parameter. Examples include learning rate, batch size, and model depth."
    ),

    (
        "What is model architecture?",
        "Model architecture describes the structure and components of a neural network, including its layers, dimensions, attention mechanisms, and connections."
    ),

    (
        "What is an attention mask?",
        "An attention mask controls which positions a transformer is allowed to attend to. A causal mask prevents a token from attending to future tokens."
    ),

    (
        "What is autoregressive generation?",
        "Autoregressive generation produces one token at a time, using previously generated tokens as context for predicting the next token."
    ),

    (
        "What is a language model prompt?",
        "A prompt is the input text provided to a language model to establish context or request a generated response."
    ),

    (
        "What is prompt generation?",
        "Prompt generation refers to producing or constructing input text that provides a model with the context needed for a desired response."
    ),

    (
        "What is model compression?",
        "Model compression reduces the computational or memory requirements of a model using techniques such as pruning, quantization, or knowledge distillation."
    ),

    (
        "What is quantization?",
        "Quantization represents model values using lower precision numerical formats. It can reduce memory usage and sometimes improve inference efficiency."
    ),

    (
        "What is knowledge distillation?",
        "Knowledge distillation trains a smaller student model using information produced by a larger teacher model."
    ),

    (
        "What is transfer learning?",
        "Transfer learning uses knowledge learned from one task or dataset as a starting point for another related task."
    ),

    (
        "What is fine tuning?",
        "Fine tuning continues training a pretrained model on a more specific dataset so that its behavior becomes better suited to a particular task or domain."
    ),

    (
        "What is pretraining?",
        "Pretraining is the initial training stage where a model learns general patterns from a large dataset before it is adapted to specific tasks."
    ),

    (
        "What is an AI pipeline?",
        "An AI pipeline is a sequence of stages used to prepare data, process inputs, run a model, evaluate outputs, and deliver results to an application."
    ),

    (
        "What is data preprocessing?",
        "Data preprocessing transforms raw data into a form suitable for analysis or model training. It can include cleaning, normalization, tokenization, and filtering."
    ),

    (
        "What is data augmentation?",
        "Data augmentation creates additional training examples by applying controlled transformations to existing data. It can improve diversity in some machine learning tasks."
    ),

    (
        "What is model generalization?",
        "Generalization is the ability of a trained model to perform effectively on new examples that were not directly used during training."
    ),

    (
        "What is an AI benchmark?",
        "An AI benchmark is a standardized evaluation setup used to measure and compare model performance on defined tasks or datasets."
    ),

    (
        "What is reproducibility in machine learning?",
        "Reproducibility means being able to repeat an experiment and obtain comparable results by controlling factors such as code, data, configuration, and random seeds."
    ),

    (
        "How should an AI project be documented?",
        "An AI project should document its purpose, architecture, dataset, tokenizer, model configuration, training process, evaluation results, limitations, and instructions for running it."
    )

]


# ============================================================
# GENERATE DATASET
# ============================================================

print()
print("Creating conversational dataset...")


lines = []

total_chars = 0

index = 0


while total_chars < TARGET_CHARS:

    question, answer = conversations[
        index % len(conversations)
    ]

    block = (
        f"User: {question}\n"
        f"EsakkiAI: {answer}\n\n"
    )

    lines.append(block)

    total_chars += len(block)

    index += 1


# ============================================================
# WRITE FILE
# ============================================================

os.makedirs(
    "data",
    exist_ok=True
)


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "".join(lines)
    )


# ============================================================
# REPORT
# ============================================================

final_text = "".join(lines)

print()
print("=" * 60)

print(
    "Conversational dataset created!"
)

print(
    "Characters:",
    f"{len(final_text):,}"
)

print(
    "Conversation examples:",
    index
)

print(
    "Output:",
    OUTPUT_PATH
)

print("=" * 60)