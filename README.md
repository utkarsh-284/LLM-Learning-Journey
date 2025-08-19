# My AI/LLM Learning Journey for Finance & Consultancy

![The Transformer Architecture](./The_Transformer-%20model_architecture..png)
*The Transformer Architecture. This diagram, from the original (["Attention Is All You Need" paper by Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)), illustrates the key components of the Transformer model, including the Encoder and Decoder stacks, Multi-Head Attention layers, and Positional Encoding.*

This repository documents my learning journey into the world of AI and Large Language Models (LLMs), with a specific focus on applications in Finance and Consultancy. The structure of this journey is based on the "AI/LLM Learning Plan for Finance & Consultancy Roles" document.

Here, I will share my notes, projects, and implementations as I progress through the learning plan.

## Phase 1: Deep LLM Foundations (Weeks 1-4)

### Week 1: Advanced Transformer Architecture & Mathematics

**Learning Objectives:**
*   Master transformer architecture from first principles
*   Understand attention mechanisms mathematically
*   Grasp positional encodings, layer normalization, and residual connections

**Projects & Learnings:**

*   **Transformers from Scratch:** This project implements the Transformer architecture from the ground up, as detailed in the seminal paper "Attention Is All You Need". The implementation is done using TensorFlow and provides a detailed, step-by-step guide to understanding the core components of a Transformer. This notebook serves as a practice guide for the concepts covered in the [Sequence Models](https://www.coursera.org/learn/nlp-sequence-models/home/module/4) course from the DeepLearning.AI Natural Language Processing Specialization on Coursera.
    *   **Jupyter Notebook:** [Transformers-from-scratch.ipynb](/Phase_1/Transformers-from-scratch.ipynb:  )
    *   **Requirements:** The `requirements.txt` file in the `Phase 1` folder contains the necessary packages for this notebook.
    *   **Key Concepts Covered:**
        *   Positional Encodings
        *   Masking (Padding and Look-Ahead)
        *   Self-Attention (Scaled Dot Product Attention)
        *   Encoder (Encoder Layer and Full Encoder)
        *   Decoder (Decoder Layer and Full Decoder)
        *   Transformer Assembly

### Week 2: Building a Mini-Transformer for Financial Sentiment Analysis

**Learning Objectives:**
*   Implement an Encoder-only Transformer model for a classification task.
*   Apply the model to a real-world financial dataset.
*   Evaluate the model's performance and identify areas for improvement.

**Projects & Learnings:**

*   **Mini-Transformer for Financial Sentiment Analysis:** This project involves building a smaller version of the Transformer model, using only the Encoder layer from the previously developed `transformers_model.py`, to perform sentiment analysis on financial news headlines. The model is trained on the `financial_phrasebank` dataset from HuggingFace.
    *   **Jupyter Notebook:** [Mini-Transformer.ipynb](/Phase_1/Mini-Transformer.ipynb)
    *   **Key Concepts Covered:**
        *   Using a pre-built Transformer Encoder.
        *   Sentiment analysis as a classification task.
        *   Data preprocessing for financial text.
        *   Training and evaluating a Transformer-based model.
        *   Analyzing model performance and suggesting improvements.

### Week 3-4: Pre-training, Fine-tuning & Transfer Learning

**Learning Objectives:**
*   Understand different pre-training objectives (MLM, CLM, etc.)
*   Master fine-tuning strategies and when to use each
*   Learn about parameter-efficient fine-tuning (LoRA, Adapters)

**Projects & Learnings:**
*   (Add your notes and project links here)

## Phase 2: Specialized NLP for Finance (Weeks 5-8)

### Week 5-6: Financial NLP Tasks & Domain Adaptation

**Learning Objectives:**
*   Master financial text preprocessing and domain-specific challenges
*   Understand financial entity recognition and relationship extraction
*   Learn about financial document summarization and key information extraction

**Projects & Learnings:**
*   (Add your notes and project links here)

### Week 7-8: Advanced NLP Applications

**Learning Objectives:**
*   Master question-answering systems for financial documents
*   Understand retrieval-augmented generation (RAG) systems
*   Learn about conversational AI for financial advisory

**Projects & Learnings:**
*   (Add your notes and project links here)

## Phase 3: Large Language Models & Production (Weeks 9-12)

### Week 9-10: Modern LLMs & Prompt Engineering

**Learning Objectives:**
*   Understand GPT family evolution (GPT-1 to GPT-4+)
*   Master prompt engineering techniques and best practices
*   Learn about in-context learning and few-shot prompting

**Projects & Learnings:**
*   (Add your notes and project links here)

### Week 11-12: Model Deployment & MLOps for LLMs

**Learning Objectives:**
*   Learn LLM deployment strategies and optimization
*   Understand model serving, caching, and scaling
*   Master monitoring and evaluation of LLM applications

**Projects & Learnings:**
*   (Add your notes and project links here)

## Phase 4: Portfolio Development & Job Preparation (Weeks 13-16)

### Week 13-14: Capstone Project & Portfolio Enhancement

**Learning Objectives:**
*   Integrate all learned concepts into a comprehensive project
*   Create professional documentation and presentation materials
*   Optimize existing projects for maximum impact

**Projects & Learnings:**
*   (Add your notes and project links here)

### Week 15-16: Interview Preparation & Industry Connections

**Learning Objectives:**
*   Master technical interviews for AI/ML roles
*   Understand business case studies relevant to finance/consulting
*   Build industry connections and personal brand

**Projects & Learnings:**
*   (Add your notes and project links here)