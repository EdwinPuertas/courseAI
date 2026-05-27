<div align="center">

# Artificial Intelligence
### Faculty of Engineering — Computer Science Program

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter&logoColor=white)
![License](https://img.shields.io/github/license/EdwinPuertas/courseAI)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![ACM Ethics](https://img.shields.io/badge/Ethics-ACM%20Code-red)

**Professor:** Edwin Puertas, Ph.D. · `epuerta@utb.edu.co`
**Institution:** Universidad Tecnológica de Bolívar
**Repository:** [github.com/EdwinPuertas/courseAI](https://github.com/EdwinPuertas/courseAI)

</div>

---

## Table of Contents

1. [Course Overview](#course-overview)
2. [Learning Outcomes](#learning-outcomes)
3. [Course Structure](#course-structure)
4. [Repository Organization](#repository-organization)
5. [Units](#units)
   - [Unit 1 — Artificial Intelligence](#-unit-1--artificial-intelligence)
   - [Unit 2 — Problem-Solving](#-unit-2--problem-solving)
   - [Unit 3 — Knowledge, Reasoning, and Planning](#-unit-3--knowledge-reasoning-and-planning)
   - [Unit 4 — Uncertain Knowledge and Reasoning](#-unit-4--uncertain-knowledge-and-reasoning)
   - [Unit 5 — Machine Learning](#-unit-5--machine-learning)
   - [Unit 6 — Final Project](#-unit-6--final-project)
   - [Supplementary — Prompt Engineering](#-supplementary--prompt-engineering)
6. [Setup & Installation](#setup--installation)
7. [Academic Policy](#academic-policy)
8. [Bibliography](#bibliography)
9. [Contact](#contact)

---

## Course Overview

This course provides a rigorous introduction to the theoretical foundations and practical applications of **Artificial Intelligence (AI)**. Through a structured, project-driven curriculum, students explore intelligent systems, autonomous agents, probabilistic reasoning, symbolic logic, and machine learning — developing the capacity to design, implement, and critically evaluate AI-driven solutions in real-world contexts.

The course follows a **theoretical-practical format**:

| Session type | Hours / week |
|---|---|
| Theoretical instruction | 1 h |
| Supervised laboratory | 2 h |
| Independent study (minimum) | **6 h** |

The professor delivers lectures to introduce each topic area and guides students through software development practices in a laboratory setting. Office hours are available for individual consultation. Periodic unannounced assessments — including reading quizzes, workshops, and project reviews — are conducted throughout the semester.

Beyond technical competence, the course aims to cultivate **creative problem-solving**, **collaborative teamwork**, **autonomous learning**, and the capacity to act as leaders and agents of change within students' professional and social environments.

> **Primary Reference:** This course follows the structure of **Russell, S. & Norvig, P.** (2022). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson — the definitive textbook in the field, whose six-part organization directly shapes the six units of this course.

---

## Learning Outcomes

Upon successful completion of this course, students will be able to:

| # | Outcome |
|---|---|
| 1 | Explain the theoretical foundations and generalities of intelligent systems and agent architectures |
| 2 | Identify and differentiate the principal types of intelligent agents and their architectural properties |
| 3 | Design and implement AI projects applying agents based on **uncertainty and probabilistic reasoning** |
| 4 | Design and implement AI projects applying agents based on **Machine Learning** techniques |

---

## Course Structure

The curriculum is organized into six units that follow the logical progression of Russell & Norvig's *Artificial Intelligence: A Modern Approach* (4th ed.), moving from foundational theory to applied machine learning:

```
Unit 1  │  Artificial Intelligence
         │  History · Agent types · Rationality · PEAS framework
         │  ─── R&N Part I: Foundations (Ch. 1–2)

Unit 2  │  Problem-Solving
         │  Uninformed search · Informed search · Heuristics · Adversarial search
         │  ─── R&N Part II: Problem-Solving (Ch. 3–5)

Unit 3  │  Knowledge, Reasoning, and Planning
         │  Propositional logic · First-order logic · Planning · Prompt engineering
         │  ─── R&N Part III: Knowledge (Ch. 7–11)

Unit 4  │  Uncertain Knowledge and Reasoning
         │  Probability · Bayes · Naïve Bayes · Probabilistic language models
         │  ─── R&N Part IV: Uncertainty (Ch. 12–16)

Unit 5  │  Machine Learning
         │  Supervised learning · Neural networks · NLP · Deep learning · Vision
         │  ─── R&N Part V: Learning (Ch. 19–21)

Unit 6  │  Final Project
         │  End-to-end AI system design, evaluation, and presentation
```

---

## Repository Organization

```
courseAI/
│
├── examples/                    # Core AI implementations — Units 2–5
│   ├── data/
│   ├── images/
│   ├── img/
│   ├── logic/
│   │
│   │  ── Unit 2: Problem-Solving ──────────────────────────────────
│   ├── dfs_bfs.ipynb
│   ├── solving_problems_search.ipynb
│   │
│   │  ── Unit 3: Knowledge, Reasoning & Planning ──────────────────
│   ├── backward_chaining.py
│   │
│   │  ── Unit 4: Uncertain Knowledge & Reasoning ──────────────────
│   ├── pronostico_demanda.ipynb
│   ├── cooccurrence.ipynb
│   ├── n-grams_model.ipynb
│   ├── tass_lexical_baseline.ipynb
│   ├── unstructured_data.ipynb
│   │
│   │  ── Unit 5: Machine Learning ──────────────────────────────────
│   ├── iris_linear_regression.ipynb
│   ├── iris_logistic_regression.ipynb
│   ├── water_quality_regression_MLOps.ipynb
│   ├── word2vec.ipynb
│   ├── poc_spacy.ipynb
│   ├── text_generation.ipynb
│   ├── neural_nets_with_keras.ipynb
│   ├── bert.ipynb
│   ├── RoBERTa.ipynb
│   ├── electra_IA.ipynb
│   ├── HuggingFace_Transformers.ipynb
│   └── yolov11.ipynb
│
├── logic_programming/           # Unit 3 – First-Order Logic & Knowledge Representation
│   ├── fol_lord_of_rings.ipynb
│   ├── fol_lor_of_ring-Sympy.ipynb
│   ├── fol_universal_marvel.ipynb
│   └── virtual_appointment_FOL.ipynb
│
├── prompt_engineering/          # Supplementary – Prompt design for LLMs
│   ├── 00_intro.ipynb
│   ├── 01_prompt_structure.ipynb
│   ├── 02_clear_and_direct.ipynb
│   ├── 03_assigning_roles.ipynb
│   ├── 04_separating_data_instructions.ipynb
│   ├── 05_formatting_output_&_speaking_claude.ipynb
│   ├── 06_precognition_thinking.ipynb
│   ├── 07_fewshot_prompting.ipynb
│   ├── 08_avoiding_hallucinations.ipynb
│   └── 09_Complex_Prompts_from_Scratch.ipynb
│
├── requirements.txt
└── LICENSE
```

---

## Units

### 🤖 Unit 1 — Artificial Intelligence

> *R&N Part I: Foundations — Chapters 1–2*

**Objectives**
- Understand the history, scope, and philosophical foundations of AI
- Define rationality and the concept of the rational agent
- Characterize agents using the **PEAS framework** (Performance, Environment, Actuators, Sensors)
- Classify environments and identify appropriate agent architectures for each type

**Key Concepts:** Turing Test · Weak and Strong AI · Simple reflex agents · Model-based agents · Goal-based agents · Utility-based agents · Learning agents

| Resource | Description |
|---|---|
| R&N Ch. 1 | *Introduction* — What is AI, history, and current state |
| R&N Ch. 2 | *Intelligent Agents* — Agents, environments, and rationality |

---

### 🔍 Unit 2 — Problem-Solving

> *R&N Part II: Problem-Solving — Chapters 3–5*
> *[`examples/`](./examples/)*

**Objectives**
- Formulate real-world problems as state-space search problems
- Implement and compare uninformed search strategies (BFS, DFS, UCS)
- Apply informed search algorithms using heuristics (A\*, greedy best-first)
- Analyze completeness, optimality, time, and space complexity of search algorithms

**Key Concepts:** State space · Search tree · Frontier · Uninformed vs. informed search · Admissible heuristics · A\* · Adversarial search

| Notebook / File | Topic |
|---|---|
| [`dfs_bfs.ipynb`](./examples/dfs_bfs.ipynb) | Depth-First Search & Breadth-First Search |
| [`solving_problems_search.ipynb`](./examples/solving_problems_search.ipynb) | Heuristic & Informed Problem Solving |

---

### 🧠 Unit 3 — Knowledge, Reasoning, and Planning

> *R&N Part III: Knowledge — Chapters 7–11*
> *[`logic_programming/`](./logic_programming/) · [`examples/`](./examples/)*

**Objectives**
- Represent knowledge using propositional and first-order logic
- Apply inference rules: modus ponens, forward chaining, and backward chaining
- Formulate planning problems using logical agents
- Use modern LLMs as applied knowledge representation and reasoning tools

**Key Concepts:** Propositional logic · FOL · Unification · Resolution · Backward/Forward chaining · STRIPS planning · Prompt engineering as applied reasoning

#### First-Order Logic (FOL)

| Notebook | Topic |
|---|---|
| [`fol_lord_of_rings.ipynb`](./logic_programming/fol_lord_of_rings.ipynb) | FOL Fundamentals — Tolkien Domain |
| [`fol_lor_of_ring-Sympy.ipynb`](./logic_programming/fol_lor_of_ring-Sympy.ipynb) | Symbolic FOL Computation with SymPy |
| [`fol_universal_marvel.ipynb`](./logic_programming/fol_universal_marvel.ipynb) | Universal Quantification — Marvel Domain |
| [`virtual_appointment_FOL.ipynb`](./logic_programming/virtual_appointment_FOL.ipynb) | FOL Applied to Scheduling & Planning |

#### Automated Reasoning

| Resource | Topic |
|---|---|
| [`backward_chaining.py`](./examples/backward_chaining.py) | Backward Chaining Inference Engine |

---

### 🎲 Unit 4 — Uncertain Knowledge and Reasoning

> *R&N Part IV: Uncertainty — Chapters 12–16*
> *[`examples/`](./examples/)*

**Objectives**
- Understand probability theory as a foundation for reasoning under uncertainty
- Apply Bayesian inference and Naïve Bayes classifiers to real-world data
- Model uncertainty in language using statistical and probabilistic approaches
- Evaluate probabilistic models using standard metrics

**Key Concepts:** Prior/posterior probability · Bayes' theorem · Naïve Bayes · Conditional independence · N-gram language models · Probabilistic text classification

| Notebook | Topic |
|---|---|
| [`cooccurrence.ipynb`](./examples/cooccurrence.ipynb) | Co-occurrence Statistics & Word Probability |
| [`n-grams_model.ipynb`](./examples/n-grams_model.ipynb) | N-Gram Probabilistic Language Modeling |
| [`tass_lexical_baseline.ipynb`](./examples/tass_lexical_baseline.ipynb) | Probabilistic Sentiment Analysis — Lexical Baseline |
| [`unstructured_data.ipynb`](./examples/unstructured_data.ipynb) | Probabilistic Processing of Unstructured Data |
| [`pronostico_demanda.ipynb`](./examples/pronostico_demanda.ipynb) | Forecasting Under Uncertainty — Demand Prediction |

---

### ⚙️ Unit 5 — Machine Learning

> *R&N Part V: Learning — Chapters 19–21*
> *[`examples/`](./examples/)*

**Objectives**
- Implement and evaluate supervised learning models (regression, classification)
- Apply deep learning architectures to NLP and computer vision tasks
- Fine-tune pre-trained transformer models for downstream tasks
- Follow MLOps practices for reproducible and scalable model development

**Key Concepts:** Supervised/unsupervised learning · Bias-variance tradeoff · Gradient descent · Word embeddings · Transformers · Transfer learning · MLOps

#### Supervised Learning & Regression

| Notebook | Topic |
|---|---|
| [`iris_linear_regression.ipynb`](./examples/iris_linear_regression.ipynb) | Linear Regression — Iris Dataset |
| [`iris_logistic_regression.ipynb`](./examples/iris_logistic_regression.ipynb) | Logistic Regression — Classification |
| [`water_quality_regression_MLOps.ipynb`](./examples/water_quality_regression_MLOps.ipynb) | Regression with MLOps Practices |

#### Natural Language Processing & Embeddings

| Notebook | Topic |
|---|---|
| [`word2vec.ipynb`](./examples/word2vec.ipynb) | Word Embeddings — Word2Vec |
| [`poc_spacy.ipynb`](./examples/poc_spacy.ipynb) | NLP Pipeline with spaCy |
| [`text_generation.ipynb`](./examples/text_generation.ipynb) | Autoregressive Text Generation |

#### Deep Learning & Transformer Models

| Notebook | Topic |
|---|---|
| [`neural_nets_with_keras.ipynb`](./examples/neural_nets_with_keras.ipynb) | Neural Networks with Keras |
| [`bert.ipynb`](./examples/bert.ipynb) | BERT — Bidirectional Encoder Representations |
| [`RoBERTa.ipynb`](./examples/RoBERTa.ipynb) | RoBERTa — Robustly Optimized BERT |
| [`electra_IA.ipynb`](./examples/electra_IA.ipynb) | ELECTRA — Efficient Pre-training |
| [`HuggingFace_Transformers.ipynb`](./examples/HuggingFace_Transformers.ipynb) | HuggingFace Transformers Pipeline |
| [`yolov11.ipynb`](./examples/yolov11.ipynb) | Real-Time Object Detection — YOLOv11 |

---

### 🎓 Unit 6 — Final Project

Students design, implement, and evaluate an **end-to-end AI system** addressing a real-world problem. Projects must integrate concepts from at least two course units and demonstrate technical rigor, innovation, and ethical awareness.

**Deliverables**

| Deliverable | Description |
|---|---|
| Technical report | IEEE-format document describing problem, methodology, results, and conclusions |
| Codebase | Reproducible Jupyter Notebook + public GitHub repository |
| Presentation | Oral defense with live system demonstration |

**Evaluation Criteria:** Problem relevance · Methodological soundness · Code quality & reproducibility · Result analysis · Communication clarity · Ethical considerations

---

### 📐 Supplementary — Prompt Engineering

> *[`prompt_engineering/`](./prompt_engineering/)*

A structured sequence on systematic prompt design for large language models — bridging Unit 3 (knowledge and reasoning) with Unit 5 (machine learning) by treating LLMs as applied intelligent systems.

| Notebook | Topic |
|---|---|
| [`00_intro.ipynb`](./prompt_engineering/00_intro.ipynb) | Introduction to Prompt Engineering |
| [`01_prompt_structure.ipynb`](./prompt_engineering/01_prompt_structure.ipynb) | Structural Design of Prompts |
| [`02_clear_and_direct.ipynb`](./prompt_engineering/02_clear_and_direct.ipynb) | Clarity and Directness |
| [`03_assigning_roles.ipynb`](./prompt_engineering/03_assigning_roles.ipynb) | Role Assignment and Persona Design |
| [`04_separating_data_instructions.ipynb`](./prompt_engineering/04_separating_data_instructions.ipynb) | Separating Data from Instructions |
| [`05_formatting_output_&_speaking_claude.ipynb`](./prompt_engineering/05_formatting_output_&_speaking_claude.ipynb) | Output Formatting Strategies |
| [`06_precognition_thinking.ipynb`](./prompt_engineering/06_precognition_thinking.ipynb) | Extended Reasoning & Chain-of-Thought |
| [`07_fewshot_prompting.ipynb`](./prompt_engineering/07_fewshot_prompting.ipynb) | Few-Shot Prompting |
| [`08_avoiding_hallucinations.ipynb`](./prompt_engineering/08_avoiding_hallucinations.ipynb) | Hallucination Mitigation Techniques |
| [`09_Complex_Prompts_from_Scratch.ipynb`](./prompt_engineering/09_Complex_Prompts_from_Scratch.ipynb) | Building Complex Prompt Pipelines |

---

## Setup & Installation

### Prerequisites

- **Python** 3.8 or higher
- **pip** or **conda** package manager
- **Jupyter Notebook** or **JupyterLab**

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/EdwinPuertas/courseAI.git
cd courseAI

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

### Dependency Overview

| Category | Packages |
|---|---|
| Data Science & ML | `numpy`, `scipy`, `pandas`, `scikit-learn`, `imbalanced-learn` |
| Deep Learning | `transformers`, `ultralytics` |
| NLP | `nltk`, `spacy`, `gensim` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Data Utilities | `datasets`, `missingno`, `tqdm` |
| Web & Scraping | `requests`, `beautifulsoup4` |
| AI APIs | `anthropic` |

---

## Academic Policy

### Attendance & Participation

Punctual attendance is mandatory for all sessions. Students are expected to complete assigned readings and workshop submissions on schedule. The professor will conduct **periodic, unannounced assessments** — including reading quizzes, workshop reviews, and project evaluations — throughout the semester.

### Independent Study Requirement

The success of this course depends on students reinforcing in-class concepts through self-directed practice. A minimum of **six (6) hours per week** of independent study is required, consistent with the standard of two study hours per contact hour.

### Academic Integrity

All work submitted must be the student's original contribution. Any use of external sources, libraries, or generated code must be explicitly acknowledged. This course adheres to the **ACM Code of Ethics and Professional Conduct**:

> [https://www.acm.org/code-of-ethics](https://www.acm.org/code-of-ethics)

Violations of academic integrity will be addressed in accordance with institutional regulations and may result in course failure or formal disciplinary proceedings.

---

## Bibliography

### Core Textbooks

1. **Russell, S. & Norvig, P.** (2022). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
   *(Primary reference — course unit structure follows Part I–V of this text)*

2. **Witten, I. H., Frank, E., & Hall, M. A.** (2011). *Data Mining: Practical Machine Learning Tools and Techniques* (3rd ed.). Morgan Kaufmann / Wiley.

3. **Engelbrecht, A.** (2007). *Computational Intelligence: An Introduction* (2nd ed.). Wiley.

### Professional Standards

4. Association for Computing Machinery. (2018). *ACM Code of Ethics and Professional Conduct*.
   [https://www.acm.org/code-of-ethics](https://www.acm.org/code-of-ethics)

---

## Contact

| | |
|---|---|
| **Professor** | Edwin Puertas, Ph.D. |
| **Email** | epuerta@utb.edu.co |
| **Institution** | Universidad Tecnológica de Bolívar |
| **Office Hours** | Available by appointment — contact via institutional channels |

---

<div align="center">

*This repository is maintained for educational purposes. All materials are intended solely for enrolled students of the Artificial Intelligence course.*

[![GitHub](https://img.shields.io/badge/GitHub-EdwinPuertas-181717?logo=github&logoColor=white)](https://github.com/EdwinPuertas)
[![UTB](https://img.shields.io/badge/Universidad-Tecnológica%20de%20Bolívar-003DA5)](https://www.utb.edu.co)
[![R&N](https://img.shields.io/badge/Reference-Russell%20%26%20Norvig%204th%20ed.-6A0DAD)](https://aima.cs.berkeley.edu/)

</div>
