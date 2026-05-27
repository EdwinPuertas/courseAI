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
5. [Modules](#modules)
   - [Module 1 — Prompt Engineering](#-module-1--prompt-engineering)
   - [Module 2 — Logic Programming](#-module-2--logic-programming--knowledge-representation)
   - [Module 3 — Applied AI & Machine Learning](#-module-3--applied-ai--machine-learning)
6. [Setup & Installation](#setup--installation)
7. [Academic Policy](#academic-policy)
8. [Bibliography](#bibliography)
9. [Contact](#contact)

---

## Course Overview

This course provides a rigorous introduction to the theoretical foundations and practical applications of **Artificial Intelligence (AI)**. Through a structured, project-driven curriculum, students explore intelligent systems, autonomous agents, probabilistic reasoning, natural language processing, and machine learning — developing the capacity to design, implement, and critically evaluate AI-driven solutions in real-world contexts.

The course follows a **theoretical-practical format**:

| Session type | Hours / week |
|---|---|
| Theoretical instruction | 1 h |
| Supervised laboratory | 2 h |
| Independent study (minimum) | **6 h** |

The professor delivers lectures to introduce each topic area and guides students through software development practices in a laboratory setting. Office hours are available for individual consultation. Periodic unannounced assessments — including reading quizzes, workshops, and project reviews — are conducted throughout the semester.

Beyond technical competence, the course aims to cultivate **creative problem-solving**, **collaborative teamwork**, **autonomous learning**, and the capacity to act as leaders and agents of change within students' professional and social environments.

---

## Learning Outcomes

Upon successful completion of this course, students will be able to:

| # | Outcome |
|---|---|
| 1 | Explain the theoretical foundations and generalities of intelligent systems |
| 2 | Identify and differentiate the principal types of intelligent agents and their architectural properties |
| 3 | Design and implement AI projects that apply agents based on **uncertainty and probabilistic reasoning** |
| 4 | Design and implement AI projects that apply agents based on **Machine Learning** techniques |

---

## Course Structure

The course progresses from foundational theory to advanced applied practice across four thematic pillars:

```
Weeks 01–02  │  Introduction to AI & Intelligent Agents
Weeks 03–04  │  Prompt Engineering & Language Model Interaction
Weeks 05–07  │  Logic Programming & Knowledge Representation (FOL)
Weeks 08–10  │  Search Algorithms & Problem Solving
Weeks 11–13  │  Machine Learning: Supervised & Unsupervised Methods
Weeks 14–16  │  Deep Learning, NLP & Computer Vision Applications
```

---

## Repository Organization

```
courseAI/
│
├── prompt_engineering/          # Module 1 – Prompt design for Large Language Models
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
├── logic_programming/           # Module 2 – First-Order Logic & Knowledge Representation
│   ├── fol_lord_of_rings.ipynb
│   ├── fol_lor_of_ring-Sympy.ipynb
│   ├── fol_universal_marvel.ipynb
│   └── virtual_appointment_FOL.ipynb
│
├── examples/                    # Module 3 – Applied AI & Machine Learning
│   ├── data/
│   ├── images/
│   ├── img/
│   ├── logic/
│   ├── dfs_bfs.ipynb
│   ├── solving_problems_search.ipynb
│   ├── backward_chaining.py
│   ├── iris_linear_regression.ipynb
│   ├── iris_logistic_regression.ipynb
│   ├── pronostico_demanda.ipynb
│   ├── water_quality_regression_MLOps.ipynb
│   ├── cooccurrence.ipynb
│   ├── n-grams_model.ipynb
│   ├── word2vec.ipynb
│   ├── text_generation.ipynb
│   ├── unstructured_data.ipynb
│   ├── poc_spacy.ipynb
│   ├── tass_lexical_baseline.ipynb
│   ├── bert.ipynb
│   ├── RoBERTa.ipynb
│   ├── electra_IA.ipynb
│   ├── HuggingFace_Transformers.ipynb
│   ├── neural_nets_with_keras.ipynb
│   └── yolov11.ipynb
│
├── requirements.txt             # Python dependencies
└── LICENSE                      # MIT License
```

---

## Modules

### 📐 Module 1 — Prompt Engineering

> *[`prompt_engineering/`](./prompt_engineering/)*

This module introduces the systematic design of prompts for large language models (LLMs). Students learn to construct effective instructions, assign contextual roles, structure complex queries, leverage chain-of-thought reasoning, and mitigate common failure modes such as hallucinations and ambiguous outputs.

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

### 🧠 Module 2 — Logic Programming & Knowledge Representation

> *[`logic_programming/`](./logic_programming/)*

This module introduces symbolic AI through **First-Order Logic (FOL)** and its application to knowledge representation and automated reasoning. Engaging domains — including Tolkien's Middle-earth and the Marvel Universe — ground abstract logical concepts in familiar, creative contexts.

| Notebook | Topic |
|---|---|
| [`fol_lord_of_rings.ipynb`](./logic_programming/fol_lord_of_rings.ipynb) | FOL Fundamentals — Tolkien Domain |
| [`fol_lor_of_ring-Sympy.ipynb`](./logic_programming/fol_lor_of_ring-Sympy.ipynb) | Symbolic Computation with SymPy |
| [`fol_universal_marvel.ipynb`](./logic_programming/fol_universal_marvel.ipynb) | Universal Quantification — Marvel Domain |
| [`virtual_appointment_FOL.ipynb`](./logic_programming/virtual_appointment_FOL.ipynb) | FOL Applied to Scheduling Systems |

---

### ⚙️ Module 3 — Applied AI & Machine Learning

> *[`examples/`](./examples/)*

This module provides hands-on implementations spanning the full AI and ML spectrum — from classical search strategies to transformer-based models and real-time computer vision — following the progression of the theoretical curriculum.

#### Search & Symbolic Reasoning

| Resource | Topic |
|---|---|
| [`dfs_bfs.ipynb`](./examples/dfs_bfs.ipynb) | Depth-First & Breadth-First Search |
| [`solving_problems_search.ipynb`](./examples/solving_problems_search.ipynb) | Heuristic Problem Solving |
| [`backward_chaining.py`](./examples/backward_chaining.py) | Backward Chaining Inference Engine |

#### Machine Learning

| Notebook | Topic |
|---|---|
| [`iris_linear_regression.ipynb`](./examples/iris_linear_regression.ipynb) | Linear Regression — Iris Dataset |
| [`iris_logistic_regression.ipynb`](./examples/iris_logistic_regression.ipynb) | Logistic Regression — Iris Dataset |
| [`pronostico_demanda.ipynb`](./examples/pronostico_demanda.ipynb) | Time-Series Demand Forecasting |
| [`water_quality_regression_MLOps.ipynb`](./examples/water_quality_regression_MLOps.ipynb) | Regression with MLOps Practices |

#### Natural Language Processing

| Notebook | Topic |
|---|---|
| [`cooccurrence.ipynb`](./examples/cooccurrence.ipynb) | Co-occurrence Matrix & Word Statistics |
| [`n-grams_model.ipynb`](./examples/n-grams_model.ipynb) | N-Gram Language Modeling |
| [`word2vec.ipynb`](./examples/word2vec.ipynb) | Word Embeddings — Word2Vec |
| [`text_generation.ipynb`](./examples/text_generation.ipynb) | Autoregressive Text Generation |
| [`unstructured_data.ipynb`](./examples/unstructured_data.ipynb) | Unstructured Data Processing |
| [`poc_spacy.ipynb`](./examples/poc_spacy.ipynb) | NLP Pipeline with spaCy |
| [`tass_lexical_baseline.ipynb`](./examples/tass_lexical_baseline.ipynb) | Sentiment Analysis — Lexical Baseline |

#### Transformer Models & Deep Learning

| Notebook | Topic |
|---|---|
| [`bert.ipynb`](./examples/bert.ipynb) | BERT — Bidirectional Encoder Representations |
| [`RoBERTa.ipynb`](./examples/RoBERTa.ipynb) | RoBERTa — Robustly Optimized BERT |
| [`electra_IA.ipynb`](./examples/electra_IA.ipynb) | ELECTRA — Efficient Pre-training |
| [`HuggingFace_Transformers.ipynb`](./examples/HuggingFace_Transformers.ipynb) | HuggingFace Transformers Pipeline |
| [`neural_nets_with_keras.ipynb`](./examples/neural_nets_with_keras.ipynb) | Neural Networks with Keras |
| [`yolov11.ipynb`](./examples/yolov11.ipynb) | Real-Time Object Detection — YOLOv11 |

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

1. **Russell, S. & Norvig, P.** (2024). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

2. **Witten, I. H., Frank, E., & Hall, M. A.** (2011). *Data Mining: Practical Machine Learning Tools and Techniques* (3rd ed.). Morgan Kaufmann / Wiley.

3. **Engelbrecht, A.** (2007). *Computational Intelligence: An Introduction* (2nd ed.). Wiley.

### Professional Standards

4. Association for Computing Machinery. (2018). *ACM Code of Ethics and Professional Conduct*. [https://www.acm.org/code-of-ethics](https://www.acm.org/code-of-ethics)

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

</div>
