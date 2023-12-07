# 50.045 Information Retrieval Project
# Leveraging Large Language Models for Efficient and Accurate Retrieval of Medicinal Drug Side Effects

### Members
- Aditi Kumaresan 1005375
- Sahana Katragadda 1005761
- Swastik Majumdar 1005802
- Harikrishnan Chalapathy Anirudh 1005501


###  Please refer to the report for a more detailed breakdown of the project.

### Introduction
In the realm of healthcare, individuals often turn to online sources to gather information about the potential side effects of prescribed medications. 
However, the reliability of such information is frequently compromised as users tend to rely on the first search result provided by search engines, 
which may not always originate from credible sources. This poses a significant challenge in ensuring that individuals receive
accurate and trustworthy information regarding the side effects of their prescribed medications.

The primary objective of this project is to harness the capabilities of a Large Language Model (LLM) to
address the challenges in information retrieval related to drug side effects. The conventional approach of
relying on the first search result from search engines is prone to misinformation, and this project aims to
provide a more reliable alternative.

Therefore, we will develop an information retrieval system that leverages a Large Language Model to
respond to user queries regarding drug side effects. We will utilize the LLM’s inherent ability to understand
the contextual nuances of natural language queries, ensuring more accurate and contextually relevant responses. 
Hence, the system will not only provides answers to user queries but also supports these responses
with citations from reputable sources which enhances the credibility of the information presented.


### Dataset
The dataset is sourced from Kaggle and can be downloaded using the following [link](https://www.kaggle.com/datasets/jithinanievarghese/drugs-side-effects-and-medical-condition/).

The dataset comprises of 2931 samples and 17 features. However, for the purpose of this project, the
primary focus is on three key features: **drug_name**, **side_effects** and **drug_link**.

### Notebooks
- `LlamaIndex_RAG.ipynb`:
    - Loading dataset
    -   Upserting vector embeddings to Pinecone index.
    -   Generation using Mistral 7B and GPT-3.5 Turbo.
    -   Verification of LLM output.
    -   Generation of citations upon verification of LLM output.
    -   Retrieval-augmented generation (RAG) using Mistral 7B and GPT-3.5 Turbo.
- `LlamaIndex_Evaluate.ipynb`: 
    - RAG Evaluation
- `AdvancedRAG1-sentencewindow.ipynb`:
    - Advanced RAG: Sentence Window Retrieval + Evaluation
- `AdvancedRAG2-reranking.ipynb`:
    - Advanced RAG: Second Stage Rerankers + Evaluation
- `AdvancedRAG3-ChildParentRetrieval.ipynb`:
    - Advanced RAG: Child-Parent Recursive Retrieval + Evaluation