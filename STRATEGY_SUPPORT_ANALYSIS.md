# Ragas Evaluation Harness - Unified Evaluation Workflow Strategy Support Analysis

**Analysis Date:** December 15, 2024  
**Ragas Version:** Based on main branch documentation  
**Analyst:** GitHub Copilot

**Classification Framework:**
- **✅ Natively Supported**: 15 strategies (44%) - Available immediately after `pip install ragas`
- **🔌 Supported via Integration**: 6 strategies (18%) - Requires external packages but has documented integration patterns
- **⚠️ Partially Supported**: 2 strategies (6%) - Native but with significant limitations
- **❌ Not Supported**: 11 strategies (32%) - Not available or not applicable

## Executive Summary

This document analyzes the Ragas evaluation harness to identify which strategies from the unified evaluation workflow are supported. Strategies are classified into three categories:

1. **Natively Supported**: Available immediately after `pip install ragas` with minimal configuration
2. **Supported via Third-Party Integration**: Requires external packages and glue code but has documented integration patterns
3. **Not Supported**: Not available or not applicable

Ragas is a comprehensive evaluation toolkit for Large Language Model (LLM) applications, with strong native support for RAG evaluation, synthetic test generation, and LLM-based metrics. Additionally, Ragas provides well-documented integrations with observability and monitoring platforms.

**Key Findings:**
- **Strong Native Support:** Phase I (Specification) and Phase III (Assessment)
- **Moderate Native Support:** Phase 0 (Provisioning) and Phase II (Execution)  
- **Limited Native Support:** Phase IV (Reporting) - but several features supported via integrations

---

## Phase 0: Provisioning (The Runtime)

### Step A: Harness Installation

#### ✅ **Strategy 1: PyPI Packages** - SUPPORTED
**Evidence:**
- Primary installation method via `pip install ragas`
- Git-based installation via `pip install git+https://github.com/vibrantlabsai/ragas`
- Editable install for development: `git clone` + `pip install -e .`

**Documentation References:**
- `/docs/getstarted/install.md`
- `README.md` (Installation section)

#### ✅ **Strategy 2: Git Clone** - SUPPORTED
**Evidence:**
- Explicitly documented for development work
- Command: `git clone https://github.com/vibrantlabsai/ragas.git` followed by `pip install -e .`

**Documentation References:**
- `/docs/getstarted/install.md`

#### ❌ **Strategy 3: Container Images** - NOT SUPPORTED
**Evidence:**
- No Docker images or container-based installation documented
- No Dockerfile found in repository root
- No references to Docker/OCI images in documentation

#### ❌ **Strategy 4: Binary Packages** - NOT SUPPORTED
**Evidence:**
- Pure Python package, no standalone executables
- No binary distribution mentioned in documentation

#### ❌ **Strategy 5: Node Package** - NOT SUPPORTED
**Evidence:**
- Python-only framework
- No npm/npx installation options

### Step B: Service Authentication

#### ❌ **Strategy 1: Evaluation Platform Authentication** - NOT SUPPORTED
**Evidence:**
- No dedicated evaluation platform or leaderboard submission APIs
- No command-line authentication flows for Ragas-specific services
- No registration/account system mentioned

#### ✅ **Strategy 2: API Provider Authentication** - SUPPORTED
**Evidence:**
- Supports API key configuration via environment variables for multiple providers:
  - OpenAI (`OPENAI_API_KEY`)
  - Anthropic (`ANTHROPIC_API_KEY`)
  - Google (`GOOGLE_API_KEY`)
- API keys used for LLM and embedding services
- Examples throughout documentation showing environment variable configuration

**Documentation References:**
- `/docs/getstarted/quickstart.md` (Step 3: Set Your API Key)
- `/docs/howtos/llm-adapters.md`
- `/docs/howtos/integrations/gemini.md`
- `/docs/howtos/integrations/amazon_bedrock.md`

#### ✅ **Strategy 3: Repository Authentication** - SUPPORTED
**Evidence:**
- Native support for Hugging Face repository authentication via environment variables and CLI
- Users can set `HF_TOKEN` environment variable for gated datasets/models
- Documentation shows `huggingface-cli login` for authentication
- Works directly with Ragas after standard authentication setup

**Documentation References:**
- `/docs/howtos/applications/text2sql.md` (HF_TOKEN usage, huggingface-cli login)
- `/docs/howtos/applications/compare_llms.md` (HuggingFace token setup)

---

## Phase I: Specification (The Contract)

### Step A: SUT Preparation

#### ✅ **Strategy 1: Model-as-a-Service (Remote Inference)** - STRONGLY SUPPORTED
**Evidence:**
- Primary use case for Ragas
- Comprehensive support for remote API inference:
  - OpenAI models
  - Anthropic Claude
  - Google Gemini
  - AWS Bedrock
  - Azure OpenAI
  - Local models via Ollama (OpenAI-compatible API)
- LLM factory pattern for unified interface: `llm_factory(model_name, provider, client)`

**Documentation References:**
- `/docs/getstarted/quickstart.md` (Multiple provider examples)
- `/docs/howtos/llm-adapters.md`
- `/docs/howtos/integrations/gemini.md`
- `/docs/howtos/integrations/amazon_bedrock.md`
- `/docs/howtos/integrations/oci_genai.md`

#### ✅ **Strategy 2: Model-in-Process (Local Inference)** - SUPPORTED
**Evidence:**
- Support for local models through adapters
- Can use local embedding models
- Support for loading custom models via LangChain/LlamaIndex wrappers

**Documentation References:**
- `/docs/getstarted/quickstart.md` (Ollama example)
- `/docs/howtos/integrations/llamaindex.md`
- `/docs/howtos/integrations/langchain.md`

#### ❌ **Strategy 3: Algorithm Implementation (In-Memory Structures)** - NOT SUPPORTED
**Evidence:**
- No support for evaluating ANN algorithms, knowledge graph embeddings, or BM25 indexes
- Focus is on LLM applications, not traditional IR/ML algorithms

#### ⚠️ **Strategy 4: Policy/Agent Instantiation (Stateful Controllers)** - PARTIALLY SUPPORTED
**Evidence:**
- Agent evaluation metrics exist (`TopicAdherence`, agent-specific metrics)
- Integration with LangGraph for agent evaluation
- Test generation for agents mentioned as "coming soon"
- No support for RL policies, robot controllers, or multi-agent systems

**Documentation References:**
- `/docs/concepts/metrics/available_metrics/agents.md`
- `/docs/howtos/integrations/_langgraph_agent_evaluation.md`
- `/docs/howtos/integrations/swarm_agent_evaluation.md`
- `/docs/concepts/test_data_generation/agents.md` (marked as future work)

### Step B: Benchmark Preparation (Inputs)

#### ✅ **Strategy 1: Benchmark Dataset Preparation (Offline)** - STRONGLY SUPPORTED
**Evidence:**
- `Dataset` class for loading and managing test data
- Support for multiple backends:
  - Local CSV (`backend="local/csv"`)
  - Local JSONL
  - Google Drive integration
  - In-memory datasets
- Dataset transformation and preprocessing
- Integration with Hugging Face datasets
- Data splitting capabilities

**Documentation References:**
- `/docs/concepts/datasets.md`
- `/docs/getstarted/quickstart.md` (load_dataset examples)
- `/docs/getstarted/rag_eval.md`

#### ✅ **Strategy 2: Synthetic Data Generation (Generative)** - STRONGLY SUPPORTED
**Evidence:**
- `TestsetGenerator` for automatic test data generation
- Evolution-based synthetic data generation paradigm
- Knowledge Graph-based approach for RAG test generation:
  - Document splitter for hierarchical nodes
  - Extractors (NER, keyphrase, etc.)
  - Relationship builders
  - Query synthesizers
- Different query types supported:
  - Single-hop vs multi-hop queries
  - Specific vs abstract queries
  - Different query styles (web search, chat, etc.)
- Scenario-based generation with parameters:
  - Query length (short, medium, long)
  - Query style
  - Persona (coming soon)

**Documentation References:**
- `/docs/concepts/test_data_generation/rag.md`
- `/docs/getstarted/rag_testset_generation.md`
- `README.md` (Test Data Generation feature)

#### ❌ **Strategy 3: Simulation Environment Setup (Simulated)** - NOT SUPPORTED
**Evidence:**
- No support for 3D virtual environments, physics simulation, or scene construction
- Focus is on LLM/RAG applications, not embodied AI or robotics

#### 🔌 **Strategy 4: Production Traffic Sampling (Online)** - SUPPORTED VIA THIRD-PARTY INTEGRATION
**Evidence:**
- No native production traffic sampling within Ragas
- Supported through documented integrations with external monitoring platforms:
  - Langfuse (fetch production traces)
  - TruLens (production monitoring)
  - Evidently (production monitoring)
  - Athina (production log analysis)
- Requires external package installation and integration code (typically ≤10 lines)

**Documentation References:**
- `/docs/howtos/integrations/_langfuse.md` (documented integration)
- `/docs/howtos/observability.md` (documented integration)
- `/docs/howtos/integrations/_athina.md` (documented integration)

### Step C: Benchmark Preparation (References)

#### ✅ **Strategy 1: Judge Preparation** - STRONGLY SUPPORTED
**Evidence:**
- LLM-as-judge is a core feature of Ragas
- Multiple judge-based metrics:
  - `AspectCritic`
  - `Faithfulness`
  - `AnswerCorrectness`
  - `FactualCorrectness`
  - Custom metric creation via decorators
- Judge alignment capabilities:
  - Fine-tune LLM judges on expert labels
  - Align judges with domain expert judgments
- Pre-configured judge models supported:
  - OpenAI models (GPT-4, GPT-3.5)
  - Anthropic Claude
  - Google Gemini
  - Custom judges via LangChain/LlamaIndex

**Documentation References:**
- `/docs/concepts/metrics/available_metrics/` (multiple metric docs)
- `/docs/howtos/applications/align-llm-as-judge.md`
- `/docs/howtos/applications/vertexai_alignment.md`
- `README.md` (Quick Start example with AspectCritic)

#### ✅ **Strategy 2: Ground Truth Preparation** - SUPPORTED
**Evidence:**
- Support for reference answers and ground truth in datasets
- Pre-loading of reference contexts for RAG evaluation
- Human annotations supported through dataset schema
- Embedding-based ground truth preparation for semantic similarity metrics

**Documentation References:**
- `/docs/concepts/datasets.md`
- `/docs/concepts/metrics/available_metrics/answer_correctness.md` (uses reference)
- `/docs/getstarted/rag_eval.md`

---

## Phase II: Execution (The Run)

### Step A: SUT Invocation

#### ✅ **Strategy 1: Batch Inference** - STRONGLY SUPPORTED
**Evidence:**
- Primary execution mode via `@experiment` decorator
- `experiment.arun(dataset)` for batch processing
- Support for async batch processing
- Parameterized experiments for testing different configurations
- Execution across entire datasets with progress tracking

**Documentation References:**
- `/docs/concepts/experimentation.md`
- `/docs/getstarted/quickstart.md` (evals.py example)
- `/docs/getstarted/experiments_quickstart.md`

#### ⚠️ **Strategy 2: Interactive Loop** - PARTIALLY SUPPORTED
**Evidence:**
- Agent evaluation metrics support multi-turn interactions
- `MultiTurnSample` schema for conversations
- Integration with LangGraph for stateful agent evaluation
- Limited support for traditional RL/robotics simulation loops

**Documentation References:**
- `/docs/concepts/metrics/available_metrics/agents.md`
- `/docs/howtos/integrations/_langgraph_agent_evaluation.md`

#### ❌ **Strategy 3: Arena Battle** - NOT SUPPORTED
**Evidence:**
- No native arena battle infrastructure in Ragas
- A/B testing pattern is just separate batch inference runs for different models
- Cannot execute the same input across multiple SUTs simultaneously for head-to-head comparison
- Comparison done post-hoc after separate batch runs, not true arena-style pairwise comparison

**Documentation References:**
- `/docs/concepts/experimentation.md` (A/B Testing as separate batch runs)
- `/docs/howtos/applications/benchmark_llm.md` (separate evaluation runs)

#### 🔌 **Strategy 4: Production Streaming** - SUPPORTED VIA THIRD-PARTY INTEGRATION
**Evidence:**
- No native streaming infrastructure in Ragas
- Supported through documented integrations with external monitoring platforms:
  - TruLens (real-time monitoring)
  - Evidently (drift monitoring)
  - Athina (production logs with automatic Ragas metrics)
- Requires external package installation and integration code for real-time metric collection

**Documentation References:**
- `/docs/howtos/observability.md` (documented integration)
- `/docs/howtos/integrations/_athina.md` (documented integration)

---

## Phase III: Assessment (The Score)

### Step A: Individual Scoring

#### ✅ **Strategy 1: Deterministic Measurement** - STRONGLY SUPPORTED
**Evidence:**
- Traditional NLP metrics:
  - String similarity (Levenshtein, Hamming, Jaro)
  - BLEU score
  - ROUGE score
  - Exact match
- Token-based metrics
- Distance-based metrics
- SQL query evaluation metrics

**Documentation References:**
- `/docs/concepts/metrics/available_metrics/traditional.md`
- `/docs/concepts/metrics/available_metrics/sql.md`

#### ✅ **Strategy 2: Embedding Measurement** - STRONGLY SUPPORTED
**Evidence:**
- `SemanticSimilarity` metric using embeddings and cosine similarity
- `AnswerCorrectness` combines semantic similarity with factual correctness
- Support for multiple embedding providers:
  - OpenAI embeddings
  - Google embeddings (auto-matched with Gemini)
  - Custom embeddings via `embedding_factory`
- BERTScore mentioned in documentation
- Embedding-based context evaluation

**Documentation References:**
- `/docs/concepts/metrics/available_metrics/semantic_similarity.md`
- `/docs/concepts/metrics/available_metrics/answer_correctness.md`
- `/docs/howtos/observability.md` (embedding visualization with Phoenix)

#### ✅ **Strategy 3: Subjective Measurement** - STRONGLY SUPPORTED
**Evidence:**
- Core strength of Ragas - LLM-as-judge metrics
- Extensive library of LLM-based metrics:
  - `AspectCritic` - Custom subjective criteria
  - `Faithfulness` - Hallucination detection
  - `FactualCorrectness` - Claims verification
  - `AnswerRelevance` - Relevance assessment
  - `ContextPrecision` - Context quality
  - `ContextRecall` - Context completeness
  - `RubricScore` - Rubric-based evaluation
- Custom metric creation via decorators
- Support for fine-tuned judge models

**Documentation References:**
- `/docs/concepts/metrics/available_metrics/aspect_critic.md`
- `/docs/concepts/metrics/available_metrics/faithfulness.md`
- `/docs/concepts/metrics/available_metrics/factual_correctness.md`
- `/docs/concepts/metrics/available_metrics/rubrics_based.md`
- `/docs/howtos/customizations/metrics/_write_your_own_metric.md`

#### ❌ **Strategy 4: Performance Measurement** - NOT SUPPORTED
**Evidence:**
- No built-in metrics for latency, throughput, memory, FLOPs
- No energy/carbon footprint measurement
- Focus is on quality metrics, not performance metrics
- Performance tracking would need to be done separately

### Step B: Collective Aggregation

#### ✅ **Strategy 1: Score Aggregation** - SUPPORTED
**Evidence:**
- Automatic aggregation of per-instance scores
- `evaluate()` function returns aggregate metrics
- Support for averaging across datasets
- Results stored in structured format (CSV, JSONL)
- Experiment comparison capabilities

**Documentation References:**
- `/docs/references/aevaluate.md`
- `/docs/concepts/experimentation.md`

#### ❌ **Strategy 2: Uncertainty Quantification** - NOT SUPPORTED
**Evidence:**
- No bootstrap resampling mentioned
- No confidence intervals or uncertainty bounds
- No Prediction-Powered Inference (PPI) support

---

## Phase IV: Reporting (The Output)

### Step A: Insight Presentation

#### 🔌 **Strategy 1: Execution Tracing** - SUPPORTED VIA THIRD-PARTY INTEGRATION
**Evidence:**
- No native tracing in Ragas
- Supported through documented integrations with external tracing platforms:
  - LangSmith (detailed traces)
  - Phoenix/Arize (trace visualization)
  - Langfuse (trace analysis)
  - OpenTelemetry via OpenInference
- Requires external package installation and integration code to enable tracing

**Documentation References:**
- `/docs/howtos/observability.md` (documented integrations)
- `/docs/howtos/integrations/_langsmith.md` (documented integration)
- `/docs/howtos/integrations/_langfuse.md` (documented integration)

#### ❌ **Strategy 2: Subgroup Analysis** - NOT SUPPORTED
**Evidence:**
- No built-in subgroup analysis features
- No demographic stratification
- No domain-based performance breakdown
- Would need custom analysis on exported results

#### 🔌 **Strategy 3: Chart Generation** - SUPPORTED VIA THIRD-PARTY INTEGRATION
**Evidence:**
- No native chart generation within Ragas
- Basic result display in console (text output only)
- Results saved to CSV for external visualization
- Supported through documented integration with Phoenix for embedding visualization and charts
- Requires external package installation and visualization code

**Documentation References:**
- `/docs/getstarted/quickstart.md` (shows console text output)
- `/docs/howtos/observability.md` (documented Phoenix integration for visualization)

#### 🔌 **Strategy 4: Dashboard Creation** - SUPPORTED VIA THIRD-PARTY INTEGRATION
**Evidence:**
- No native dashboard in Ragas
- Supported through documented integrations with external dashboard platforms:
  - Phoenix/Arize (embedding visualization, cluster analysis)
  - Langfuse (evaluation dashboards)
  - Athina (evaluation dashboard)
  - Zeno (interactive evaluation browser)
- Requires external package installation and integration code to create dashboards

**Documentation References:**
- `/docs/howtos/observability.md` (documented integration)
- `/docs/howtos/integrations/_langfuse.md` (documented integration)
- `/docs/howtos/integrations/_athina.md` (documented integration)
- `/docs/howtos/integrations/_zeno.md` (documented integration)

#### ❌ **Strategy 5: Leaderboard Publication** - NOT SUPPORTED
**Evidence:**
- No leaderboard submission capabilities
- No public or private leaderboard infrastructure
- Results are local or pushed to third-party platforms

#### 🔌 **Strategy 6: Regression Alerting** - SUPPORTED VIA THIRD-PARTY INTEGRATION
**Evidence:**
- No native alerting system in Ragas
- No automatic performance degradation detection natively
- Supported through documented integrations with monitoring platforms:
  - Evidently (regression detection and alerting)
  - Athina (automatic evals on production logs with alerting)
- Requires external package installation and integration code

**Documentation References:**
- `/docs/howtos/integrations/_athina.md` (documented integration with automatic evals on production logs)

---

## Summary Tables

### Phase 0: Provisioning
| Strategy | Support Level | Notes |
|----------|--------------|-------|
| 0.A.1 PyPI Packages | ✅ Native | Primary installation method |
| 0.A.2 Git Clone | ✅ Native | Documented for development |
| 0.A.3 Container Images | ❌ Not Supported | Not available |
| 0.A.4 Binary Packages | ❌ Not Supported | Python-only |
| 0.A.5 Node Package | ❌ Not Supported | Python-only |
| 0.B.1 Platform Auth | ❌ Not Supported | No Ragas platform |
| 0.B.2 API Provider Auth | ✅ Native | Multiple providers supported |
| 0.B.3 Repository Auth | ✅ Native | HF_TOKEN, huggingface-cli |

### Phase I: Specification
| Strategy | Support Level | Notes |
|----------|--------------|-------|
| I.A.1 Model-as-a-Service | ✅ Native | Primary use case |
| I.A.2 Model-in-Process | ✅ Native | Local models supported |
| I.A.3 Algorithm Implementation | ❌ Not Supported | Not applicable |
| I.A.4 Policy/Agent | ⚠️ Partial | Native agent metrics & schemas |
| I.B.1 Benchmark Datasets | ✅ Native | Multiple backends |
| I.B.2 Synthetic Generation | ✅ Native | Core feature |
| I.B.3 Simulation Environment | ❌ Not Supported | Not applicable |
| I.B.4 Production Sampling | 🔌 Integration | Via Langfuse, TruLens, Evidently, Athina |
| I.C.1 Judge Preparation | ✅ Native | Core strength |
| I.C.2 Ground Truth | ✅ Native | Well supported |

### Phase II: Execution
| Strategy | Support Level | Notes |
|----------|--------------|-------|
| II.A.1 Batch Inference | ✅ Native | Primary mode |
| II.A.2 Interactive Loop | ⚠️ Partial | Native multi-turn support |
| II.A.3 Arena Battle | ❌ Not Supported | Separate batch runs, not true arena |
| II.A.4 Production Streaming | 🔌 Integration | Via TruLens, Evidently, Athina |

### Phase III: Assessment
| Strategy | Support Level | Notes |
|----------|--------------|-------|
| III.A.1 Deterministic | ✅ Native | Traditional NLP metrics |
| III.A.2 Embedding | ✅ Native | Semantic similarity |
| III.A.3 Subjective | ✅ Native | Core strength - LLM judges |
| III.A.4 Performance | ❌ Not Supported | Not available |
| III.B.1 Score Aggregation | ✅ Native | Automatic |
| III.B.2 Uncertainty | ❌ Not Supported | Not available |

### Phase IV: Reporting
| Strategy | Support Level | Notes |
|----------|--------------|-------|
| IV.A.1 Execution Tracing | 🔌 Integration | Via LangSmith, Phoenix, Langfuse |
| IV.A.2 Subgroup Analysis | ❌ Not Supported | Not available |
| IV.A.3 Chart Generation | 🔌 Integration | Via Phoenix for visualization |
| IV.A.4 Dashboard Creation | 🔌 Integration | Via Phoenix, Langfuse, Athina, Zeno |
| IV.A.5 Leaderboard | ❌ Not Supported | Not available |
| IV.A.6 Regression Alerting | 🔌 Integration | Via Evidently, Athina |

---

## Overall Assessment

### Strengths
1. **Judge Preparation & Subjective Measurement**: Industry-leading LLM-as-judge capabilities
2. **Synthetic Data Generation**: Sophisticated knowledge graph-based approach
3. **Batch Inference**: Well-designed experiment framework
4. **API Integration**: Comprehensive support for multiple LLM providers
5. **Metrics Library**: Extensive collection of RAG and LLM evaluation metrics
6. **Third-Party Integrations**: Well-documented integrations with major observability and monitoring platforms

### Native Gaps (Available via Integration)
1. **Reporting & Visualization**: No native dashboards or charts (🔌 available via Phoenix, Langfuse, Athina, Zeno)
2. **Production Features**: No native production streaming, traffic sampling, or alerting (🔌 available via TruLens, Evidently, Athina, Langfuse)
3. **Execution Tracing**: No native tracing (🔌 available via LangSmith, Phoenix, Langfuse)

### True Gaps (Not Available)
1. **Performance Measurement**: No latency, throughput, or resource metrics
2. **Uncertainty Quantification**: No statistical confidence measures or bootstrap resampling
3. **Container Support**: No Docker/OCI images or containerized deployment options
4. **Subgroup Analysis**: No demographic stratification or domain-based performance breakdown
5. **Arena Battle**: No simultaneous head-to-head model comparison
6. **Leaderboard**: No native leaderboard publication capabilities

### Recommended Use Cases
- ✅ RAG system evaluation (native)
- ✅ LLM application quality assessment (native)
- ✅ Synthetic test data generation (native)
- ✅ LLM-as-judge development (native)
- 🔌 Production monitoring and alerting (via integrations)
- 🔌 Execution tracing and debugging (via integrations)
- 🔌 Dashboard and visualization (via integrations)
- ⚠️ Agent evaluation (native but limited)
- ❌ Performance benchmarking
- ❌ Reinforcement learning evaluation
- ❌ Traditional ML/IR algorithm evaluation

---

## Legend
- ✅ **Natively Supported**: Available immediately after `pip install ragas` with only import statements and minimal configuration (≤2 lines). No external dependencies or glue code required.
- 🔌 **Supported via Third-Party Integration**: Requires installing ≥1 external package(s) and glue code (typically ≤10 lines), but has documented integration pattern or official example. Functionality enabled through third-party tools.
- ⚠️ **Partially Supported**: Natively available but with significant limitations (e.g., limited to specific use cases)
- ❌ **Not Supported**: Feature is not available or not applicable to this harness

---

**Document Version:** 2.0  
**Last Updated:** December 15, 2024
