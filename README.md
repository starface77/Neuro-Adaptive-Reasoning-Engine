# NARE (Non-parametric Amortized Reasoning Evolution)
*Deterministic routing of logic tasks via semantic compression and executable reflexes.*

[Читать на русском языке (Read in Russian)](#russian)

NARE is a Skill-Based Cognitive Architecture designed to transition inference-heavy LLM reasoning (System 2) into zero-shot deterministic execution (System 1). The system dynamically learns from its own reasoning trajectories, compiles Python-based abstract algorithms during a consolidation phase, and executes them to solve recurring logical classes with O(1) latency and zero API cost.

## Core Architecture

- **Reasoning Amortization**: Shifts computational complexity from auto-regressive LLM generation to local procedural execution.
- **Executable Reflexes**: Automatically synthesizes and compiles Abstract Syntax Trees (AST) based on consolidated episodic memory to solve recurring logical patterns.
- **Dynamic 4-Way Routing Protocol**:
  1. **REFLEX (Execution)**: O(1) procedural execution of crystallized Python skills. Bypasses LLM generation entirely.
  2. **FAST (Cache)**: Deterministic retrieval of exact-match prior solutions via dense vector similarity.
  3. **HYBRID (Delta-Reasoning)**: Context-augmented inference leveraging past reasoning traces to solve structurally similar, but novel variants.
  4. **SLOW (Chain-of-Thought)**: Deep, multi-sample exploratory reasoning evaluated by an internal Elo-based Hybrid Critic.
- **Fault-Tolerant Skill Registry (Confidence Gating)**: Generated algorithms are evaluated in an isolated execution environment. Runtime exceptions dynamically penalize the skill's confidence scalar, prompting a safe fallback to inference-based reasoning.

## Cognitive Workflow

1. **Episodic Encoding**: The agent processes a novel stimulus via the SLOW path. Successful reasoning trajectories are embedded and stored in a dense FAISS index.
2. **Consolidation (Sleep Phase)**: Upon reaching a density threshold of semantically analogous episodes, the agent initiates consolidation. It extracts the underlying heuristic and compiles an abstract Python algorithm (comprising `trigger()` and `execute()` functions).
3. **Procedural Execution**: Subsequent stimuli matching the consolidated semantic boundary are intercepted by the `trigger()` function. The agent bypasses the neural generation pipeline and invokes the procedural `execute()` function, achieving 100% token conservation.

## Benchmark Metrics
Empirical evaluation demonstrates the architecture's efficiency in structural logic tasks:
```text
Total Tasks: 7
SLOW Paths: 1 (14.3%)
HYBRID Paths: 3 (42.9%)
REFLEX Paths (Executable): 2 (28.6%)
FAST Paths (Cache): 1 (14.3%)

Speedup via Executable Reflex: Exponential 
Token Savings on Reflex Tasks: 100% (0 generation tokens used)
```

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/nare.git
cd nare

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
echo "GEMINI_API_KEY=your_key_here" > .env

# 4. Execute the architectural benchmark
python benchmarks/metrics_benchmark.py
```

---

<a name="russian"></a>
# NARE (Непараметрическая Эволюция Амортизированных Рассуждений)
*Детерминированный роутинг логических задач через семантическое сжатие и исполняемые рефлексы.*

NARE представляет собой когнитивную архитектуру, основанную на навыках, разработанную для перевода вычислительно затратных LLM-рассуждений (System 2) в детерминированное исполнение (System 1). Система динамически обучается на собственных траекториях рассуждений, компилирует абстрактные алгоритмы на Python во время фазы консолидации и выполняет их для решения повторяющихся классов логических задач с задержкой O(1) и нулевыми затратами на API.

## Базовая архитектура

- **Амортизация рассуждений**: Перенос вычислительной сложности с авторегрессионной генерации LLM на локальное процедурное исполнение.
- **Исполняемые рефлексы**: Автоматический синтез и компиляция алгоритмов на базе абстрактных синтаксических деревьев (AST) для решения повторяющихся паттернов.
- **Протокол 4-х фазного роутинга**:
  1. **REFLEX (Execution)**: O(1) процедурное исполнение кристаллизованных навыков. Полностью обходит этап LLM-генерации.
  2. **FAST (Cache)**: Детерминированное извлечение точных совпадений через плотное векторное сходство.
  3. **HYBRID (Delta-Reasoning)**: Контекстно-аугментированный вывод, использующий прошлые траектории рассуждений для решения структурно схожих вариантов задач.
  4. **SLOW (Chain-of-Thought)**: Глубокое исследовательское рассуждение с многовариантной выборкой и оценкой внутренним турнирным Критиком.
- **Отказоустойчивый реестр навыков (Confidence Gating)**: Сгенерированные алгоритмы оцениваются в изолированной среде. Исключения во время выполнения (Runtime exceptions) динамически штрафуют показатель уверенности навыка, инициируя безопасный откат к нейросетевым рассуждениям.

## Когнитивный процесс

1. **Эпизодическое кодирование**: Агент обрабатывает новый стимул через маршрут SLOW. Успешные траектории рассуждений эмбеддятся и сохраняются в векторном индексе FAISS.
2. **Консолидация (Фаза Сна)**: По достижении порога плотности семантически аналогичных эпизодов агент инициирует консолидацию. Он извлекает базовую эвристику и компилирует абстрактный алгоритм на Python (включающий функции `trigger()` и `execute()`).
3. **Процедурное исполнение**: Последующие стимулы, попадающие в консолидированную семантическую границу, перехватываются функцией `trigger()`. Агент обходит нейронный конвейер и вызывает процедурную функцию `execute()`, достигая 100% экономии токенов.

## Метрики бенчмарка
Эмпирическая оценка демонстрирует эффективность архитектуры на задачах структурной логики:
```text
Total Tasks: 7
SLOW Paths: 1 (14.3%)
HYBRID Paths: 3 (42.9%)
REFLEX Paths (Executable): 2 (28.6%)
FAST Paths (Cache): 1 (14.3%)

Ускорение за счет Executable Reflex: Экспоненциальное
Экономия токенов на Reflex-задачах: 100% (потрачено 0 токенов генерации)
```

## Быстрый старт

```bash
# 1. Клонирование репозитория
git clone https://github.com/yourusername/nare.git
cd nare

# 2. Установка зависимостей
pip install -r requirements.txt

# 3. Конфигурация окружения
echo "GEMINI_API_KEY=ваш_ключ" > .env

# 4. Запуск архитектурного бенчмарка
python benchmarks/metrics_benchmark.py
```
