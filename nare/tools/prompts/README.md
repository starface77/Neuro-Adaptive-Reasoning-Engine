# Compact Prompt Tools System

## Проблема

Раньше NARE отправлял огромные system prompts (5-6k токенов) даже для простых задач типа "Привет" или "создай файл test.py".

## Решение

Создана система специализированных инструментов с минимальными промптами:

```
nare/tools/prompts/
├── __init__.py
├── list_files.py       # Список файлов (~50 токенов)
├── read_file.py        # Чтение файла (~40 токенов)
├── write_file.py       # Создание файла (~50 токенов)
├── edit_file.py        # Редактирование (~60 токенов)
└── bash_command.py     # Bash команды (~50 токенов)
```

## Как работает

1. **Классификация**: Session определяет intent (QUESTION/EDIT/EXPLORE)
2. **Проверка**: `should_use_tools()` решает, использовать ли компактные инструменты
3. **Генерация**: Вместо огромного промпта отправляется:
   - Минимальный system prompt (~20 токенов)
   - Список доступных инструментов (~200 токенов)
   - Запрос пользователя (~50 токенов)
4. **Выполнение**: LLM выбирает инструмент, система его выполняет

## Экономия токенов

| Задача | Было | Стало | Экономия |
|--------|------|-------|----------|
| "Привет" | 6.7k | ~500 | 92% |
| "создай test.py" | 6.5k | ~300 | 95% |
| "покажи main.py" | 6.8k | ~400 | 94% |
| Сложная задача | 7k | 7k | 0% (fallback) |

## Когда используются компактные инструменты

✅ **Используются для**:
- Простые файловые операции (read, write, edit)
- Список файлов
- Bash команды
- Вопросы с ключевыми словами: "создай", "покажи", "список"

❌ **НЕ используются для**:
- Сложные задачи с рассуждениями
- Многошаговые workflow
- Вопросы требующие глубокого анализа
- Fallback на обычный Router

## Добавление новых инструментов

Создайте файл `nare/tools/prompts/new_tool.py`:

```python
"""Tool description."""

TOOL_NAME = "tool_name"
TOOL_DESCRIPTION = "Short description"

TOOL_PROMPT = """Tool usage instructions.

Args:
- param1: Description
- param2: Description (default: "value")

Example:
User: "example query"
tool_name("arg1", "arg2")
"""

def execute(param1: str, param2: str = "default") -> str:
    """Execute tool logic."""
    # Your code here
    return "Result"
```

Инструмент автоматически загрузится при следующем запуске.

## Архитектура

```
Session._solve_async()
    ↓
should_use_tools(query, intent)
    ↓ (if True)
generate_with_tools()
    ↓
load_prompt_tools() → get_tools_schema()
    ↓
LLM выбирает инструмент
    ↓
execute_tool(tool_name, **params)
    ↓
Результат пользователю
```

## Конфигурация

Система работает автоматически. Для отключения:

```python
# В nare/tools/compact_generation.py
def should_use_tools(query: str, intent: str) -> bool:
    return False  # Отключить компактные инструменты
```

## Логирование

```
[Session] Intent: EDIT
[Session] Using compact tool-based generation
[PromptTools] Loaded 5 tools
[PromptTools] Executing: write_file
```

## Тестирование

```bash
> создай test.py с функцией hello
# Должно использовать write_file (~300 токенов)

> Привет
# Должно использовать обычный Router (~500 токенов, без repo_map)

> сложная задача с анализом кода
# Должно использовать обычный Router (fallback)
```
