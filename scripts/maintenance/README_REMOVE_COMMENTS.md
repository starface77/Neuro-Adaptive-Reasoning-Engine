# Скрипты для удаления комментариев

## Два варианта

### 1. remove_comments.py - Умное удаление

Удаляет комментарии, но **сохраняет docstrings** (документацию функций/классов).

```bash
# Обработать всю папку
python scripts/maintenance/remove_comments.py nare/

# Обработать один файл
python scripts/maintenance/remove_comments.py nare/cli/session.py

# Удалить и docstrings тоже
python scripts/maintenance/remove_comments.py nare/ --no-docstrings

# Не создавать backup файлы
python scripts/maintenance/remove_comments.py nare/ --no-backup
```

**Что удаляет:**
- ✓ Однострочные комментарии `# comment`
- ✓ Многострочные комментарии `"""comment"""`
- ✗ Docstrings функций/классов (сохраняет)

**Что сохраняет:**
- ✓ Shebang `#!/usr/bin/env python3`
- ✓ Docstrings модулей/классов/функций
- ✓ Исполняемый код

### 2. strip_all_comments.py - Агрессивное удаление

Удаляет **ВСЁ** - комментарии и docstrings. Остаётся только код.

```bash
# Обработать всю папку
python scripts/maintenance/strip_all_comments.py nare/

# Обработать один файл
python scripts/maintenance/strip_all_comments.py nare/cli/session.py
```

**Что удаляет:**
- ✓ Однострочные комментарии `# comment`
- ✓ Многострочные комментарии `"""comment"""`
- ✓ ВСЕ docstrings (модули, классы, функции)
- ✓ Пустые строки

**Что сохраняет:**
- ✓ Shebang `#!/usr/bin/env python3`
- ✓ Исполняемый код

## Безопасность

Оба скрипта создают **backup файлы** с расширением `.bak`:

```bash
# Если что-то пошло не так, восстановить:
mv nare/cli/session.py.bak nare/cli/session.py

# Или восстановить всё:
find nare/ -name "*.bak" -exec bash -c 'mv "$0" "${0%.bak}"' {} \;

# Удалить все backup файлы после проверки:
find nare/ -name "*.bak" -delete
```

## Примеры

### До (с комментариями)

```python
"""Module docstring."""

import os

# This is a comment
def hello(name: str) -> str:
    """Say hello to someone.
    
    Args:
        name: Person's name
        
    Returns:
        Greeting message
    """
    # Build greeting
    return f"Hello, {name}!"  # Return it
```

### После remove_comments.py (сохранил docstrings)

```python
"""Module docstring."""

import os

def hello(name: str) -> str:
    """Say hello to someone.
    
    Args:
        name: Person's name
        
    Returns:
        Greeting message
    """
    return f"Hello, {name}!"
```

### После strip_all_comments.py (удалил всё)

```python
import os

def hello(name: str) -> str:
    return f"Hello, {name}!"
```

## Когда использовать

### remove_comments.py (рекомендуется)
- Хотите убрать лишние комментарии
- Но сохранить документацию API
- Для production кода

### strip_all_comments.py (агрессивно)
- Нужен минимальный размер
- Документация не важна
- Для обфускации или минификации

## Статистика

После обработки показывается статистика:

```
✓ nare/cli/session.py
  357 → 298 lines (59 removed)

✓ nare/cli/repl.py
  357 → 312 lines (45 removed)

Processed 140 files
```

## Восстановление

Если результат не понравился:

```bash
# Восстановить один файл
mv nare/cli/session.py.bak nare/cli/session.py

# Восстановить всю папку
cd /c/Users/danik/Documents/NareCLI
find nare/ -name "*.py.bak" | while read f; do
    mv "$f" "${f%.bak}"
done

# Или через git (если под контролем версий)
git checkout nare/
```

## Рекомендации

1. **Сначала протестируйте на одном файле:**
   ```bash
   python scripts/maintenance/remove_comments.py nare/cli/session.py
   # Проверьте результат
   cat nare/cli/session.py
   ```

2. **Проверьте что код работает:**
   ```bash
   python -m pytest tests/
   python -m nare.cli
   ```

3. **Если всё ок - обработайте всё:**
   ```bash
   python scripts/maintenance/remove_comments.py nare/
   ```

4. **Удалите backup файлы:**
   ```bash
   find nare/ -name "*.bak" -delete
   ```

## Ограничения

- Не обрабатывает файлы в `__pycache__` и `.git`
- Может неправильно обработать сложные строковые литералы
- Не проверяет синтаксис после удаления
- Рекомендуется запустить тесты после обработки
