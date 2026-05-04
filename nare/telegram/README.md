# Telegram Bot

Telegram интерфейс для управления NARE daemon.

## Компоненты

- `bot.py` - Основной Telegram bot
- `handlers.py` - Обработчики команд
- `notifications.py` - Уведомления о завершении задач
- `questions.py` - Интерактивные вопросы от агента
- `keyboards.py` - Inline клавиатуры для подтверждений

## Команды

```
/start - Запустить бота
/task <описание> - Добавить задачу
/status - Статус текущих задач
/logs <task_id> - Логи задачи
/memory - Статистика памяти
/skills - Список скомпилированных скиллов
/pause - Приостановить выполнение
/resume - Возобновить
/cancel <task_id> - Отменить задачу
```

## Настройка

1. Создать бота через @BotFather
2. Получить токен
3. Добавить в `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_token_here
   TELEGRAM_CHAT_ID=your_chat_id
   ```
4. Запустить: `python -m nare.telegram.bot`

## Документация

См. `docs/guides/telegram-bot.md` для подробностей.
