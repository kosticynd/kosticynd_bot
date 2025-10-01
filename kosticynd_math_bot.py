"""
Telegram Educational Bot with ChatGPT integration

Features implemented in this single-file prototype:
- Registration of users (students) with full FIO on first start
- Teacher identification by TELEGRAM_TEACHER_ID env var
- Teacher commands:
    /add_topic <topic title>  - generate test (questions + answers) via OpenAI and save PDF
    /report <topic id or title> - generate Excel report for the topic and send to teacher
- Students:
    /take_test  - list available topics and start a test session
- Test behavior:
    - Tests are generated as short open-answer questions (no multiple choice)
    - Correct answers are obtained from OpenAI when generating the test and saved in DB
    - During the test, student's short answers are checked for semantic correctness using OpenAI
    - If any answer is incorrect, the bot generates 5 additional targeted questions for each wrong answer
    - If all answers correct -> student moves to next topic (simple progression marker saved)
- Storage: SQLite (simple for <=10 students)
- Files: PDF (reportlab) for tests, Excel (openpyxl) for teacher report
- Environment variables (required):
    TELEGRAM_BOT_TOKEN - Telegram bot token
    OPENAI_API_KEY - OpenAI API key
    TELEGRAM_TEACHER_ID - Telegram numeric ID of the single teacher
    OPENAI_MODEL - optional, default: gpt-4

Notes & limitations:
- This is a prototype with focus on clarity and modularity; production hardening (rate limits, retries,
  background task handling, secure file storage, async PDF/Excel generation worker) is not included here.
- For deployment to free hosting (Render/Railway), see the instructions in the README section below.

Run locally:
- python3 -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt
- export TELEGRAM_BOT_TOKEN="..."
- export OPENAI_API_KEY="..."
- export TELEGRAM_TEACHER_ID="123456789"
- python telegram_edu_bot.py

"""

import os
from dotenv import load_dotenv
load_dotenv()
import json
import sqlite3
import logging
import tempfile
import datetime
from io import BytesIO

from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.types import ParseMode
from aiogram.utils import executor
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup

import openai
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from openpyxl import Workbook

# ------------------------- Configuration & Logging -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TEACHER_TELEGRAM_ID = os.environ.get("TELEGRAM_TEACHER_ID")  # should be string of teacher's numeric ID
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4")

if not TELEGRAM_TOKEN or not OPENAI_API_KEY or not TEACHER_TELEGRAM_ID:
    logger.error("Missing one of required env vars: TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, TELEGRAM_TEACHER_ID")
    raise SystemExit("Set TELEGRAM_BOT_TOKEN, OPENAI_API_KEY and TELEGRAM_TEACHER_ID environment variables.")

openai.api_key = OPENAI_API_KEY

bot = Bot(token=TELEGRAM_TOKEN, parse_mode=ParseMode.HTML)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

# ------------------------- Database helpers -------------------------
DB_PATH = "edu_bot.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        telegram_id INTEGER UNIQUE,
        full_name TEXT,
        role TEXT CHECK(role IN ('student','teacher')) DEFAULT 'student',
        created_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS topics(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT UNIQUE,
        created_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tests(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic_id INTEGER,
        questions_json TEXT,
        pdf_path TEXT,
        created_at TEXT,
        FOREIGN KEY(topic_id) REFERENCES topics(id)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS progress(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        topic_id INTEGER,
        result_json TEXT,
        score INTEGER,
        updated_at TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(topic_id) REFERENCES topics(id)
    )
    """)
    conn.commit()
    conn.close()

init_db()

# ------------------------- Utility functions -------------------------

def db_execute(query, params=(), fetch=False, one=False):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(query, params)
    if fetch:
        rows = cur.fetchall()
        conn.close()
        return rows[0] if one and rows else rows
    else:
        conn.commit()
        last = cur.lastrowid
        conn.close()
        return last

# ------------------------- OpenAI wrappers -------------------------

def openai_generate_test(topic_title, num_questions=5):
    """Ask OpenAI to generate `num_questions` short open-answer questions and clear concise correct answers."""
    prompt = (
        f"Сгенерируй {num_questions} коротких открытых вопросов для контрольного теста по теме: '{topic_title}'."
        "\nТребования:\n"
        "- Для каждого вопроса дай краткий правильный ответ (1-2 предложения).\n"
        "- Формат вывода JSON: [{\"q\": ..., \"a\": ...}, ...] без дополнительных полей.\n"
        "- Стиль: деловой строгий, без эмоциональной окраски."
    )
    logger.info("Requesting OpenAI to generate test for topic: %s", topic_title)
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Ты помогаешь генерировать тесты и ответы в формате JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.2
    )
    text = resp.choices[0].message['content']
    # Attempt to extract JSON from response
    try:
        # find first '{' or '['
        start = text.find('[')
        json_text = text[start:]
        data = json.loads(json_text)
        # Ensure list of dicts with keys q and a
        questions = []
        for item in data:
            q = item.get('q') or item.get('question')
            a = item.get('a') or item.get('answer')
            if q and a:
                questions.append({'q': q.strip(), 'a': a.strip()})
        return questions
    except Exception as e:
        logger.exception('Failed to parse OpenAI output for test generation: %s', e)
        # fallback: try to parse lines
        return []


def openai_check_answer(question, correct_answer, student_answer):
    """Use OpenAI to semantically compare student's short answer to the correct answer.
    Returns tuple (is_correct: bool, feedback: str)
    """
    system = (
        "Ты эксперт-оценщик. Оцени, эквивалентен ли по смыслу ответ ученика правильному ответу."
        " Отвечай строго, деловой тон. Верни JSON: {\"correct\": true/false, \"comment\": \"...\"}"
    )
    user = (
        f"Вопрос: {question}\n"
        f"Правильный ответ: {correct_answer}\n"
        f"Ответ ученика: {student_answer}\n"
        "Оцени смысловую эквивалентность. Учти возможные орфографические ошибки и регистр."
    )
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=200,
        temperature=0.0
    )
    text = resp.choices[0].message['content']
    # parse JSON snippet
    try:
        start = text.find('{')
        json_text = text[start:]
        obj = json.loads(json_text)
        return bool(obj.get('correct')), obj.get('comment', '').strip()
    except Exception as e:
        logger.exception('Failed to parse OpenAI check output: %s', e)
        # conservative fallback: mark incorrect and provide generic feedback
        return False, "Ответ не распознан автоматизированной системой проверки."


def openai_generate_followups(question, student_answer, num=5):
    """Generate `num` follow-up questions focusing on the misconception in student's_answer relative to question."""
    prompt = (
        f"Вопрос: {question}\n"
        f"Ответ ученика: {student_answer}\n"
        f"Сформулируй {num} дополнительных коротких вопросов, которые помогут устранить ошибочное представление, проявленное в ответе ученика."
        " Ответы не давай. Выведи JSON-список строк."
    )
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": "Генератор уточняющих контрольных вопросов."}, {"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3
    )
    text = resp.choices[0].message['content']
    try:
        start = text.find('[')
        json_text = text[start:]
        data = json.loads(json_text)
        return [s.strip() for s in data][:num]
    except Exception as e:
        logger.exception('Failed to parse followups: %s', e)
        return []

# ------------------------- PDF and Excel generation -------------------------

def generate_pdf_for_questions(title, questions):
    """Generate a simple PDF with numbered questions. `questions` is list of dicts with key 'q'."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 72
    c.setFont('Helvetica-Bold', 16)
    c.drawString(72, y, f"Тест по теме: {title}")
    y -= 36
    c.setFont('Helvetica', 12)
    for i, item in enumerate(questions, start=1):
        text = f"{i}. {item['q']}"
        # wrap text
        lines = []
        while text:
            if len(text) < 90:
                lines.append(text)
                break
            else:
                split = text[:90].rfind(' ')
                if split == -1:
                    split = 90
                lines.append(text[:split])
                text = text[split+1:]
        for line in lines:
            if y < 72:
                c.showPage()
                y = height - 72
            c.drawString(72, y, line)
            y -= 18
        y -= 6
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


def generate_excel_report(topic_title, rows):
    """Rows: list of dicts with keys: full_name, score, details (dict of q -> {student, correct, correct_flag})"""
    wb = Workbook()
    ws = wb.active
    ws.title = "Report"
    headers = ["ФИО", "Тема", "Дата", "Оценка (из %)", "Подробности"]
    ws.append(headers)
    for r in rows:
        details_text = json.dumps(r['details'], ensure_ascii=False, indent=2)
        ws.append([r['full_name'], topic_title, r.get('date', ''), r['score'], details_text])
    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf

# ------------------------- FSM States -------------------------
class RegistrationStates(StatesGroup):
    waiting_fullname = State()

class TestStates(StatesGroup):
    choosing_topic = State()
    in_test = State()

# ------------------------- Bot command handlers -------------------------

@dp.message_handler(commands=["start"])  # general entry
async def cmd_start(message: types.Message):
    tg_id = message.from_user.id
    user = db_execute("SELECT id, full_name FROM users WHERE telegram_id=?", (tg_id,), fetch=True)
    if not user:
        # new user: ask for full name
        db_execute("INSERT OR IGNORE INTO users(telegram_id, created_at) VALUES(?,?)", (tg_id, datetime.datetime.utcnow().isoformat()))
        await message.answer("Добро пожаловать. Пожалуйста, укажи своё полное ФИО (Фамилия Имя Отчество) для регистрации:")
        await RegistrationStates.waiting_fullname.set()
    else:
        # existing user
        cur = sqlite3.connect(DB_PATH).cursor()
        cur.execute("SELECT role, full_name FROM users WHERE telegram_id=?", (tg_id,))
        row = cur.fetchone()
        role = row[0]
        full = row[1]
        await message.answer(f"Здравствуйте, {full}. Ваша роль: {role}.\nИспользуйте /take_test чтобы пройти тест или /help для списка команд.")

@dp.message_handler(state=RegistrationStates.waiting_fullname)
async def process_fullname(message: types.Message, state: FSMContext):
    full = message.text.strip()
    # basic validation: at least 2 words
    if len(full.split()) < 2:
        await message.answer("Пожалуйста, укажи полное ФИО (минимум фамилия и имя).")
        return
    tg_id = message.from_user.id
    role = 'teacher' if str(tg_id) == TEACHER_TELEGRAM_ID else 'student'
    db_execute("UPDATE users SET full_name=?, role=? WHERE telegram_id=?", (full, role, tg_id))
    await state.finish()
    await message.answer(f"Регистрация завершена. Здравствуйте, {full}. Ваша роль: {role}.")

@dp.message_handler(commands=["help"])   
async def cmd_help(message: types.Message):
    text = (
        "/take_test - пройти тест\n"
        "/start - перезапустить диалог\n"
        "/help - показать это сообщение\n"
        "/add_topic <title> - (учитель) добавить тему и сгенерировать тест в PDF\n"
        "/report <topic id or title> - (учитель) получить Excel-отчёт по теме\n"
    )
    await message.answer(text)

# Teacher command: add topic
@dp.message_handler(commands=["add_topic"])
async def cmd_add_topic(message: types.Message):
    tg_id = str(message.from_user.id)
    if tg_id != TEACHER_TELEGRAM_ID:
        await message.answer("Команда доступна только учителю.")
        return
    payload = message.get_args().strip()
    if not payload:
        await message.answer("Использование: /add_topic <название темы>")
        return
    title = payload
    # insert topic
    try:
        topic_id = db_execute("INSERT INTO topics(title, created_at) VALUES(?,?)", (title, datetime.datetime.utcnow().isoformat()))
    except Exception:
        # maybe exists
        row = db_execute("SELECT id FROM topics WHERE title=?", (title,), fetch=True, one=True)
        topic_id = row[0] if row else None
    # generate test via OpenAI
    questions = openai_generate_test(title, num_questions=5)
    if not questions:
        await message.answer("Не удалось сгенерировать тест автоматически. Попробуйте позже.")
        return
    # save questions as JSON
    qjson = json.dumps(questions, ensure_ascii=False)
    pdf_buf = generate_pdf_for_questions(title, questions)
    pdf_path = f"tests/topic_{topic_id}_{int(datetime.datetime.utcnow().timestamp())}.pdf"
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    with open(pdf_path, 'wb') as f:
        f.write(pdf_buf.getvalue())
    db_execute("INSERT INTO tests(topic_id, questions_json, pdf_path, created_at) VALUES(?,?,?,?)",
               (topic_id, qjson, pdf_path, datetime.datetime.utcnow().isoformat()))
    await message.answer(f"Тема '{title}' добавлена с id={topic_id}. Тест сохранён в PDF.")
    with open(pdf_path, 'rb') as f:
        await bot.send_document(message.from_user.id, f, caption=f"Тест по теме: {title}")

# Teacher command: report
@dp.message_handler(commands=["report"])
async def cmd_report(message: types.Message):
    tg_id = str(message.from_user.id)
    if tg_id != TEACHER_TELEGRAM_ID:
        await message.answer("Команда доступна только учителю.")
        return
    arg = message.get_args().strip()
    if not arg:
        await message.answer("Использование: /report <topic id или точное название темы>")
        return
    # find topic
    row = None
    if arg.isdigit():
        row = db_execute("SELECT id, title FROM topics WHERE id=?", (int(arg),), fetch=True, one=True)
    if not row:
        row = db_execute("SELECT id, title FROM topics WHERE title=?", (arg,), fetch=True, one=True)
    if not row:
        await message.answer("Тема не найдена.")
        return
    topic_id, title = row
    # collect progress rows for this topic
    rows = db_execute("SELECT p.result_json, p.score, p.updated_at, u.full_name FROM progress p JOIN users u ON p.user_id=u.id WHERE p.topic_id=?", (topic_id,), fetch=True)
    formatted = []
    for item in rows:
        result_json, score, updated_at, full_name = item
        details = json.loads(result_json)
        formatted.append({"full_name": full_name, "score": score, "details": details, "date": updated_at})
    if not formatted:
        await message.answer("Нет результатов по этой теме.")
        return
    excel_buf = generate_excel_report(title, formatted)
    await bot.send_document(message.from_user.id, ('report.xlsx', excel_buf))

# Student: list and take test
@dp.message_handler(commands=["take_test"])
async def cmd_take_test(message: types.Message):
    tg_id = message.from_user.id
    # ensure user registered
    u = db_execute("SELECT id, full_name FROM users WHERE telegram_id=?", (tg_id,), fetch=True, one=True)
    if not u or not u[1]:
        await message.answer("Пожалуйста, начните с /start и укажите своё полное ФИО.")
        return
    # list topics
    topics = db_execute("SELECT id, title FROM topics", fetch=True)
    if not topics:
        await message.answer("Пока нет доступных тем. Свяжитесь с учителем.")
        return
    text = "Доступные темы:\n"
    for t in topics:
        text += f"{t[0]} - {t[1]}\n"
    text += "\nОтправьте id темы, чтобы начать тест."
    await message.answer(text)
    await TestStates.choosing_topic.set()

@dp.message_handler(state=TestStates.choosing_topic)
async def process_topic_choice(message: types.Message, state: FSMContext):
    arg = message.text.strip()
    if not arg.isdigit():
        await message.answer("Пожалуйста, отправьте числовой id темы из списка.")
        return
    topic_row = db_execute("SELECT id, title FROM topics WHERE id=?", (int(arg),), fetch=True, one=True)
    if not topic_row:
        await message.answer("Тема не найдена. Попробуйте ещё раз.")
        return
    topic_id, title = topic_row
    # get latest test for topic
    test_row = db_execute("SELECT id, questions_json FROM tests WHERE topic_id=? ORDER BY created_at DESC LIMIT 1", (topic_id,), fetch=True, one=True)
    if not test_row:
        await message.answer("Для этой темы нет теста. Свяжитесь с учителем.")
        await state.finish()
        return
    test_id, qjson = test_row
    questions = json.loads(qjson)
    # store test session in FSM data
    await state.update_data(topic_id=topic_id, test_id=test_id, questions=questions, current_index=0, answers=[])
    await message.answer(f"Тест по теме '{title}' начинается. Отвечайте кратко.\nВопрос 1:\n{questions[0]['q']}")
    await TestStates.in_test.set()

@dp.message_handler(state=TestStates.in_test)
async def process_test_answer(message: types.Message, state: FSMContext):
    data = await state.get_data()
    idx = data['current_index']
    questions = data['questions']
    q = questions[idx]
    student_answer = message.text.strip()
    # check answer via OpenAI
    is_correct, comment = openai_check_answer(q['q'], q['a'], student_answer)
    answers = data['answers']
    answers.append({'q': q['q'], 'correct_answer': q['a'], 'student_answer': student_answer, 'is_correct': is_correct, 'comment': comment})
    idx += 1
    if idx >= len(questions):
        # test finished - evaluate
        total = len(answers)
        correct = sum(1 for a in answers if a['is_correct'])
        score = int(100 * correct / total)
        # if all correct -> mark progression, otherwise generate follow-ups for wrong ones
        followups = {}
        if correct == total:
            # save progress
            tg_id = message.from_user.id
            user = db_execute("SELECT id FROM users WHERE telegram_id=?", (tg_id,), fetch=True, one=True)
            user_id = user[0]
            topic_id = data['topic_id']
            db_execute("INSERT INTO progress(user_id, topic_id, result_json, score, updated_at) VALUES(?,?,?,?,?)",
                       (user_id, topic_id, json.dumps(answers, ensure_ascii=False), score, datetime.datetime.utcnow().isoformat()))
            await message.answer(f"Все ответы верны. Отлично, вы можете переходить к следующей теме. Оценка: {score}%")
            await state.finish()
            return
        else:
            # for each wrong answer, generate 5 followups
            for a in answers:
                if not a['is_correct']:
                    qtext = a['q']
                    sa = a['student_answer']
                    extras = openai_generate_followups(qtext, sa, num=5)
                    followups[qtext] = extras
            # save progress
            tg_id = message.from_user.id
            user = db_execute("SELECT id FROM users WHERE telegram_id=?", (tg_id,), fetch=True, one=True)
            user_id = user[0]
            topic_id = data['topic_id']
            db_execute("INSERT INTO progress(user_id, topic_id, result_json, score, updated_at) VALUES(?,?,?,?,?)",
                       (user_id, topic_id, json.dumps(answers, ensure_ascii=False), score, datetime.datetime.utcnow().isoformat()))
            # send follow-ups grouped
            text = f"Некоторые ответы содержат ошибки. Оценка: {score}%. Для каждой ошибки подготовлены дополнительные вопросы:\n"
            for qtext, extras in followups.items():
                text += f"\nОшибка в вопросе: {qtext}\n"
                for i, ex in enumerate(extras, start=1):
                    text += f"   {i}. {ex}\n"
            await message.answer(text)
            await state.finish()
            return
    else:
        # ask next question
        await state.update_data(current_index=idx, answers=answers)
        await message.answer(f"Вопрос {idx+1}:\n{questions[idx]['q']}")

# ------------------------- Startup -------------------------
if __name__ == '__main__':
    logger.info('Bot started')
    executor.start_polling(dp, skip_updates=True)
