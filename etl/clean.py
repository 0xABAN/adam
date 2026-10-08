"""Clean the Discord export into one chronological JSONL of Adam's messages.

Reads  data/discord/Messages/c<id>/{channel.json,messages.json}
Writes data/clean/messages.jsonl  one row per message from 2022 on:
           {"channel", "type", "time", "text"}, grouped by channel, oldest first
       data/clean/dropped.jsonl   every removed message and the rule that removed it

A removed message keeps its row with text "" (like an image-only message), so the
gaps between the remaining messages stay exactly as Adam sent them. One-word
replies are not removed here: that check runs after messages are merged.
"""

import json
import re
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
SOURCE = DATA / "discord" / "Messages"
OUT = DATA / "clean"
CUTOFF = "2022-01-01"  # ISO timestamps sort correctly as strings

URL = re.compile(r"https?://\S+")

# Messages dropped whole.
BOT = re.compile(r"^[$.?&!;%,>][A-Za-z]")  # $wa, .c, ?solved, &daily; not $1750 or "^ weird"
API_KEY = re.compile(r"\b(sk-[\w-]{20,}|xai-\w{20,}|ghp_\w{20,}|AKIA[0-9A-Z]{16})|\b[A-Z_]*(API_KEY|TOKEN|SECRET)[A-Z_]*\s*=")
BOX_DRAWING = re.compile(r"[│┌┐└┘├┤─]")  # not ╯, which kaomoji use: (╯°□°)╯︵ ┻━┻
HEADING = re.compile(r"^#{1,3} ", re.MULTILINE)
CODE_LINE = re.compile(
    r"^\s{2,}\S"  # indented
    r"|[;{}]\s*$"  # statement or block end
    r"|^\s*[\w.$\[\]]+\(.*\)\s*$"  # the whole line is a call
    r"|^\s*[\w.$\[\]]+\s*=[^=]"  # assignment
    r"|^\s*(def|class|import|from|return|print|const|let|function|public)\b"
    r"|<-"  # R assignment
)
DATA_DUMP = re.compile(r"^\s*[\[{].{28,}[\]}]\s*$", re.DOTALL)  # [[0, 1], ...] or {'key': ...}
COMMAND = re.compile(r"^\s*(PS [A-Z]:\\|curl -|cmake -)")
CALL_SHAPE = re.compile(r"^[\w.]*[(\[].*[)\]][\w.,:;]*$")  # O(n^2), matrix[0][1]

# Pieces stripped out of a message that is otherwise kept.
STRIP = [
    re.compile(r"```.*?```", re.DOTALL),  # code blocks
    re.compile(r"^> .*$", re.MULTILINE),  # quotes of someone else's message
    URL,
    re.compile(r"<[@#][!&]?\d+>|@everyone|@here"),  # mentions
    re.compile(r"<a?:\w+:\d+>"),  # custom emoji
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),  # emails
    re.compile(r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b"),  # phone numbers
    re.compile(r"\|\||\*\*|__|`"),  # spoiler bars and markdown, keeping the text inside
]


def strip(text: str) -> str:
    for pattern in STRIP:
        text = pattern.sub(" ", text)

    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def has_credential(raw: str) -> bool:
    """API keys, passwords and tokens Adam pasted into chats.

    Besides known key formats, flags any token (links aside) that looks generated:
    a password mixing lowercase, uppercase, digits and symbols, or a long blob mixing
    cases and digits. Math like O(n^2) and command-line flags are skipped.
    """
    if API_KEY.search(raw):
        return True

    for token in strip(raw).split():
        if CALL_SHAPE.match(token) or token.startswith("-"):
            continue

        mixed = any(c.islower() for c in token) and any(c.isupper() for c in token) and any(c.isdigit() for c in token)
        symbol = any(c in "!@#$%^&*+=?~" for c in token)
        if mixed and ((len(token) >= 8 and symbol) or len(token) >= 20):
            return True
    return False


def looks_like_code(text: str) -> bool:
    """Code pasted without backticks, terminal output and data dumps.

    Judged by line shape, not symbol counts: Adam's own messages are full of
    >>>>> and fill-in-the-blank _, which symbol counts mistake for code.
    """
    if BOX_DRAWING.search(text) or DATA_DUMP.match(text) or COMMAND.match(text):
        return True

    lines = text.splitlines()
    if len(lines) == 1:
        return text.rstrip().endswith(";") and ("(" in text or "=" in text)  # not "jkfd;afj;"
    return sum(bool(CODE_LINE.search(line)) for line in lines) >= len(lines) / 2


def looks_formal(text: str) -> bool:
    """Pasted ChatGPT answers, essays and announcements.

    Adam's own long messages are lowercase run-ons. Text where most sentences start
    with a capital and end with punctuation was written for somewhere else.
    """
    if HEADING.search(text):
        return True

    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if re.search(r"[A-Za-z]", s)]
    if len(text) < 200 or len(sentences) < 3:
        return False

    capitalized = sum(s.lstrip('"“*-( ')[:1].isupper() for s in sentences) / len(sentences)
    punctuated = sum(s.rstrip('"”*) ')[-1:] in ".!?:" for s in sentences) / len(sentences)
    return capitalized >= 0.75 and punctuated >= 0.5


def looks_like_lines(text: str) -> bool:
    """Song lyrics and title lists: short lines that each start with a capital."""
    lines = text.splitlines()
    if len(lines) < 4:
        return False

    capitalized = sum(line.lstrip("\"“'(-* ")[:1].isupper() for line in lines) / len(lines)
    return capitalized >= 0.7 and sum(map(len, lines)) / len(lines) <= 60


def drop_reason(raw: str, text: str, copies: Counter) -> str | None:
    """Return why a message should be dropped, or None to keep it."""
    if has_credential(raw):
        return "secret"
    if BOT.match(raw.strip()):
        return "bot"
    if not text:
        return None  # nothing left to judge; the row stays as an empty event
    if looks_like_code(text):
        return "code"
    if looks_formal(text):
        return "formal"
    if looks_like_lines(text):
        return "lines"

    # Copypastas: the same long message sent more than twice. Sending one update
    # to two friends is normal, so the first two copies stay.
    key = " ".join(text.lower().split())
    if len(key) >= 60:
        copies[key] += 1
        if copies[key] > 2:
            return "duplicate"
    return None


def main() -> None:
    OUT.mkdir(exist_ok=True)
    copies = Counter()
    reasons = Counter()
    kept = 0

    with open(OUT / "messages.jsonl", "w") as clean, open(OUT / "dropped.jsonl", "w") as dropped:
        for folder in sorted(SOURCE.iterdir()):
            if not (folder / "channel.json").exists():
                continue

            kind = json.loads((folder / "channel.json").read_text()).get("type")
            records = json.loads((folder / "messages.json").read_text())
            records.reverse()  # the export lists newest first

            for record in records:
                if record["Timestamp"] < CUTOFF:
                    continue

                raw = record["Contents"]
                text = strip(raw)
                reason = drop_reason(raw, text, copies)

                if reason:
                    reasons[reason] += 1
                    dropped.write(
                        json.dumps({"channel": folder.name, "time": record["Timestamp"], "reason": reason, "text": raw}, ensure_ascii=False)
                        + "\n"
                    )
                    text = ""
                elif text:
                    kept += 1

                row = {"channel": folder.name, "type": kind, "time": record["Timestamp"], "text": text}
                clean.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"kept {kept} messages with text; dropped {sum(reasons.values())}: {dict(reasons.most_common())}")


if __name__ == "__main__":
    main()
