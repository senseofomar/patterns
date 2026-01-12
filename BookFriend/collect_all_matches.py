import os
import re
from collections import namedtuple
from rapidfuzz import fuzz

Match = namedtuple("Match", ["file", "sentence", "start", "end", "keyword", "snippet", "is_fuzzy"])

def collect_all_matches(folder_path, keywords, fuzzy=False, threshold=80):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(folder_path)

    matches = []

    # whole-word, case-insensitive patterns
    patterns = {
        kw: re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
        for kw in keywords
    }

    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(folder_path, fname)
        try:
            text = open(path, encoding="utf-8").read()
        except Exception:
            continue

        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            sent_lower = sentence.lower()

            for kw, pat in patterns.items():
                # exact matches
                for m in pat.finditer(sentence):
                    start, end = m.start(), m.end()
                    snippet = sentence[max(0, start-30): end+30]
                    matches.append(Match(fname, sentence, start, end, kw, snippet, False))

                # fuzzy word-level matches
                if fuzzy:
                    words = re.findall(r"\w+", sent_lower)
                    for word in words:
                        score = fuzz.ratio(kw.lower(), word)
                        if score >= threshold and not pat.search(sentence):
                            start = sent_lower.find(word)
                            end = start + len(word)
                            snippet = sentence[max(0, start-30): end+30]
                            matches.append(Match(fname, sentence, start, end,
                                                 f"{kw} (fuzzy {score}%)",
                                                 snippet, True))

    return matches
