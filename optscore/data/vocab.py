import re

from . import kb, templates

PAD, CLS, UNK = "<pad>", "<cls>", "<unk>"
PAD_ID, CLS_ID, UNK_ID = 0, 1, 2
SPECIALS = [PAD, CLS, UNK]

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("'", ""))


class Vocab:
    def __init__(self, tokens: list[str]):
        if tokens[: len(SPECIALS)] != SPECIALS:
            raise ValueError("vocab must start with the special tokens")
        self.tokens = list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str, max_len: int = 48) -> list[int]:
        ids = [self.stoi.get(t, UNK_ID) for t in tokenize(text)]
        return [CLS_ID] + ids[: max_len - 1]


def _template_words(tpl: str) -> list[str]:
    return tokenize(tpl.replace("{a}", " ").replace("{list}", " "))


def all_statement_templates() -> list[str]:
    out = []
    for group in (templates.TRAIN_STATEMENTS, templates.HELD_OUT_STATEMENTS):
        for tpls in group.values():
            out.extend(tpls)
    return out


def build_default_vocab() -> Vocab:
    words: set[str] = {"a", "or"}
    for tpl in all_statement_templates() + templates.TRAIN_QUESTIONS + templates.HELD_OUT_QUESTIONS:
        words.update(_template_words(tpl))
    for filler in templates.FILLERS:
        words.update(tokenize(filler))
    words.update(kb.DRINKS)
    for syns in kb.ATTRIBUTES.values():
        words.update(syns)
    return Vocab(SPECIALS + sorted(words))


def train_template_words() -> set[str]:
    words: set[str] = {"a", "or"}
    for tpls in templates.TRAIN_STATEMENTS.values():
        for tpl in tpls:
            words.update(_template_words(tpl))
    for tpl in templates.TRAIN_QUESTIONS:
        words.update(_template_words(tpl))
    for filler in templates.FILLERS:
        words.update(tokenize(filler))
    words.update(kb.DRINKS)
    for syns in kb.ATTRIBUTES.values():
        words.update(syns)
    return words
