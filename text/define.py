# NOTE: this should be move to other place

from text.symbols import symbols

"""
0: en   1: zh   2: fr   3:de
4: ru   5: es   6: jp
"""
def get_phoneme_set(path, encoding='utf-8'):
    phns = []
    with open(path, 'r', encoding=encoding) as f:
        for line in f:
            if line == '\n':
                continue
            phns.append('@' + line.strip())
    return phns


LANG_ID2SYMBOLS = {
    0: symbols,
    1: symbols,
    2: get_phoneme_set("../MFA/lexicon/French/phoneset.txt"),
    3: get_phoneme_set("../MFA/lexicon/German/phoneset.txt"),
    4: [],
    5: get_phoneme_set("../MFA/lexicon/Spanish/phoneset.txt"),
    6: [],
}
