import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


# --- Q2 helpers strictly following the two guidelines ---

from nltk.corpus import stopwords

# Make sure stopwords are downloaded
import nltk
nltk.download('stopwords')

# --- Hard OOD transform: synonym replacement + QWERTY typos ---

_QWERTY_NEIGHBORS = {
    'q':'was', 'w':'qesa', 'e':'wrsd', 'r':'etdf', 't':'ryfg', 'y':'tugh', 'u':'yihj', 'i':'uojk',
    'o':'ipkl', 'p':'ol',
    'a':'qwsz', 's':'awedxz', 'd':'serfcx', 'f':'drtgv', 'g':'ftyhbv', 'h':'gyujnb',
    'j':'huikmn', 'k':'jiolmn', 'l':'kop',
    'z':'xsa', 'x':'zsdc', 'c':'xdfv', 'v':'cfgb', 'b':'vghn', 'n':'bhjm', 'm':'njk'
}

_STOPWORDS = set(stopwords.words("english"))
# Keep “not” out of the list to avoid flipping sentiment
_STOPWORDS.discard("not")

def _preserve_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src.istitle():
        return repl.capitalize()
    return repl

def _pick_wordnet_synonym(word: str) -> str | None:
    w = word.lower()
    lemmas = set()
    for syn in wordnet.synsets(w):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.isalpha() and ' ' not in name and name.lower() != w:
                lemmas.add(name.lower())
    if not lemmas:
        return None
    # Bias toward rarer ones for stronger perturbation
    pool = sorted(lemmas, reverse=True)
    return random.choice(pool[: max(3, len(pool)//2)])

def _qwerty_typo(word: str) -> str:
    """One typo: swap an interior pair or replace one char with a QWERTY neighbor."""
    if len(word) < 4:
        return word
    if random.random() < 0.5:
        i = random.randrange(1, len(word) - 1)
        chars = list(word)
        chars[i], chars[i+1] = chars[i+1], chars[i]
        return ''.join(chars)
    i = random.randrange(1, len(word) - 1)
    ch = word[i].lower()
    neighbors = _QWERTY_NEIGHBORS.get(ch)
    if not neighbors:
        return word
    new_ch = random.choice(neighbors)
    if word[i].isupper():
        new_ch = new_ch.upper()
    return word[:i] + new_ch + word[i+1:]


def custom_transform(example):
    """
    Aggressive version using NLTK stopwords:
      - Replace ~30% of eligible words by WordNet synonyms (max 6 per sentence).
      - Inject QWERTY-typos on ~15% of remaining words (max 5 per sentence).
    Keeps semantics while creating realistic OOD noise.
    """
    text = example.get("text", "")
    if not isinstance(text, str) or not text:
        return example

    tokens = word_tokenize(text)

    p_syn = 0.7
    p_typo = 0.7
    max_syn = 15
    max_typo = 15

    syn_used = 0
    typo_used = 0
    changed = set()

    # --- Synonym replacement ---
    for i, tok in enumerate(tokens):
        if syn_used >= max_syn:
            break
        if not tok.isalpha():
            continue
        if len(tok) < 3 or tok.lower() in _STOPWORDS:
            continue
        if tok[0].isupper() and i != 0:
            continue
        if random.random() < p_syn:
            syn = _pick_wordnet_synonym(tok)
            if syn:
                tokens[i] = _preserve_case(tok, syn)
                changed.add(i)
                syn_used += 1

    # --- Typo injection ---
    for i, tok in enumerate(tokens):
        if typo_used >= max_typo:
            break
        if i in changed:
            continue
        if not tok.isalpha() or len(tok) < 4 or tok.lower() == "not":
            continue
        if random.random() < p_typo:
            new_tok = _qwerty_typo(tok)
            if new_tok != tok:
                tokens[i] = new_tok
                typo_used += 1

    example["text"] = TreebankWordDetokenizer().detokenize(tokens)
    return example

