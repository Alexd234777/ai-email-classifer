import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re


# DEVICE (CPU ONLY)

device = torch.device("cpu")
print("[MODEL] Using device:", device)


# LOAD MODELS (ENSEMBLE)


# 1) Toxicity model
TOX_MODEL_NAME = "unitary/toxic-bert"
tox_tokenizer = AutoTokenizer.from_pretrained(TOX_MODEL_NAME)
tox_model = AutoModelForSequenceClassification.from_pretrained(TOX_MODEL_NAME).to(device)

# 2) Offensive language model
OFF_MODEL_NAME = "cardiffnlp/twitter-roberta-base-offensive"
off_tokenizer = AutoTokenizer.from_pretrained(OFF_MODEL_NAME)
off_model = AutoModelForSequenceClassification.from_pretrained(OFF_MODEL_NAME).to(device)

# 3) Emotion model
EMO_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
emo_tokenizer = AutoTokenizer.from_pretrained(EMO_MODEL_NAME)
emo_model = AutoModelForSequenceClassification.from_pretrained(EMO_MODEL_NAME).to(device)

# 4) Sentiment model
SENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
sent_tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL_NAME)
sent_model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL_NAME).to(device)

# Emotion labels for mapping
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


# HELPERS
def softmax_np(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()



# STRICT RULES — INSULTS, PROFANITY, POLITE, FRIENDLY


INSULTS = [
    "stupid", "idiot", "dumb", "incompetent", "useless",
    "trash", "garbage", "worthless", "pathetic", "clown",
    "moron", "failure", "shut up", "hate you"
]

PROFANITY = [
    "fuck", "shit", "bitch", "asshole", "bastard",
    "motherfucker", "prick", "dickhead"
]

POLITE = [
    "please", "thank you", "thanks", "would you mind",
    "if possible", "kindly", "when you have a chance"
]

FRIENDLY = [
    "awesome", "amazing", "great job", "fantastic",
    "love this", "appreciate you", "good vibes",
    "wonderful", "you're the best"
]



# UPGRADED THREAT DETECTION V1.5


THREATS = [
    # Direct physical threats
    "hurt you", "harm you", "kill you", "beat you", "attack you",
    "destroy you", "ruin you", "break you", "smash you",
    "you're dead", "you are dead", "dead man",

    # Variations of “I’m going to/gonna”
    "i'm gonna hurt you", "i'm going to hurt you",
    "i am going to hurt you", "im going to hurt you",
    "i will hurt you", "i'll hurt you",
    "im gonna hurt you", "i am gonna hurt you",

    # Consequence threats
    "you will regret", "you'll regret",
    "you will pay", "you'll pay", "make you pay",
    "there will be consequences",

    # Implied threats
    "watch your back", "better watch out",
    "last warning", "final warning",
    "this is your last warning",
    "see what happens",
    "try me and see what happens",
    "don't test me", "don’t push me",

    # Coercive threats
    "fix this or else", "or else"
]

# Regex patterns to catch flexible phrasing
THREAT_PATTERNS = [
    r"\bi[' ]?m\s+gonna\s+(hurt|harm|kill|beat)\b",
    r"\bi[' ]?m\s+going\s+to\s+(hurt|harm|kill|beat)\b",
    r"\bi\s+will\s+(hurt|harm|kill|beat)\b",
    r"\byou\s+(will|gonna|going to)\s+(pay|regret)\b",
    r"\bwatch\s+your\s+back\b",
    r"\byou('re|r)?\s+dead\b",
    r"\b(last|final)\s+warning\b",
    r"\bsee\s+what\s+happens\b",
]

# Stem detection to catch:
# hurting you / hurt u / hurt ya / gon hurt u / etc.
THREAT_STEMS = ["hurt", "harm", "kill", "beat", "attack", "destroy", "ruin"]



# MAIN CLASSIFIER

def classify_tone_rich(text: str):

    lowered = text.lower().strip()
    explanation = []

    
    # 1) Toxicity
    
    tox_inputs = tox_tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        tox_logits = tox_model(**tox_inputs).logits[0].cpu().numpy()
    toxic_score = softmax_np(tox_logits)[1]

    
    # 2) Offensive model
    
    off_inputs = off_tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        off_logits = off_model(**off_inputs).logits[0].cpu().numpy()
    off_probs = softmax_np(off_logits)
    offensive_score = off_probs[-1]

    
    # 3) Sentiment
    
    sent_inputs = sent_tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        sent_logits = sent_model(**sent_inputs).logits[0].cpu().numpy()
    sent_probs = softmax_np(sent_logits)
    pos_score = sent_probs[1]
    neg_score = sent_probs[0]

    
    # 4) Emotions (Anger / Joy)
    
    emo_inputs = emo_tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        emo_logits = emo_model(**emo_inputs).logits[0].cpu().numpy()
    emo_probs = softmax_np(emo_logits)
    emo_anger = emo_probs[0]
    emo_joy = emo_probs[3]

    
    # RULE FLAGS
    
    has_insult = any(w in lowered for w in INSULTS)
    has_profanity = any(w in lowered for w in PROFANITY)
    has_polite = any(w in lowered for w in POLITE)
    has_friendly = any(w in lowered for w in FRIENDLY)
    has_sarcasm = bool(re.search(r"sure|yeah right|as if", lowered))

    # THREAT FLAGS
    has_kw_threat = any(w in lowered for w in THREATS)
    has_regex_threat = any(re.search(p, lowered) for p in THREAT_PATTERNS)
    has_stem_threat = any(stem in lowered for stem in THREAT_STEMS)

    # ML-based threat (to catch implied threats)
    ml_threat = (toxic_score > 0.45 and neg_score > 0.55)

    threat_detected = (
        has_kw_threat or
        has_regex_threat or
        has_stem_threat or
        ml_threat
    )

    
    # THREAT OVERRIDE
    
    if threat_detected:
        explanation.append("Threat override activated.")
        return {
            "label": "Aggressive",
            "confidence": 97,
            "severity": 95,
            "threat_score": 98,
            "politeness_score": 0,
            "friendly_score": 0,
            "aggressive_prob": 0.98,
            "positive_prob": pos_score,
            "has_threat": True,
            "has_profanity": has_profanity,
            "has_sarcasm": has_sarcasm,
            "explanation": explanation
        }

    # PROFANITY OVERRIDE
    
    if has_profanity:
        return {
            "label": "Aggressive",
            "confidence": 90,
            "severity": 90,
            "threat_score": 0,
            "politeness_score": 0,
            "friendly_score": 0,
            "aggressive_prob": 0.90,
            "positive_prob": pos_score,
            "has_threat": False,
            "has_profanity": True,
            "has_sarcasm": has_sarcasm,
            "explanation": explanation
        }

    
    # INSULT OVERRIDE
    
    if has_insult:
        return {
            "label": "Aggressive",
            "confidence": 88,
            "severity": 88,
            "threat_score": 0,
            "politeness_score": 0,
            "friendly_score": 0,
            "aggressive_prob": 0.88,
            "positive_prob": pos_score,
            "has_threat": False,
            "has_profanity": has_profanity,
            "has_sarcasm": has_sarcasm,
            "explanation": explanation
        }

    
    # FRIENDLY
    
    if has_friendly and pos_score > 0.60:
        return {
            "label": "Friendly",
            "confidence": int(pos_score * 100),
            "severity": 0,
            "threat_score": 0,
            "politeness_score": 0,
            "friendly_score": int(pos_score * 100),
            "positive_prob": pos_score,
            "aggressive_prob": 0.0,
            "has_threat": False,
            "has_profanity": False,
            "has_sarcasm": has_sarcasm,
            "explanation": explanation
        }

    
    # POLITE
    
    if has_polite and pos_score > 0.50:
        return {
            "label": "Polite",
            "confidence": int(pos_score * 100),
            "severity": 0,
            "threat_score": 0,
            "politeness_score": int(pos_score * 100),
            "friendly_score": 0,
            "aggressive_prob": 0.0,
            "positive_prob": pos_score,
            "has_threat": False,
            "has_profanity": False,
            "has_sarcasm": has_sarcasm,
            "explanation": explanation
        }

    
    # NEUTRAL (default)
    
    return {
        "label": "Neutral",
        "confidence": int((1 - neg_score) * 100),
        "severity": 0,
        "threat_score": 0,
        "politeness_score": int(pos_score * 100),
        "friendly_score": int(pos_score * 100),
        "aggressive_prob": 0.0,
        "positive_prob": pos_score,
        "has_threat": False,
        "has_profanity": False,
        "has_sarcasm": has_sarcasm,
        "explanation": explanation
    }




def classify_tone(text: str):
    r = classify_tone_rich(text)
    return r["label"], r["confidence"], r["aggressive_prob"], r["positive_prob"]










