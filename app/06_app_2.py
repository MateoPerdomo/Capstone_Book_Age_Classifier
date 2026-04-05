"""
06_app.py — Book Age Classifier Desktop App
Run with: python 06_app.py
Requires: best_model.pkl, scaler.pkl, label_encoder.pkl, feature_cols.pkl
          (all produced by notebook 04)
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import pickle, re, threading
import numpy as np
import pandas as pd
import nltk, textstat
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression

# ── Download NLTK data silently ───────────────────────────────────────────────
for pkg in ['punkt', 'punkt_tab', 'stopwords', 'vader_lexicon']:
    nltk.download(pkg, quiet=True)

# ── Load artefacts ────────────────────────────────────────────────────────────
model        = pickle.load(open('best_model.pkl',    'rb'))
scaler       = pickle.load(open('scaler.pkl',        'rb'))
le           = pickle.load(open('label_encoder.pkl', 'rb'))
feature_cols = pickle.load(open('feature_cols.pkl',  'rb'))

# ── Sensitivity word lists ────────────────────────────────────────────────────
VIOLENCE_WORDS  = set(["kill","killed","killing","killer","murder","murdered",
    "stab","stabbed","shoot","shooting","gun","guns","pistol","rifle","bullet",
    "fight","fought","punch","kick","attack","attacked","assault","beat","beaten",
    "blood","bloody","bleed","bleeding","wound","wounded","injury","injured",
    "death","died","dying","corpse","war","battle","combat","weapon","bomb",
    "explosion","torture","gore","slaughter","massacre","violence","violent",
    "threaten","threat","knife","sword","axe","abuse","abused","victim",
    "horror","terror","terrifying","nightmare","suffer","suffering","execute"])
PROFANITY_WORDS = set(["damn","damned","crap","bastard","bitch","piss","pissed",
    "shit","shitty","bullshit","fuck","fucked","fucker","fucking","motherfucker",
    "asshole","arsehole","wanker","whore","slut"])
ADULT_WORDS     = set(["sex","sexual","sexually","naked","nude","nudity","erotic",
    "orgasm","intercourse","breast","breasts","penis","vagina","genitals","porn",
    "pornography","rape","raped","molest","prostitute","seduce","lust","affair",
    "adultery","sensual","condom","abortion"])
DRUG_WORDS      = set(["alcohol","alcoholic","drunk","drunken","beer","wine",
    "whiskey","vodka","rum","gin","liquor","booze","hangover","cigarette",
    "smoking","tobacco","cocaine","heroin","marijuana","cannabis","weed","meth",
    "ecstasy","overdose","addict","addiction","stoned","opioid","fentanyl"])

_sia        = SentimentIntensityAnalyzer()
_stop_words = set(stopwords.words('english'))

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(text):
    sentences = sent_tokenize(text)
    words     = [w for w in word_tokenize(text.lower()) if w.isalpha()]
    content   = [w for w in words if w not in _stop_words]
    total     = max(len(words), 1)
    scores    = _sia.polarity_scores(text)
    quoted    = re.findall(r'"[^"]*"', text)
    sent_ct   = max(text.count('.') + text.count('!') + text.count('?'), 1)

    return {
        'flesch_reading_ease':  textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'gunning_fog':          textstat.gunning_fog(text),
        'dale_chall':           textstat.dale_chall_readability_score(text),
        'smog_index':           textstat.smog_index(text),
        'avg_sentence_length':  np.mean([len(word_tokenize(s)) for s in sentences]) if sentences else 0,
        'vocab_richness':       len(set(words)) / total,
        'avg_word_length':      np.mean([len(w) for w in words]) if words else 0,
        'long_word_ratio':      sum(1 for w in words if len(w) > 6) / total,
        'num_sentences':        len(sentences),
        'num_words':            len(words),
        'content_word_ratio':   len(content) / total,
        'violence_score':       round(sum(1 for w in words if w in VIOLENCE_WORDS)  * 1000 / total, 4),
        'profanity_score':      round(sum(1 for w in words if w in PROFANITY_WORDS) * 1000 / total, 4),
        'adult_score':          round(sum(1 for w in words if w in ADULT_WORDS)     * 1000 / total, 4),
        'drug_score':           round(sum(1 for w in words if w in DRUG_WORDS)      * 1000 / total, 4),
        'sentiment_positive':   scores['pos'],
        'sentiment_negative':   scores['neg'],
        'sentiment_neutral':    scores['neu'],
        'sentiment_compound':   scores['compound'],
        'dialogue_ratio':       sum(len(q) for q in quoted) / max(len(text), 1),
        'exclamation_ratio':    text.count('!') / sent_ct,
        'question_ratio':       text.count('?') / sent_ct,
    }

# ── LR surrogate (trained once at startup) ───────────────────────────────────
def train_lr_surrogate():
    try:
        df = pd.read_csv('data_features.csv')
        df['age_group'] = df['age_group'].astype(str).str.strip()
        df['age_group'] = df['age_group'].apply(lambda x: '+' + x if not x.startswith('+') else x)
        X = scaler.transform(df[feature_cols].values)
        y = le.transform(df['age_group'].values)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X, y)
        return lr
    except Exception:
        return None

# ── Colour palette ────────────────────────────────────────────────────────────
BG        = '#1a1a2e'
PANEL     = '#16213e'
ACCENT    = '#0f3460'
TEXT      = '#e0e0e0'
MUTED     = '#888888'
LABEL_CLR = {'+4': '#66c2a5', '+10': '#fc8d62', '+12': '#8da0cb', '+18': '#e78ac3'}
BAR_CLR   = {'+4': '#66c2a5', '+10': '#fc8d62', '+12': '#8da0cb', '+18': '#e78ac3'}

# ── Main application ──────────────────────────────────────────────────────────
class BookClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Book Age Classifier')
        self.configure(bg=BG)
        self.geometry('960x780')
        self.resizable(True, True)
        self.minsize(800, 640)

        self.lr_surrogate = None
        self._build_ui()

        # Train surrogate in background so startup is instant
        threading.Thread(target=self._load_surrogate, daemon=True).start()

    def _load_surrogate(self):
        self.lr_surrogate = train_lr_surrogate()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # Title bar
        title_bar = tk.Frame(self, bg=ACCENT, pady=12)
        title_bar.pack(fill='x')
        tk.Label(title_bar, text='📚  Book Age Classifier',
                 font=('Georgia', 18, 'bold'), bg=ACCENT, fg=TEXT).pack()
        tk.Label(title_bar, text='Paste a book excerpt and classify its age suitability',
                 font=('Georgia', 10), bg=ACCENT, fg=MUTED).pack()

        # Main content
        content = tk.Frame(self, bg=BG)
        content.pack(fill='both', expand=True, padx=20, pady=16)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)

        # ── Text input ────────────────────────────────────────────────────────
        tk.Label(content, text='Book Excerpt', font=('Georgia', 11, 'bold'),
                 bg=BG, fg=TEXT, anchor='w').grid(row=0, column=0, sticky='w', pady=(0, 4))

        self.text_input = scrolledtext.ScrolledText(
            content, height=9, wrap='word',
            font=('Georgia', 11), bg=PANEL, fg=TEXT,
            insertbackground=TEXT, relief='flat',
            borderwidth=0, padx=10, pady=8)
        self.text_input.grid(row=1, column=0, sticky='nsew', pady=(0, 10))

        # Classify button
        btn_frame = tk.Frame(content, bg=BG)
        btn_frame.grid(row=2, column=0, sticky='ew', pady=(0, 14))

        self.classify_btn = tk.Button(
            btn_frame, text='  Classify  →',
            font=('Georgia', 12, 'bold'),
            bg='#e94560', fg='white',
            activebackground='#c73652', activeforeground='white',
            relief='flat', padx=20, pady=8, cursor='hand2',
            command=self._on_classify)
        self.classify_btn.pack(side='left')

        self.clear_btn = tk.Button(
            btn_frame, text='Clear',
            font=('Georgia', 10), bg=PANEL, fg=MUTED,
            activebackground=ACCENT, activeforeground=TEXT,
            relief='flat', padx=12, pady=8, cursor='hand2',
            command=self._on_clear)
        self.clear_btn.pack(side='left', padx=(10, 0))

        self.status_lbl = tk.Label(btn_frame, text='', font=('Georgia', 10),
                                   bg=BG, fg=MUTED)
        self.status_lbl.pack(side='left', padx=14)

        # ── Results panel (hidden until first prediction) ─────────────────────
        self.results_frame = tk.Frame(content, bg=BG)
        self.results_frame.grid(row=3, column=0, sticky='nsew')
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.columnconfigure(1, weight=1)
        content.rowconfigure(3, weight=2)

        # Left: prediction + confidence
        left = tk.Frame(self.results_frame, bg=PANEL, padx=16, pady=14)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        left.columnconfigure(0, weight=1)

        tk.Label(left, text='Prediction', font=('Georgia', 10, 'bold'),
                 bg=PANEL, fg=MUTED).pack(anchor='w')

        self.pred_label = tk.Label(left, text='—',
                                   font=('Georgia', 42, 'bold'),
                                   bg=PANEL, fg=TEXT)
        self.pred_label.pack(anchor='w', pady=(2, 0))

        self.pred_desc = tk.Label(left, text='',
                                  font=('Georgia', 11), bg=PANEL, fg=MUTED,
                                  wraplength=280, justify='left')
        self.pred_desc.pack(anchor='w', pady=(0, 14))

        tk.Label(left, text='Confidence', font=('Georgia', 10, 'bold'),
                 bg=PANEL, fg=MUTED).pack(anchor='w')

        self.bar_frame = tk.Frame(left, bg=PANEL)
        self.bar_frame.pack(fill='x', pady=(6, 0))

        # Right: top 5 feature contributions
        right = tk.Frame(self.results_frame, bg=PANEL, padx=16, pady=14)
        right.grid(row=0, column=1, sticky='nsew', padx=(8, 0))
        right.columnconfigure(0, weight=1)

        tk.Label(right, text='Top 5 Feature Contributions',
                 font=('Georgia', 10, 'bold'), bg=PANEL, fg=MUTED).pack(anchor='w')
        tk.Label(right, text='Why did the model predict this?',
                 font=('Georgia', 9), bg=PANEL, fg=MUTED).pack(anchor='w', pady=(0, 10))

        self.contrib_frame = tk.Frame(right, bg=PANEL)
        self.contrib_frame.pack(fill='both', expand=True)

    # ── Descriptions ──────────────────────────────────────────────────────────
    DESCRIPTIONS = {
        '+4':  'Suitable for ages 4 and up.\nSimple vocabulary, short sentences, positive tone.',
        '+10': 'Suitable for ages 10 and up.\nMiddle-grade complexity, adventure and discovery themes.',
        '+12': 'Suitable for ages 12 and up.\nYoung adult themes — identity, emotion, mild conflict.',
        '+18': 'Suitable for ages 18 and up.\nMature content, complex themes, adult language.',
    }

    # ── Event handlers ────────────────────────────────────────────────────────
    def _on_clear(self):
        self.text_input.delete('1.0', 'end')
        self.pred_label.config(text='—', fg=TEXT)
        self.pred_desc.config(text='')
        self.status_lbl.config(text='')
        for w in self.bar_frame.winfo_children():
            w.destroy()
        for w in self.contrib_frame.winfo_children():
            w.destroy()

    def _on_classify(self):
        text = self.text_input.get('1.0', 'end').strip()
        if len(text.split()) < 10:
            self.status_lbl.config(text='⚠  Please enter at least 10 words.', fg='#e94560')
            return
        self.status_lbl.config(text='Classifying…', fg=MUTED)
        self.classify_btn.config(state='disabled')
        threading.Thread(target=self._classify_worker, args=(text,), daemon=True).start()

    def _classify_worker(self, text):
        try:
            feats    = extract_features(text)
            X_raw    = np.array([[feats[c] for c in feature_cols]])
            X_scaled = scaler.transform(X_raw)
            pred_enc = model.predict(X_scaled)[0]
            label    = le.inverse_transform([pred_enc])[0]

            proba = None
            if hasattr(model, 'predict_proba'):
                proba = dict(zip(le.classes_, model.predict_proba(X_scaled)[0]))

            contribs = None
            if self.lr_surrogate is not None:
                ci        = list(le.classes_).index(label)
                coefs     = self.lr_surrogate.coef_[ci]
                contrib   = pd.Series(coefs * X_scaled[0], index=feature_cols)
                top5_idx  = contrib.abs().sort_values(ascending=False).head(5).index
                contribs  = [(f, float(contrib[f]), float(feats[f])) for f in top5_idx]

            self.after(0, self._update_ui, label, proba, contribs)
        except Exception as e:
            self.after(0, lambda: self.status_lbl.config(
                text=f'Error: {e}', fg='#e94560'))
            self.after(0, lambda: self.classify_btn.config(state='normal'))

    def _update_ui(self, label, proba, contribs):
        color = LABEL_CLR.get(label, TEXT)

        # Prediction label
        self.pred_label.config(text=label, fg=color)
        self.pred_desc.config(text=self.DESCRIPTIONS.get(label, ''))

        # Confidence bars
        for w in self.bar_frame.winfo_children():
            w.destroy()

        age_order = ['+4', '+10', '+12', '+18']
        if proba:
            for cls in age_order:
                p      = proba.get(cls, 0)
                is_pred = cls == label
                row = tk.Frame(self.bar_frame, bg=PANEL)
                row.pack(fill='x', pady=3)

                tk.Label(row, text=f'{cls:>4}', width=5,
                         font=('Courier', 10, 'bold' if is_pred else 'normal'),
                         bg=PANEL, fg=color if is_pred else MUTED,
                         anchor='e').pack(side='left')

                track = tk.Frame(row, bg=ACCENT, height=18)
                track.pack(side='left', fill='x', expand=True, padx=(6, 6))
                track.pack_propagate(False)

                # Draw bar using canvas
                canvas = tk.Canvas(track, height=18, bg=ACCENT,
                                   highlightthickness=0, bd=0)
                canvas.pack(fill='both', expand=True)
                canvas.update_idletasks()
                w_px = int(canvas.winfo_width() * p)
                bar_color = BAR_CLR.get(cls, '#888')
                canvas.create_rectangle(0, 0, w_px, 18,
                                        fill=bar_color, outline='')

                tk.Label(row, text=f'{p*100:5.1f}%',
                         font=('Courier', 10, 'bold' if is_pred else 'normal'),
                         bg=PANEL, fg=color if is_pred else MUTED,
                         width=7, anchor='w').pack(side='left')

                if is_pred:
                    tk.Label(row, text='← predicted',
                             font=('Courier', 9), bg=PANEL, fg=color).pack(side='left')
        else:
            tk.Label(self.bar_frame,
                     text='Probability unavailable for this model type.\nRe-train SVM with probability=True in notebook 04.',
                     font=('Georgia', 9), bg=PANEL, fg=MUTED,
                     justify='left').pack(anchor='w')

        # Feature contributions
        for w in self.contrib_frame.winfo_children():
            w.destroy()

        if contribs:
            for feat, val, raw in contribs:
                is_pos = val > 0
                direction = f'▲ toward {label}' if is_pos else f'▼ away from {label}'
                dir_color = '#66c2a5' if is_pos else '#e78ac3'

                row = tk.Frame(self.contrib_frame, bg=PANEL, pady=4)
                row.pack(fill='x')

                tk.Label(row, text=feat.replace('_', ' '),
                         font=('Courier', 9, 'bold'), bg=PANEL, fg=TEXT,
                         anchor='w').pack(fill='x')

                detail = tk.Frame(row, bg=PANEL)
                detail.pack(fill='x')
                tk.Label(detail, text=direction,
                         font=('Courier', 9), bg=PANEL, fg=dir_color,
                         anchor='w').pack(side='left')
                tk.Label(detail,
                         text=f'  contribution={val:+.3f}   raw={raw:.3f}',
                         font=('Courier', 9), bg=PANEL, fg=MUTED,
                         anchor='w').pack(side='left')

                ttk.Separator(self.contrib_frame, orient='horizontal').pack(
                    fill='x', pady=1)
        elif self.lr_surrogate is None:
            tk.Label(self.contrib_frame,
                     text='Loading surrogate model…\nTry classifying again in a moment.',
                     font=('Georgia', 9), bg=PANEL, fg=MUTED,
                     justify='left').pack(anchor='w')
        else:
            tk.Label(self.contrib_frame,
                     text='data_features.csv not found.\nContributions unavailable.',
                     font=('Georgia', 9), bg=PANEL, fg=MUTED,
                     justify='left').pack(anchor='w')

        self.status_lbl.config(text=f'✓  Done  ({len(self.text_input.get("1.0","end").split())} words)',
                                fg='#66c2a5')
        self.classify_btn.config(state='normal')


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = BookClassifierApp()
    app.mainloop()
