
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import re, warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.sparse import hstack, csr_matrix

INPUT_CSV   = "comments_translated.csv"  
OUTPUT_CSV  = "tata_predictions.csv"
OUTPUT_PNG  = "tata_model_report.png"

print("=" * 60)
print("  Tata Motors — Purchase Intent Prediction Pipeline")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)
print(f"\n✓ Loaded {len(df):,} comments")

df['text'] = df['translated_text'].fillna(df['original_text']).fillna('').astype(str)

STOPWORDS = {
    'i','me','my','we','our','you','your','he','she','it','they',
    'this','that','is','are','was','were','be','been','have','has',
    'had','do','does','did','will','would','could','should','may',
    'might','a','an','the','and','but','if','or','as','not','so',
    'yet','than','then','too','s','t','re','ll','ve',
    'tata','motors','car','vehicle',
}

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'http\S+|www\S+', '', text)       
    text = re.sub(r'@\w+|#\w+', '', text)            
    text = re.sub(r'[^a-z\s]', ' ', text)            
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)
df['word_count']  = df['clean_text'].apply(lambda x: len(x.split()))
df = df[df['word_count'] >= 2].reset_index(drop=True)
print(f"✓ After cleaning: {len(df):,} comments")

POS_WORDS = {
    'love':3,'amazing':3,'excellent':3,'outstanding':3,'fantastic':3,
    'wonderful':3,'brilliant':3,'superb':3,'perfect':3,'best':3,
    'great':2,'good':2,'awesome':3,'incredible':3,'terrific':2,
    'buy':2,'buying':2,'bought':2,'purchase':2,'purchased':2,
    'booking':2,'booked':2,'book':2,'order':2,'delivery':1,
    'excited':2,'planning':1,'definitely':2,'recommend':3,
    'satisfied':2,'happy':2,'pleased':2,'impressed':2,
    'worth':2,'reliable':2,'trust':2,'comfortable':2,'smooth':2,
    'powerful':2,'stylish':2,'beautiful':2,'stunning':2,'premium':2,
    'affordable':2,'efficient':2,'safe':2,'safety':2,'value':1,
}

NEG_WORDS = {
    'worst':3,'terrible':3,'horrible':3,'awful':3,'pathetic':3,
    'disgusting':3,'useless':3,'fraud':3,'cheat':3,'scam':3,
    'disappointed':2,'disappointing':2,'bad':2,'poor':2,
    'issue':2,'problem':2,'defect':3,'defective':3,'fault':2,
    'broken':2,'failed':2,'failure':2,'breakdown':3,'repair':2,
    'complaint':2,'hate':3,'angry':2,'frustrated':2,
    'ridiculous':2,'unacceptable':3,'delay':2,'rude':2,
    'unsafe':3,'overpriced':2,'cancel':2,'cancelled':2,
    'stuck':2,'stranded':2,'never':1,
}

INTENSIFIERS = {
    'very':1.5,'extremely':2.0,'so':1.3,'really':1.3,
    'absolutely':1.8,'highly':1.5,'super':1.5,'totally':1.3,
}
NEGATORS = {
    'not','no','never','dont','doesnt','didnt','wont','cant','cannot',
}

def sentiment_score(text):
    tokens = str(text).lower().split()
    score  = 0
    for i, word in enumerate(tokens):
        w    = re.sub(r'[^a-z]', '', word)
        mult = 1.0
        # negation window: look 3 words back
        for j in range(max(0, i - 3), i):
            if re.sub(r'[^a-z]', '', tokens[j]) in NEGATORS:
                mult = -1.0
                break
        # intensifier: look 1 word back
        if i > 0:
            prev = re.sub(r'[^a-z]', '', tokens[i - 1])
            if prev in INTENSIFIERS:
                mult *= INTENSIFIERS[prev]
        if w in POS_WORDS:
            score += POS_WORDS[w] * mult
        elif w in NEG_WORDS:
            score -= NEG_WORDS[w]
    return round(score / max(len(tokens) ** 0.5, 1), 4)

df['sentiment_score'] = df['text'].apply(sentiment_score)
df['sentiment_label'] = pd.cut(
    df['sentiment_score'],
    bins=[-999, -0.3, 0.3, 999],
    labels=['Negative', 'Neutral', 'Positive']
)

print("\nSentiment distribution:")
print(df['sentiment_label'].value_counts().to_string())

HIGH_INTENT_PATTERNS = [
    r'\bbuy\b', r'\bbuying\b', r'\bbought\b',
    r'\bpurchase\b', r'\bpurchased\b',
    r'\bbook(ing|ed)?\b', r'\border(ed)?\b',
    r'\bown(ing|er)?\b', r'\btest drive\b',
    r'\bshowroom\b', r'\bdealer\b', r'\bdelivery\b',
    r'\bwaiting\b', r'\bupgrade\b', r'\bdream car\b',
    r'\bprice\b', r'\bcost\b', r'\bon road\b',
    r'\bvariant\b', r'\bspec\b', r'\bemi\b', r'\bloan\b',
]

LOW_INTENT_PATTERNS = [
    r'\bworst\b', r'\bnever buy\b', r"\bdon.t buy\b",
    r'\bwont buy\b', r'\bavoid\b', r'\bscam\b',
    r'\bfraud\b', r'\bcheat\b', r'\brefund\b',
    r'\bcancell?(ed|ing)?\b',
]

INTENT_MAP = {0: 'No Intent', 1: 'Moderate Intent', 2: 'High Intent'}

def label_purchase_intent(row):
    text = str(row['text']).lower()
    sent = row['sentiment_score']

    if any(re.search(p, text) for p in LOW_INTENT_PATTERNS):
        return 0                                         
    if any(re.search(p, text) for p in HIGH_INTENT_PATTERNS) and sent >= -0.2:
        return 2                                          
    if any(re.search(p, text) for p in HIGH_INTENT_PATTERNS) and sent < -0.2:
        return 0                                          
    if sent > 0.5:
        return 1                                          
    if sent < -0.5:
        return 0                                         
    return 1                                              

df['purchase_intent']       = df.apply(label_purchase_intent, axis=1)
df['purchase_intent_label'] = df['purchase_intent'].map(INTENT_MAP)

print("\nPurchase Intent distribution:")
print(df['purchase_intent_label'].value_counts().to_string())

PRICE_KW    = ['price','cost','budget','affordable','expensive','emi','loan','lakh']
SERVICE_KW  = ['service','dealer','showroom','repair','maintenance','center','technician']
PRODUCT_KW  = ['design','feature','engine','performance','mileage','safety','interior','comfort']
INTENT_KW   = ['buy','booking','booked','purchase','order','delivery','waiting','own','upgrade']
COMPLAINT_KW= ['issue','problem','complaint','fault','broken','defect','worst','poor','bad','failed']
PRAISE_KW   = ['love','great','best','amazing','excellent','perfect','good','awesome','recommend','satisfied']
MONTH_MAP   = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
               'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}

def make_handcrafted_features(df):
    t = df['text'].str.lower()
    feats = pd.DataFrame({
        'word_count':       df['clean_text'].apply(lambda x: len(x.split())),
        'char_count':       df['text'].apply(len),
        'sentiment_score':  df['sentiment_score'],
        'sentiment_pos':    df['sentiment_score'].clip(lower=0),
        'sentiment_neg':    df['sentiment_score'].clip(upper=0).abs(),
        'is_positive':      (df['sentiment_label'] == 'Positive').astype(int),
        'is_negative':      (df['sentiment_label'] == 'Negative').astype(int),
        'is_neutral':       (df['sentiment_label'] == 'Neutral').astype(int),
        'was_translated':   df['was_translated'].astype(int),
        'has_price_kw':     t.apply(lambda x: int(any(k in x for k in PRICE_KW))),
        'has_service_kw':   t.apply(lambda x: int(any(k in x for k in SERVICE_KW))),
        'has_product_kw':   t.apply(lambda x: int(any(k in x for k in PRODUCT_KW))),
        'has_intent_kw':    t.apply(lambda x: int(any(k in x for k in INTENT_KW))),
        'has_complaint_kw': t.apply(lambda x: int(any(k in x for k in COMPLAINT_KW))),
        'has_praise_kw':    t.apply(lambda x: int(any(k in x for k in PRAISE_KW))),
        'exclamation':      df['text'].apply(lambda x: x.count('!')),
        'question':         df['text'].apply(lambda x: x.count('?')),
        'month_num':        df['month'].map(MONTH_MAP).fillna(0),
    })
    return feats

hand_feats = make_handcrafted_features(df)

# TF-IDF (unigrams + bigrams)
tfidf   = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
X_tfidf = tfidf.fit_transform(df['clean_text'])
X_hand  = csr_matrix(hand_feats.values.astype(float))
X       = hstack([X_tfidf, X_hand])
y       = df['purchase_intent'].values

print(f"\n✓ Feature matrix: {X.shape}  "
      f"(TF-IDF: {X_tfidf.shape[1]}, Handcrafted: {X_hand.shape[1]})")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

MODELS = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, C=1.0, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight='balanced',
        random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
TARGET_NAMES = ['No Intent', 'Moderate Intent', 'High Intent']

print("\n" + "=" * 60)
print("  MODEL RESULTS")
print("=" * 60)

for name, model in MODELS.items():
    print(f"\n{'─'*40}\n  {name}\n{'─'*40}")
    model.fit(X_train, y_train)
    y_pred    = model.predict(X_test)
    acc       = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(y_test, y_pred, target_names=TARGET_NAMES, output_dict=True)

    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  CV  Accuracy  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))

    results[name] = {
        'model':   model,
        'acc':     acc,
        'cv_mean': cv_scores.mean(),
        'cv_std':  cv_scores.std(),
        'cm':      cm,
        'report':  report,
        'y_pred':  y_pred,
    }

best_name  = max(results, key=lambda k: results[k]['cv_mean'])
best_model = results[best_name]['model']
print(f"\n{'='*60}")
print(f"  BEST MODEL: {best_name}  (CV acc = {results[best_name]['cv_mean']:.4f})")
print(f"{'='*60}")

df['predicted_intent']       = best_model.predict(X)
df['predicted_intent_label'] = df['predicted_intent'].map(INTENT_MAP)

out_cols = ['id','month','original_text','translated_text','was_translated',
            'sentiment_score','sentiment_label','purchase_intent_label','predicted_intent_label']
df[out_cols].to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Predictions saved → {OUTPUT_CSV}")

BG    = '#0F172A'
CARD  = '#1E293B'
TEXT  = '#F1F5F9'
MUTED = '#94A3B8'
ACCENT= '#6366F1'
COLORS = {
    'No Intent':       '#EF4444',
    'Moderate Intent': '#F59E0B',
    'High Intent':     '#10B981',
    'Negative':        '#EF4444',
    'Neutral':         '#94A3B8',
    'Positive':        '#10B981',
}

fig = plt.figure(figsize=(22, 26), facecolor=BG)
fig.suptitle('Tata Motors — Purchase Intention Prediction\nSentiment Analysis on Instagram Comments',
             fontsize=22, fontweight='bold', color=TEXT, y=0.98)
gs  = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35,
               left=0.06, right=0.97, top=0.94, bottom=0.04)

def card_ax(ax, title):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values(): spine.set_edgecolor('#334155')
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=10)

ax1 = fig.add_subplot(gs[0, 0])
card_ax(ax1, 'Sentiment Distribution')
sc   = df['sentiment_label'].value_counts()
bars = ax1.bar(sc.index, sc.values, color=[COLORS[l] for l in sc.index],
               width=0.5, edgecolor='none')
for bar, val in zip(bars, sc.values):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
             f'{val:,}', ha='center', color=TEXT, fontsize=9, fontweight='bold')
ax1.set_ylabel('Count', color=MUTED, fontsize=9)
ax1.set_ylim(0, sc.max()*1.18)

ax2 = fig.add_subplot(gs[0, 1])
card_ax(ax2, 'Purchase Intent Distribution')
ic   = df['purchase_intent_label'].value_counts()
wedges, _, autotexts = ax2.pie(
    ic.values, labels=None, autopct='%1.1f%%',
    colors=[COLORS[l] for l in ic.index],
    startangle=140, pctdistance=0.75,
    wedgeprops={'edgecolor': BG, 'linewidth': 2}
)
for at in autotexts:
    at.set_color(TEXT); at.set_fontsize(9); at.set_fontweight('bold')
ax2.legend(ic.index, loc='lower center', bbox_to_anchor=(0.5, -0.08),
           ncol=1, fontsize=8, labelcolor=TEXT, facecolor=CARD, edgecolor='none')

ax3 = fig.add_subplot(gs[0, 2])
card_ax(ax3, 'Sentiment × Purchase Intent (%)')
cross     = pd.crosstab(df['sentiment_label'], df['purchase_intent_label'])
cross     = cross.reindex(columns=['No Intent','Moderate Intent','High Intent'], fill_value=0)
cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
im = ax3.imshow(cross_pct.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
ax3.set_xticks(range(3))
ax3.set_xticklabels(['No\nIntent','Moderate\nIntent','High\nIntent'], color=TEXT, fontsize=8)
ax3.set_yticks(range(len(cross_pct)))
ax3.set_yticklabels(cross_pct.index, color=TEXT, fontsize=9)
for i in range(len(cross_pct)):
    for j in range(3):
        v = cross_pct.values[i, j]
        ax3.text(j, i, f'{v:.1f}%', ha='center', va='center',
                 color='black' if v > 50 else TEXT, fontsize=9, fontweight='bold')
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04).ax.tick_params(colors=MUTED)

# ── Row 1: Accuracy bars / Confusion matrix / F1 per class
ax4 = fig.add_subplot(gs[1, 0])
card_ax(ax4, 'Model Accuracy Comparison')
names = list(results.keys())
short_names = ['Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting']
accs  = [results[n]['acc']     for n in names]
cvs   = [results[n]['cv_mean'] for n in names]
stds  = [results[n]['cv_std']  for n in names]
x     = np.arange(len(names))
b1 = ax4.bar(x-0.2, accs, 0.35, label='Test Acc',  color=ACCENT,    alpha=0.85, edgecolor='none')
b2 = ax4.bar(x+0.2, cvs,  0.35, label='CV Acc',    color='#06B6D4', alpha=0.85, edgecolor='none')
ax4.errorbar(x+0.2, cvs, yerr=stds, fmt='none', color=TEXT, capsize=4, linewidth=1.5)
ax4.set_xticks(x); ax4.set_xticklabels(short_names, color=TEXT, fontsize=8)
ax4.set_ylim(0.75, 1.05)
ax4.set_ylabel('Accuracy', color=MUTED, fontsize=9)
ax4.legend(fontsize=8, labelcolor=TEXT, facecolor=CARD, edgecolor='none')
for bar, val in zip(b1, accs):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
             f'{val:.3f}', ha='center', color=TEXT, fontsize=8, fontweight='bold')
for bar, val in zip(b2, cvs):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
             f'{val:.3f}', ha='center', color=TEXT, fontsize=8, fontweight='bold')

ax5 = fig.add_subplot(gs[1, 1])
card_ax(ax5, f'Confusion Matrix — {best_name}')
cm      = results[best_name]['cm']
cm_pct  = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
im2     = ax5.imshow(cm_pct, cmap='Blues', aspect='auto', vmin=0, vmax=100)
labels  = ['No\nIntent', 'Moderate\nIntent', 'High\nIntent']
ax5.set_xticks(range(3)); ax5.set_xticklabels(labels, color=TEXT, fontsize=8)
ax5.set_yticks(range(3)); ax5.set_yticklabels(labels, color=TEXT, fontsize=8)
ax5.set_xlabel('Predicted', color=MUTED, fontsize=9)
ax5.set_ylabel('Actual',    color=MUTED, fontsize=9)
for i in range(3):
    for j in range(3):
        ax5.text(j, i, f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)',
                 ha='center', va='center',
                 color='white' if cm_pct[i, j] > 50 else TEXT,
                 fontsize=8, fontweight='bold')

ax6 = fig.add_subplot(gs[1, 2])
card_ax(ax6, f'Precision / Recall / F1 — {best_name}')
report  = results[best_name]['report']
x       = np.arange(3)
offsets = [-0.25, 0, 0.25]
cols    = ['#6366F1', '#06B6D4', '#10B981']
for metric, off, col in zip(['precision','recall','f1-score'], offsets, cols):
    vals = [report[c][metric] for c in TARGET_NAMES]
    ax6.bar(x+off, vals, 0.23, label=metric.capitalize(), color=col, alpha=0.85, edgecolor='none')
ax6.set_xticks(x); ax6.set_xticklabels(['No\nIntent','Moderate\nIntent','High\nIntent'],color=TEXT,fontsize=8)
ax6.set_ylim(0, 1.1)
ax6.set_ylabel('Score', color=MUTED, fontsize=9)
ax6.legend(fontsize=8, labelcolor=TEXT, facecolor=CARD, edgecolor='none')
ax6.axhline(0.9, color='white', linestyle='--', alpha=0.2, linewidth=1)

# ── Row 2: Feature importance / Intent by month
ax7 = fig.add_subplot(gs[2, :2])
card_ax(ax7, f'Top 20 Feature Importances — {best_name}')
feat_names  = tfidf.get_feature_names_out().tolist() + hand_feats.columns.tolist()
importances = best_model.feature_importances_
top_idx     = np.argsort(importances)[::-1][:20]
top_feats   = [feat_names[i] for i in top_idx]
top_vals    = [importances[i] for i in top_idx]
bar_colors  = ['#10B981' if '_' in top_feats[i] else ACCENT for i in range(len(top_feats))]
ax7.barh(range(len(top_feats))[::-1], top_vals, color=bar_colors, edgecolor='none', alpha=0.85)
ax7.set_yticks(range(len(top_feats))[::-1])
ax7.set_yticklabels(top_feats, color=TEXT, fontsize=9)
ax7.set_xlabel('Importance Score', color=MUTED, fontsize=9)
ax7.legend(handles=[
    mpatches.Patch(color=ACCENT,    label='TF-IDF Token'),
    mpatches.Patch(color='#10B981', label='Engineered Feature'),
], fontsize=8, labelcolor=TEXT, facecolor=CARD, edgecolor='none')

ax8 = fig.add_subplot(gs[2, 2])
card_ax(ax8, 'Purchase Intent by Month')
month_order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
monthly     = (df.groupby('month')['predicted_intent_label']
               .value_counts(normalize=True).unstack(fill_value=0) * 100)
monthly     = monthly.reindex([m for m in month_order if m in monthly.index])
bottom      = np.zeros(len(monthly))
for intent, color in [('No Intent','#EF4444'),('Moderate Intent','#F59E0B'),('High Intent','#10B981')]:
    if intent in monthly.columns:
        vals = monthly[intent].values
        ax8.bar(range(len(monthly)), vals, bottom=bottom, color=color,
                label=intent, edgecolor='none', alpha=0.9)
        bottom += vals
ax8.set_xticks(range(len(monthly)))
ax8.set_xticklabels([m.capitalize() for m in monthly.index], color=TEXT, fontsize=7, rotation=45)
ax8.set_ylabel('% of Comments', color=MUTED, fontsize=9)
ax8.legend(fontsize=7, labelcolor=TEXT, facecolor=CARD, edgecolor='none', loc='upper left')

ax9 = fig.add_subplot(gs[3, 0])
card_ax(ax9, 'Sentiment Score Distribution by Intent')
for label, color in [('No Intent','#EF4444'),('Moderate Intent','#F59E0B'),('High Intent','#10B981')]:
    scores = df.loc[df['predicted_intent_label']==label, 'sentiment_score'].clip(-5, 5)
    ax9.hist(scores, bins=60, alpha=0.6, color=color, label=label, density=True)
ax9.axvline(0, color='white', linewidth=1, linestyle='--', alpha=0.4)
ax9.set_xlabel('Sentiment Score', color=MUTED, fontsize=9)
ax9.set_ylabel('Density',         color=MUTED, fontsize=9)
ax9.legend(fontsize=8, labelcolor=TEXT, facecolor=CARD, edgecolor='none')

ax10 = fig.add_subplot(gs[3, 1:])
ax10.set_facecolor(CARD)
for spine in ax10.spines.values(): spine.set_edgecolor('#334155')
ax10.set_xlim(0, 1); ax10.set_ylim(0, 1); ax10.axis('off')
ax10.set_title('Model Performance Summary', color=TEXT, fontsize=11, fontweight='bold', pad=10)
headers = ['Model','Test Acc','CV Acc ± Std','F1 (No Intent)','F1 (Moderate)','F1 (High)']
col_x   = [0.02, 0.20, 0.36, 0.54, 0.70, 0.86]
for h, cx in zip(headers, col_x):
    ax10.text(cx, 0.88, h, color=MUTED, fontsize=8, fontweight='bold', transform=ax10.transAxes)
for i, (name, res) in enumerate(results.items()):
    y_pos     = 0.72 - i * 0.2
    rep       = res['report']
    is_best   = (name == best_name)
    row_color = '#10B981' if is_best else TEXT
    star      = ' ★' if is_best else ''
    vals = [
        name + star,
        f"{res['acc']:.4f}",
        f"{res['cv_mean']:.4f} ± {res['cv_std']:.4f}",
        f"{rep['No Intent']['f1-score']:.4f}",
        f"{rep['Moderate Intent']['f1-score']:.4f}",
        f"{rep['High Intent']['f1-score']:.4f}",
    ]
    for cx, v in zip(col_x, vals):
        ax10.text(cx, y_pos, v, color=row_color, fontsize=9,
                  fontweight='bold' if is_best else 'normal',
                  transform=ax10.transAxes)

plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"✓ Dashboard saved → {OUTPUT_PNG}")
print(f"\n✅ Pipeline complete!\n")
