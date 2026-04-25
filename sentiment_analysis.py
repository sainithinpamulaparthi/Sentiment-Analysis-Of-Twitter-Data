# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
"""
Advanced Twitter Sentiment Analysis Dashboard
Uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner), 
which is highly optimized for social media text (handles emojis, punctuation, caps, and context).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# ── 1. Engine Setup (VADER) ──────────────────────────────────────────────────
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Silently download the VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER sentiment lexicon (first time only)...")
    nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

# ── 2. High-Quality Dataset ──────────────────────────────────────────────────
# A highly realistic, diverse dataset to test edge cases, punctuation, and emojis.
RAW_DATA = [
    # Technology
    ("I absolutely LOVE the new iPhone!!! The battery life is insane 😍", "Technology"),
    ("The recent software update BRICKED my laptop. Worst experience ever. smh.", "Technology"),
    ("ChatGPT is kinda scary but incredibly useful for coding.", "Technology"),
    ("My wifi has been down for 3 hours. I am losing my mind! 😡", "Technology"),
    ("Just bought a new mechanical keyboard. It clicks nicely.", "Technology"),
    ("Who thought this UI update was a good idea? It is complete garbage, but the backend is fast.", "Technology"),
    ("Apple event was okay. Nothing groundbreaking, just iterative updates.", "Technology"),
    ("Loving the new noise-cancellation headphones. 10/10!", "Technology"),
    ("Tech support took 45 minutes to answer. Unacceptable.", "Technology"),
    ("Python 3.12 is out. Cool I guess.", "Technology"),
    
    # Finance & Crypto
    ("Bitcoin is pumping!!! To the moon! 🚀🌕", "Finance"),
    ("Lost 50% of my portfolio today. I am devastated and ruined.", "Finance"),
    ("Inflation data came out exactly as expected. Market reacting neutrally.", "Finance"),
    ("Why are gas prices so high right now? This is hurting my wallet.", "Finance"),
    ("Just paid off my student loans! Best feeling in the world! 🎉", "Finance"),
    ("Stock market is extremely volatile today. Proceed with caution.", "Finance"),
    ("My bank charged me a $35 overdraft fee for a $2 coffee. I hate them.", "Finance"),
    ("Solid earnings report from Microsoft today. Very promising.", "Finance"),
    ("Taxes are confusing.", "Finance"),
    ("Just diversified my portfolio. Feeling smart and secure.", "Finance"),

    # Sports
    ("WHAT A GAME!!! Unbelievable buzzer beater to win the championship! 🏆", "Sports"),
    ("The referee completely rigged that match. Disgusting bias.", "Sports"),
    ("Arsenal played a decent game, ending in a 1-1 draw.", "Sports"),
    ("My favorite player just tore his ACL. Heartbroken. 😭", "Sports"),
    ("Just finished a 10k run! Feeling exhausted but amazing! 🏃‍♂️", "Sports"),
    ("The new stadium looks fantastic, great atmosphere.", "Sports"),
    ("Ferrari strategy team ruins the race AGAIN. I can't watch this anymore.", "Sports"),
    ("Good defensive play by both teams tonight.", "Sports"),
    ("LeBron is still dominating at his age. Absolute GOAT. 🐐", "Sports"),
    ("Ticket prices for the playoffs are a total scam.", "Sports"),

    # Entertainment
    ("That new Marvel movie was phenomenal! The CGI was top notch ✨", "Entertainment"),
    ("I fell asleep watching the new sci-fi flick. So boring and drawn out.", "Entertainment"),
    ("The season finale of Succession was a masterpiece. Brilliant writing.", "Entertainment"),
    ("Why did they cancel my favorite show? Network executives are clueless.", "Entertainment"),
    ("Listening to the new album. It has some good tracks and some skips.", "Entertainment"),
    ("The concert was ALRIGHT, but the acoustics in that arena are terrible.", "Entertainment"),
    ("Tickets secured for Taylor Swift!!! I AM CRYING 😭❤️", "Entertainment"),
    ("Just watching TV.", "Entertainment"),
    ("The book was much better than the movie. Disappointed.", "Entertainment"),
    ("Incredible performance by the lead actor. Oscar worthy!", "Entertainment"),

    # Health
    ("Feeling so much better today! The medication finally kicked in 💊", "Health"),
    ("This migraine is killing me. Worst pain ever.", "Health"),
    ("Just had my annual checkup. Everything is normal.", "Health"),
    ("Mental health days are so important. Glad I took time to rest.", "Health"),
    ("The healthcare system is entirely broken and completely unaffordable.", "Health"),
    ("Started doing yoga every morning. It's truly life-changing! 🧘‍♀️", "Health"),
    ("Can't shake off this flu. I feel awful.", "Health"),
    ("Ate a salad for lunch. Healthy living.", "Health"),
    ("Hospital wait times are ridiculously long. Waited 6 hours in the ER.", "Health"),
    ("Drinking water and minding my business.", "Health"),

    # Food
    ("This pizza is literally the best thing I've ever eaten in my life 🍕🔥", "Food"),
    ("Food arrived cold and 2 hours late. Never ordering from here again.", "Food"),
    ("Tried the new sushi place. It was okay, a bit overpriced.", "Food"),
    ("I could eat tacos every single day and never get tired of them.", "Food"),
    ("Found a hair in my soup. Absolutely disgusting! 🤢", "Food"),
    ("Baking cookies on a rainy afternoon. The house smells wonderful.", "Food"),
    ("The coffee is too bitter for my taste.", "Food"),
    ("Spicy ramen is exactly what I needed today. Perfect.", "Food"),
    ("I burnt the toast.", "Food"),
    ("Excellent service and magnificent steaks at the steakhouse tonight.", "Food"),

    # Travel
    ("Paris is breathtakingly beautiful. Having the time of my life! ✈️🗼", "Travel"),
    ("Flight cancelled again. Stranded at the airport. I hate this airline.", "Travel"),
    ("The hotel is average, but the location is convenient.", "Travel"),
    ("Lost my luggage! This vacation is turning into a nightmare.", "Travel"),
    ("Just booked my tickets to Japan! So excited! 🇯🇵", "Travel"),
    ("The beach is so relaxing. Pure bliss.", "Travel"),
    ("Public transport in this city is so inefficient and dirty.", "Travel"),
    ("Road trip going smoothly so far.", "Travel"),
    ("The hiking trail had stunning views, highly recommend it.", "Travel"),
    ("Terrible jet lag today. Can barely keep my eyes open.", "Travel"),

    # Politics
    ("The new policy is a fantastic step forward for the community.", "Politics"),
    ("These politicians are all corrupt liars. Disgraceful.", "Politics"),
    ("Voting today. Civic duty complete.", "Politics"),
    ("I am so frustrated by the lack of action on climate change.", "Politics"),
    ("The debate last night was an absolute trainwreck.", "Politics"),
    ("Very inspiring speech by the mayor today.", "Politics"),
    ("Taxes are going up again. Great. Just great. 🙄", "Politics"),
    ("Local council meeting was largely uneventful.", "Politics"),
    ("Proud to see peaceful protests making a real difference.", "Politics"),
    ("The economic policies are completely destroying the middle class.", "Politics")
]

# ── 3. Processing ────────────────────────────────────────────────────────────
def analyze_tweet(text):
    # Get VADER sentiment scores
    scores = sia.polarity_scores(text)
    
    # Compound score [-1 to 1] represents the polarity
    polarity = scores['compound']
    
    # Subjectivity heuristic using VADER:
    # 'neu' is the proportion of text that is neutral. 
    # Therefore, 1 - neu is the proportion of text that carries emotional weight.
    subjectivity = 1.0 - scores['neu']
    
    # Standard thresholding for social media
    if polarity >= 0.05:
        label = "Positive"
    elif polarity <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
        
    return polarity, label, subjectivity

# Build DataFrame with realistic random timestamps
base_date = datetime.now() - timedelta(days=30)
rows = []

random.seed(42) # For reproducible random timestamps
for i, (tweet, topic) in enumerate(RAW_DATA):
    polarity, label, subjectivity = analyze_tweet(tweet)
    
    # Randomize time within the last 30 days
    random_days = random.uniform(0, 30)
    tweet_time = base_date + timedelta(days=random_days)
    
    rows.append({
        "id": i+1, 
        "tweet": tweet, 
        "topic": topic,
        "timestamp": tweet_time,
        "score": round(polarity, 4), 
        "label": label,
        "subjectivity": round(subjectivity, 3)
    })

df = pd.DataFrame(rows)
df = df.sort_values("timestamp").reset_index(drop=True)

# ── 4. Console Output ────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 65)
print("     ADVANCED TWITTER SENTIMENT ANALYSIS (VADER ENGINE)")
print("=" * 65)
print(f"\n  Total tweets analysed : {len(df)}")
print(f"  Date range            : {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
print(f"  Topics covered        : {df['topic'].nunique()} ({', '.join(df['topic'].unique())})")

counts = df["label"].value_counts()
total  = len(df)
print("\n  ── Sentiment Distribution ──────────────────────────")
for lbl in ["Positive", "Neutral", "Negative"]:
    c = counts.get(lbl, 0)
    pct = c / total * 100
    bar = "█" * int(pct / 2.5) # Scaled for wider dataset
    print(f"  {lbl:<10} {c:>3} tweets  ({pct:5.1f}%)  {bar}")

print(f"\n  Average polarity score : {df['score'].mean():+.3f}")
print(f"  Average subjectivity   : {df['subjectivity'].mean():.3f}")

print("\n  ── Top 3 Most Positive Tweets ──────────────────────")
for _, r in df.nlargest(3, "score").iterrows():
    print(f"  [{r['score']:+.3f}] {r['tweet'][:65]}...")

print("\n  ── Top 3 Most Negative Tweets ──────────────────────")
for _, r in df.nsmallest(3, "score").iterrows():
    print(f"  [{r['score']:+.3f}] {r['tweet'][:65]}...")

print("\n  ── Sentiment by Topic ──────────────────────────────")
topic_grp = df.groupby("topic")["score"].mean().sort_values(ascending=False)
for topic, mean_sc in topic_grp.items():
    emoji = "[+]" if mean_sc > 0.05 else ("[-]" if mean_sc < -0.05 else "[~]")
    print(f"  {emoji} {topic:<15} {mean_sc:+.3f}")

# ── 5. Advanced Visualisation ─────────────────────────────────────────────────
plt.style.use('dark_background') # Base style for better cross-platform compatibility
fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor("#0f172a")

COLORS = {"Positive": "#22c55e", "Neutral": "#94a3b8", "Negative": "#ef4444"}
PANEL  = "#1e293b"
TEXT   = "#f1f5f9"

def styled_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.set_title(title, color=TEXT, fontweight="bold", fontsize=12, pad=12)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
        spine.set_linewidth(1.2)
    ax.grid(color="#334155", linestyle='--', linewidth=0.5, alpha=0.5)

# ── Panel 1: Donut Chart ──
ax1 = fig.add_subplot(2, 3, 1)
ax1.set_facecolor(PANEL)
vc = df["label"].value_counts()
wedge_props = {"linewidth": 3, "edgecolor": PANEL, "width": 0.4} # Donut chart style
wedges, texts, autotexts = ax1.pie(
    vc, labels=vc.index, colors=[COLORS[l] for l in vc.index],
    autopct="%1.1f%%", startangle=140, wedgeprops=wedge_props,
    textprops={"color": TEXT, "fontsize": 10, "fontweight": "bold"}
)
for at in autotexts:
    at.set_color("#0f172a")
ax1.set_title("Overall Sentiment Distribution", color=TEXT, fontweight="bold", fontsize=12)

# ── Panel 2: Average Score by Topic ──
ax2 = fig.add_subplot(2, 3, 2)
styled_ax(ax2, "Avg Polarity Score by Topic")
topic_avg = df.groupby("topic")["score"].mean().sort_values()
bar_colors = [COLORS["Positive"] if v >= 0.05 else (COLORS["Negative"] if v <= -0.05 else COLORS["Neutral"]) for v in topic_avg]
bars = ax2.barh(topic_avg.index, topic_avg.values, color=bar_colors, edgecolor="#0f172a", linewidth=1)
ax2.axvline(0, color=TEXT, linewidth=1.5, alpha=0.8)
ax2.set_xlabel("VADER Compound Score", color=TEXT, fontsize=10)

# ── Panel 3: Score Distribution (Histogram) ──
ax3 = fig.add_subplot(2, 3, 3)
styled_ax(ax3, "Polarity Score Histogram")
for lbl, color in COLORS.items():
    data = df[df["label"] == lbl]["score"]
    if not data.empty:
        ax3.hist(data, bins=10, color=color, alpha=0.8, label=lbl, edgecolor=PANEL, linewidth=1.2)
ax3.axvline(0.05, color="#22c55e", linestyle="--", alpha=0.8, linewidth=1.5)
ax3.axvline(-0.05, color="#ef4444", linestyle="--", alpha=0.8, linewidth=1.5)
ax3.set_xlabel("VADER Compound Score", color=TEXT, fontsize=10)
ax3.set_ylabel("Number of Tweets", color=TEXT, fontsize=10)
ax3.legend(fontsize=9, facecolor=PANEL, edgecolor="#334155", labelcolor=TEXT)

# ── Panel 4: Subjectivity vs Polarity ──
ax4 = fig.add_subplot(2, 3, 4)
styled_ax(ax4, "Polarity vs. Subjectivity Engine")
for lbl, grp in df.groupby("label"):
    ax4.scatter(grp["score"], grp["subjectivity"],
                c=COLORS[lbl], label=lbl, alpha=0.85, s=100,
                edgecolors="#0f172a", linewidths=0.8)
ax4.axvline(0, color=TEXT, linewidth=1, linestyle="--", alpha=0.5)
ax4.set_xlabel("Polarity (Negative ↔ Positive)", color=TEXT, fontsize=10)
ax4.set_ylabel("Subjectivity (Objective ↔ Emotional)", color=TEXT, fontsize=10)
ax4.legend(fontsize=9, facecolor=PANEL, edgecolor="#334155", labelcolor=TEXT)

# ── Panel 5: Time Series Analysis ──
ax5 = fig.add_subplot(2, 3, 5)
styled_ax(ax5, "Sentiment Timeline (30 Days)")
# Group by day and calculate mean to smooth the line
daily = df.groupby(df['timestamp'].dt.date)['score'].mean().reset_index()
ax5.plot(daily['timestamp'], daily['score'], color="#38bdf8", linewidth=2.5, marker='o', markersize=6, label="Daily Average")
ax5.fill_between(daily['timestamp'], daily['score'], 0, where=(daily['score'] >= 0), color="#22c55e", alpha=0.2, interpolate=True)
ax5.fill_between(daily['timestamp'], daily['score'], 0, where=(daily['score'] < 0), color="#ef4444", alpha=0.2, interpolate=True)
ax5.axhline(0, color=TEXT, linewidth=1, linestyle="--", alpha=0.5)
ax5.set_xlabel("Date", color=TEXT, fontsize=10)
ax5.set_ylabel("Average Score", color=TEXT, fontsize=10)
plt.setp(ax5.get_xticklabels(), rotation=30, ha="right")

# ── Panel 6: Stacked Bar - Sentiment Mix by Topic ──
ax6 = fig.add_subplot(2, 3, 6)
styled_ax(ax6, "Proportional Sentiment Mix by Topic")
topics = df["topic"].unique()

# Calculate percentages instead of raw counts for better comparison
topic_data = []
for t in topics:
    t_df = df[df["topic"] == t]
    total_t = len(t_df)
    if total_t > 0:
        p = len(t_df[t_df["label"] == "Positive"]) / total_t * 100
        n = len(t_df[t_df["label"] == "Neutral"]) / total_t * 100
        neg = len(t_df[t_df["label"] == "Negative"]) / total_t * 100
        topic_data.append((t, p, n, neg))

# Sort by percentage of positive tweets
topic_data.sort(key=lambda x: x[1]) 
sorted_topics = [x[0] for x in topic_data]
pos_pct = [x[1] for x in topic_data]
neu_pct = [x[2] for x in topic_data]
neg_pct = [x[3] for x in topic_data]

y = np.arange(len(sorted_topics))
ax6.barh(y, pos_pct, color=COLORS["Positive"], label="Positive", edgecolor=PANEL, height=0.7)
ax6.barh(y, neu_pct, left=pos_pct, color=COLORS["Neutral"], label="Neutral", edgecolor=PANEL, height=0.7)
ax6.barh(y, neg_pct, left=[p+n for p,n in zip(pos_pct, neu_pct)], color=COLORS["Negative"], label="Negative", edgecolor=PANEL, height=0.7)

ax6.set_yticks(y)
ax6.set_yticklabels(sorted_topics, fontsize=9)
ax6.set_xlabel("Percentage (%)", color=TEXT, fontsize=10)
ax6.set_xlim(0, 100)
ax6.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9, facecolor=PANEL, edgecolor="#334155", labelcolor=TEXT)

# ── Final Layout & Export ────────────────────────────────────────────────────
fig.suptitle("Advanced AI Sentiment Analysis Dashboard (VADER Engine)", color=TEXT, fontsize=18, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect to leave room for suptitle

img_path = os.path.join(OUT_DIR, "advanced_sentiment_dashboard.png")
csv_path = os.path.join(OUT_DIR, "advanced_sentiment_results.csv")

plt.savefig(img_path, dpi=200, bbox_inches="tight", facecolor="#0f172a")
print(f"\n[+] High-res Dashboard saved : {img_path}")
df.to_csv(csv_path, index=False)
print(f"[+] Dataset Export saved     : {csv_path}")
print("\nDone! Opening dashboard...")

plt.show() # Opens interactive chart window