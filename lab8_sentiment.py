"""
TFB3133/TEB3133 Data Visualization - Lab 8
Qualitative Data and Sentiment Analysis

Run with:
    streamlit run lab8_sentiment.py
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

# ==========================================
# LESSON 1: PREPARE AND CLASSIFY THE COMMENTS
# ==========================================

reviews = [
    "The bottle keeps my water cold all day.",
    "Great size and easy to carry to class.",
    "I love the simple design and sturdy lid.",
    "It leaks in my bag. Very disappointing.",
    "The cap is hard to open and the bottle feels heavy.",
    "Good value, but cleaning the narrow opening is difficult.",
    "Keeps drinks cold, though the paint scratches easily.",
    "The bottle is fine. Nothing special either way.",
    "I expected better insulation for this price.",
    "Lightweight, useful, and easy to wash.",
    "Not bad, but I would prefer a larger capacity.",
    "Excellent bottle. I use it every day.",
    "The lid broke after one week. Poor quality.",
    "It works as described and arrived on time.",
    "Nice color, but it does not keep drinks cold.",
    "Affordable and practical for carrying water.",
    "The bottle is okay; the handle is uncomfortable.",
    "Very happy with the purchase and the quality.",
]

df = pd.DataFrame({"id": [f"R{i:02}" for i in range(1, 19)],
                   "text": reviews})

analyzer = SentimentIntensityAnalyzer()


def label_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive", score
    if score <= -0.05:
        return "Negative", score
    return "Neutral", score


df[["sentiment", "compound"]] = df["text"].apply(
    lambda text: pd.Series(label_sentiment(text)))

# Assignment 4: main topic of each comment (assigned manually by reading it)
topics = {
    "R01": "Insulation",    "R02": "Size & weight", "R03": "Design",
    "R04": "Durability",    "R05": "Lid & cap",     "R06": "Cleaning",
    "R07": "Insulation",    "R08": "General",       "R09": "Insulation",
    "R10": "Size & weight", "R11": "Size & weight", "R12": "General",
    "R13": "Durability",    "R14": "General",       "R15": "Insulation",
    "R16": "Price",         "R17": "Design",        "R18": "General",
}
df["topic"] = df["id"].map(topics)

# ==========================================
# LESSON 2: STREAMLIT DASHBOARD
# ==========================================

st.set_page_config(page_title="Lab 8 | Feedback", layout="wide")
st.title("Product Feedback: A Text Summary")
st.caption("Practice data only. Automated labels need human review.")

ORDER = ["Positive", "Neutral", "Negative"]
COLORS = {"Positive": "#2a9d8f", "Neutral": "#adb5bd", "Negative": "#e76f51"}

left, right = st.columns(2)

with left:
    st.subheader("Comment counts by sentiment")
    counts = df["sentiment"].value_counts().reindex(ORDER, fill_value=0)
    st.bar_chart(counts, horizontal=True,
                 x_label="VADER label", y_label="Number of comments")
    st.caption(f"Total comments: {len(df)}. Labels use VADER compound "
               "thresholds of ±0.05.")

with right:
    st.subheader("Words used in the comments")
    # Assignment 2:
    #  - ADDED "day" as a stop word: it appears twice ("all day", "every day")
    #    but only as a time phrase, so it says nothing about the product.
    #  - RETAINED "drinks"/"drink" (removed them from the stop-word list):
    #    "drinks" appears next to "keep" and "cold", so keeping it makes the
    #    temperature theme ("keep drinks cold") easier for readers to notice.
    stop_words = set(STOPWORDS) | {"bottle", "day"}
    cloud = WordCloud(width=900, height=500, background_color="white",
                      colormap="viridis", stopwords=stop_words,
                      collocations=False).generate(" ".join(df["text"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)
    st.caption(
        "Word size shows how often a word appears after filler words "
        "(and 'bottle', 'day') are removed. Position and colour carry no "
        "meaning, and size ignores context: 'cold' is large because it "
        "appears in both praise ('keeps my water cold') and complaints "
        "('does not keep drinks cold'). With only 18 comments, a word used "
        "twice already looks important."
    )

st.subheader("Review the original comments")
st.dataframe(df[["id", "text", "topic", "sentiment", "compound"]],
             use_container_width=True, hide_index=True)
st.caption("A compound score is a model output, not a confidence rating.")

# ==========================================
# LESSON 3: ASSIGNMENT
# ==========================================

st.divider()
st.header("Lab 8 Assignment")

# ---------- 1. Run and describe ----------
st.subheader("1. Run and describe")
largest = counts.idxmax()
st.markdown(f"""
- The app shows **{len(df)} comments**: **{counts['Positive']} positive**,
  **{counts['Neutral']} neutral** and **{counts['Negative']} negative**.
- **Largest group:** {largest} ({counts[largest]} of {len(df)} comments,
  {counts[largest] / len(df):.0%}).
- **Recurring theme in the word cloud:** temperature. *cold* and *keep* are
  the largest words (3 times each), together with *water* and *drinks*, so
  keeping drinks cold is what customers talk about most.
- **Mixed-tone comment:** R06, *"Good value, but cleaning the narrow opening
  is difficult."* It is labelled **Negative** (compound
  {df.loc[df.id == 'R06', 'compound'].iloc[0]:.3f}). The single label drops
  the praise for value and does not say *what* the problem is (cleaning).
  R15 is the opposite case: *"Nice color, but it does not keep drinks
  cold"* is labelled **Positive** even though the main point is a failure of
  the bottle's key feature.
""")

# ---------- 2. Improve the qualitative view ----------
st.subheader("2. Improve the qualitative view")
st.markdown("""
- **Removed term: "day".** It appears twice ("all day", "every day") but only
  as part of a time phrase, so it was taking up space in the cloud without
  saying anything about the product.
- **Retained term: "drinks".** The original list removed *drinks/drink*.
  When the word is kept, it shows up beside *keep* and *cold*, so readers see
  the phrase "keep drinks cold" and notice temperature as the main theme
  instead of seeing *cold* on its own.
- A caption under the word cloud now explains what word size means and
  what its limits are.
""")

# ---------- 3. Audit sentiment labels ----------
st.subheader("3. Audit of sentiment labels")
audit = pd.DataFrame([
    ("R02", "Reasonable", "Clear praise for size and portability."),
    ("R09", "Incorrect", "A complaint about insulation. The word 'better' "
                         "pushes the score positive; 'expected better' "
                         "actually means disappointed."),
    ("R14", "Reasonable", "Factual, meets expectations. Could be read as "
                          "mildly positive, but Neutral is acceptable."),
    ("R01", "Incorrect", "Keeping water cold all day is the main job of the "
                         "bottle, so this is praise. VADER has no "
                         "sentiment word to pick up."),
    ("R13", "Reasonable", "Lid broke and 'poor quality' - clearly negative."),
    ("R08", "Borderline", "'Fine... nothing special' is lukewarm and "
                          "closer to neutral; scored only slightly "
                          "negative (-0.12)."),
], columns=["id", "manual verdict", "reason"])
audit = audit.merge(df[["id", "text", "sentiment", "compound"]], on="id")
audit = audit[["id", "text", "sentiment", "compound",
               "manual verdict", "reason"]]
st.dataframe(audit, use_container_width=True, hide_index=True)
st.markdown("Two positive (R02, R09), two neutral (R14, R01) and two "
            "negative (R13, R08) labels were checked: **3 reasonable, "
            "1 borderline, 2 incorrect.**")

st.markdown("**Negation test**")
negated = "The bottle is not easy to clean."
test = pd.DataFrame(
    [(t, *label_sentiment(t)) for t in
     ["The bottle is easy to clean.", negated]],
    columns=["text", "sentiment", "compound"])
st.dataframe(test, use_container_width=True, hide_index=True)
st.markdown("VADER handles simple negation correctly here: adding *not* "
            "flips the label from Positive to Negative. It fails on "
            "indirect wording instead (R09, R15), where the negative "
            "meaning comes from context rather than a negation word.")

# ---------- 4. Compare by topic ----------
st.subheader("4. Sentiment by topic")
topic_table = (pd.crosstab(df["topic"], df["sentiment"])
               .reindex(columns=ORDER, fill_value=0))
topic_table["Total"] = topic_table.sum(axis=1)
topic_table = topic_table.sort_values("Total")

fig, ax = plt.subplots(figsize=(8, 4))
left_edge = pd.Series(0, index=topic_table.index)
for label in ORDER:
    ax.barh(topic_table.index, topic_table[label], left=left_edge,
            color=COLORS[label], label=label)
    left_edge += topic_table[label]
ax.set_xlabel("Number of comments")
ax.set_title("VADER sentiment by main topic (n = 18)")
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.get_major_locator().set_params(integer=True)
st.pyplot(fig, clear_figure=True)

st.markdown("""
This sample is **too small for a fair comparison by topic**. Each topic has
only 1 to 4 comments, so one comment changes a topic's picture completely.
Insulation is the most discussed topic (4 comments) and looks mostly
positive, but our audit shows that two of those "positive" labels (R09,
R15) are really complaints. Insulation feedback is actually split 2-2.
Topic results should be treated as hints for what to ask about next, not as
findings.
""")

# ---------- Summary for the product team ----------
st.subheader("Summary for the product team")
st.markdown("""
Across 18 practice comments, the VADER model labelled 9 positive, 3 neutral
and 6 negative. **Finding:** keeping drinks cold matters most to customers.
"Cold" and "keep" are the largest words in the word cloud, and insulation is
the most discussed topic. Reading those comments shows a split: two praise
the bottle for staying cold, while two say the insulation falls short. Lids,
caps and durability are the next most common complaints, including a leak
and a lid that broke after one week.

**Limitation:** this is a small, synthetic sample, and the automated labels
cannot be trusted on their own. In our audit of six comments, two labels
were wrong and one was borderline, mostly because mixed comments ("Nice
color, but...") and polite complaints ("I expected better insulation")
confuse the model.

**Recommendation:** collect a larger set of real reviews with a short
structured form: a 1-5 star rating and a required topic choice (lid,
insulation, cleaning, size, price) next to the free-text comment. The
customer's own rating can then be used to check the sentiment labels, and
topics can be compared fairly.
""")

# ---------- Reflection ----------
st.subheader("Reflection")
st.markdown("""
- **What the visuals revealed:** most comments are labelled positive, and
  temperature (cold / keep / drinks) is the main recurring theme, followed
  by the lid and durability.
- **What could mislead a reader:** the bar chart hides mixed opinions and
  wrong labels (R01, R09, R15), so it overstates how positive the feedback
  is. The word cloud shows frequency, not sentiment: a large "cold" does not
  mean customers are happy with insulation. Colour and position in the cloud
  mean nothing, and with 18 comments, small differences are noise.
- **How to validate:** have two people label a sample by hand and compare
  their labels with the model (agreement rate), check labels against star
  ratings, and look at every comment close to the ±0.05 thresholds. For more
  representative feedback, collect real reviews from several channels and
  time periods, not only from customers who chose to write a comment.
""")

st.caption("TFB3133/TEB3133 Data Visualization - Lab 8 - September 2026")
