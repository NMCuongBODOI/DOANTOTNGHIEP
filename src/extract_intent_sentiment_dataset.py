"""
extract_intent_sentiment_dataset.py
Trích xuất và chuẩn hóa các câu bình luận ngắn từ Zomato Reviews
phục vụ bài toán huấn luyện AI phân loại Ý định (Intent) và Cảm xúc (Sentiment) của khách hàng.
"""

import os
import sys
import re
import json
import pandas as pd
from collections import Counter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Định nghĩa từ khóa nhận diện Intent
INTENT_RULES = {
    "Recommend_Positive": [
        "must try", "highly recommend", "must visit", "definitely recommend", 
        "loved it", "best place", "awesome experience", "worth visiting", "superb"
    ],
    "Complain_Negative": [
        "worst experience", "never go again", "waste of money", "horrible", 
        "terrible", "disappointed", "poor quality", "avoid this place", "declined"
    ],
    "Food_Taste": [
        "delicious", "tasty", "spicy", "sweet", "flavor", "biryani", "chicken", 
        "bland", "overcooked", "undercooked", "crispy", "dessert", "sauce", "fresh"
    ],
    "Pricing_Value": [
        "pocket friendly", "expensive", "overpriced", "cheap", "costly", 
        "value for money", "reasonable price", "bill", "worth the price"
    ],
    "Ambience_Vibe": [
        "ambience", "atmosphere", "music", "decor", "vibe", "rooftop", 
        "seating", "romantic", "peaceful", "crowded", "interior", "lighting"
    ],
    "Service_Staff": [
        "service was slow", "quick service", "staff is polite", "waiter", 
        "rude staff", "courteous", "delayed", "attentive", "hospitable"
    ],
    "Reservation_Booking": [
        "table booking", "reserve a table", "waiting time", "queue", 
        "prior reservation", "line up", "seated"
    ]
}


def determine_sentiment(rating_val):
    try:
        r = float(rating_val)
        if r >= 4.0:
            return "Positive"
        elif r <= 2.5:
            return "Negative"
        else:
            return "Neutral"
    except:
        return "Neutral"


def extract_sentences(input_csv, output_csv, min_words=4, max_words=25, max_samples=5000):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Không tìm thấy file: {input_csv}")

    df = pd.read_csv(input_csv)
    df_valid = df.dropna(subset=['Review', 'Rating'])
    
    extracted_records = []
    intent_stats = Counter()
    sentiment_stats = Counter()

    for _, row in df_valid.iterrows():
        text = str(row['Review']).strip()
        rating = row['Rating']
        restaurant = str(row.get('Restaurant', '')).strip()
        sentiment = determine_sentiment(rating)

        # Tách câu theo dấu câu kết thúc
        sentences = re.split(r'[.!?\n\r]+', text)
        for s in sentences:
            s_clean = re.sub(r'\s+', ' ', s).strip()
            words = s_clean.split()
            word_count = len(words)

            if min_words <= word_count <= max_words:
                s_lower = s_clean.lower()
                matched_intents = []

                for intent, keywords in INTENT_RULES.items():
                    if any(kw in s_lower for kw in keywords):
                        matched_intents.append(intent)

                # Nếu câu có ý định rõ ràng hoặc có cảm xúc mạnh
                primary_intent = matched_intents[0] if matched_intents else "General_Feedback"
                
                extracted_records.append({
                    "sentence": s_clean,
                    "word_count": word_count,
                    "restaurant": restaurant,
                    "rating": rating,
                    "sentiment": sentiment,
                    "primary_intent": primary_intent,
                    "all_intents": ", ".join(matched_intents) if matched_intents else "None"
                })

                intent_stats[primary_intent] += 1
                sentiment_stats[sentiment] += 1

                if len(extracted_records) >= max_samples:
                    break
        if len(extracted_records) >= max_samples:
            break

    df_out = pd.DataFrame(extracted_records)
    df_out.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print("=" * 60)
    print("🎯 TRÍCH XUẤT TẬP DỮ LIỆU Ý ĐỊNH & CẢM XÚC THÀNH CÔNG")
    print("=" * 60)
    print(f"Tổng số câu bình luận ngắn xuất bản: {len(df_out)}")
    print(f"Đường dẫn lưu: {output_csv}")
    print("\n📊 Phân bố Ý định (Intent Distribution):")
    for intent, count in intent_stats.most_common():
        pct = (count / len(df_out)) * 100
        print(f"  • {intent:<22}: {count:>5} câu ({pct:.1f}%)")

    print("\n📊 Phân bố Cảm xúc (Sentiment Distribution):")
    for sent, count in sentiment_stats.most_common():
        pct = (count / len(df_out)) * 100
        print(f"  • {sent:<22}: {count:>5} câu ({pct:.1f}%)")

    return df_out


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = os.path.join(base_dir, "Zomato_Dataset", "zomato_reviews.csv")
    out_path = os.path.join(base_dir, "Zomato_Dataset", "intent_sentiment_extracted_5k.csv")
    extract_sentences(in_path, out_path, max_samples=5000)
