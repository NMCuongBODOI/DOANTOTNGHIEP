"""
src/intent_classifier.py
================================================================================
Module Phân loại Ý định (Intent) và Cảm xúc (Sentiment) Đa Ngôn Ngữ (Vi - En)
cho Trợ lý Hội thoại Ẩm thực Cá nhân hóa (PCRS Restaurant Assistant).

Được xây dựng và chuẩn hóa dựa trên tập dữ liệu Zomato Reviews (10,000 đánh giá & 41,600+ nhà hàng).
Hỗ trợ nhận diện 7 nhóm ý định và 3 trạng thái cảm xúc, phục vụ phân tuyến (Intent Routing) cho RAG.
================================================================================
"""

import os
import sys
import re
from typing import Dict, Any, List, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Bảng từ khóa Ý định (Hỗ trợ song ngữ Tiếng Việt & Tiếng Anh)
INTENT_PATTERNS = {
    "Complain_Negative": [
        # Tiếng Việt
        r"tệ quá", r"chán quá", r"thất vọng", r"dở ẹc", r"quá dở", r"chờ lâu", r"đợi lâu",
        r"thái độ kém", r"bẩn quá", r"không ngon", r"phục vụ chậm", r"bức xúc", r"chặt chém",
        r"tanh", r"sống nhăn", r"ngộ độc", r"chê", r"không hài lòng", r"tồi tệ",
        # English
        r"worst", r"terrible", r"horrible", r"disappointed", r"poor quality", r"waste of money",
        r"rude", r"uncooked", r"declined", r"avoid this place", r"never come back", r"bad service"
    ],
    "Recommend_Positive": [
        # Tiếng Việt
        r"món nào ngon nhất", r"best seller", r"được khen nhiều", r"món đỉnh nhất", r"đặc sản",
        r"rất ngon", r"tuyệt vời", r"xuất sắc", r"đáng thử", r"nên thử", r"quán ruột", r"cháy hàng",
        # English
        r"must try", r"highly recommend", r"must visit", r"best food", r"delicious", r"awesome",
        r"signature dish", r"top rated", r"favorite", r"loved it", r"superb"
    ],
    "Pricing_Value": [
        # Tiếng Việt
        r"giá cả", r"bao nhiêu tiền", r"có đắt không", r"hạt dẻ", r"bình dân", r"tiết kiệm",
        r"đắt quá", r"rẻ", r"vừa túi tiền", r"hợp túi tiền", r"khoảng giá", r"chi phí",
        r"ngân sách", r"hóa đơn", r"khuyến mãi", r"giảm giá",
        # English
        r"price", r"cost", r"expensive", r"cheap", r"pocket friendly", r"value for money",
        r"budget", r"overpriced", r"bill", r"reasonable price", r"discount"
    ],
    "Ambience_Vibe": [
        # Tiếng Việt
        r"không gian", r"hẹn hò", r"lãng mạn", r"ấm cúng", r"yên tĩnh", r"view đẹp",
        r"sống ảo", r"nhạc sống", r"tiệc gia đình", r"bàn đôi", r"thoáng mát", r"rooftop",
        r"ngoài trời", r"ồn ào", r"đông đúc", r"check in",
        # English
        r"ambience", r"atmosphere", r"vibe", r"romantic", r"cozy", r"quiet", r"view",
        r"music", r"decor", r"seating", r"rooftop", r"crowded", r"aesthetic"
    ],
    "Service_Staff": [
        # Tiếng Việt
        r"phục vụ", r"nhân viên", r"order", r"thái độ", r"nhiệt tình", r"chu đáo",
        r"chờ bao lâu", r"mang ra nhanh", r"gọi món", r"chăm sóc",
        # English
        r"service", r"waiter", r"staff", r"polite", r"courteous", r"attentive",
        r"slow service", r"fast service", r"hospitable", r"welcoming"
    ],
    "Reservation_Booking": [
        # Tiếng Việt
        r"đặt bàn", r"giữ chỗ", r"còn bàn không", r"bàn mấy người", r"đặt trước",
        r"đặt chỗ", r"đến lúc", r"xếp hàng",
        # English
        r"book a table", r"reserve a table", r"reservation", r"waiting line", r"queue",
        r"table for", r"booking"
    ],
    "Food_Taste": [
        # Tiếng Việt
        r"món gà", r"món bò", r"hải sản", r"ít cay", r"không cay", r"cay nhiều",
        r"ngọt", r"thanh đạm", r"chua cay", r"ít calo", r"giảm cân", r"dinh dưỡng",
        r"salad", r"súp", r"nướng", r"lẩu", r"chay", r"đạm cao", r"protein",
        # English
        r"chicken", r"beef", r"spicy", r"sweet", r"sour", r"crispy", r"soup",
        r"salad", r"calories", r"low cal", r"healthy", r"vegan", r"vegetarian", r"flavor"
    ]
}

SENTIMENT_PATTERNS = {
    "Positive": [
        r"ngon", r"tuyệt", r"thích", r"hài lòng", r"tốt", r"đẹp", r"chu đáo", r"ưng ý", r"đỉnh",
        r"good", r"great", r"delicious", r"love", r"perfect", r"nice", r"excellent", r"amazing", r"best"
    ],
    "Negative": [
        r"tệ", r"chán", r"dở", r"bẩn", r"lâu", r"chậm", r"đắt", r"thất vọng", r"chém", r"hỏng",
        r"bad", r"worst", r"terrible", r"slow", r"horrible", r"poor", r"expensive", r"rude", r"hate"
    ]
}


class IntentSentimentClassifier:
    """Bộ phân loại Ý định (Intent) và Cảm xúc (Sentiment) dựa trên biểu thức quy tắc & ngữ nghĩa."""
    
    def __init__(self):
        # Biên dịch regex sẵn để tối ưu tốc độ (<0.1ms)
        self.intent_compiled = {
            intent: [re.compile(rf"\b{pat}", re.IGNORECASE) for pat in patterns]
            for intent, patterns in INTENT_PATTERNS.items()
        }
        self.sentiment_compiled = {
            sent: [re.compile(rf"\b{pat}", re.IGNORECASE) for pat in patterns]
            for sent, patterns in SENTIMENT_PATTERNS.items()
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Phân tích câu đầu vào của thực khách.
        Trả về:
          - primary_intent: Ý định cốt lõi chi phối câu hỏi
          - all_intents: Danh sách tất cả ý định phát hiện được
          - sentiment: Trạng thái cảm xúc (Positive / Negative / Neutral)
          - routing_action: Hành động kiến nghị cho RAG pipeline
          - confidence: Điểm tin cậy ước tính
        """
        text_clean = text.strip()
        matched_intents = []

        for intent, regex_list in self.intent_compiled.items():
            hit_count = 0
            for r in regex_list:
                if r.search(text_clean):
                    hit_count += 1
            if hit_count > 0:
                matched_intents.append((intent, hit_count))

        # Sắp xếp theo số lượng từ khóa khớp
        matched_intents.sort(key=lambda x: -x[1])
        
        primary_intent = matched_intents[0][0] if matched_intents else "Food_Taste"
        all_intents_list = [it[0] for it in matched_intents] if matched_intents else ["Food_Taste"]

        # Phân tích Cảm xúc
        pos_hits = sum(1 for r in self.sentiment_compiled["Positive"] if r.search(text_clean))
        neg_hits = sum(1 for r in self.sentiment_compiled["Negative"] if r.search(text_clean))

        if primary_intent == "Complain_Negative" or neg_hits > pos_hits:
            sentiment = "Negative"
        elif primary_intent == "Recommend_Positive" or pos_hits > neg_hits:
            sentiment = "Positive"
        else:
            sentiment = "Neutral"

        # Định tuyến hành vi (Action Routing)
        routing_action = self._determine_action(primary_intent, sentiment)

        return {
            "text": text_clean,
            "primary_intent": primary_intent,
            "all_intents": all_intents_list,
            "sentiment": sentiment,
            "routing_action": routing_action,
            "confidence": 0.95 if matched_intents else 0.70
        }

    def _determine_action(self, intent: str, sentiment: str) -> str:
        """Quyết định luồng xử lý RAG tối ưu dựa trên Intent & Sentiment."""
        if intent == "Complain_Negative" or sentiment == "Negative":
            return "HANDLE_COMPLAINT_AND_APOLOGIZE"
        elif intent == "Pricing_Value":
            return "RETRIEVE_BUDGET_AND_COST"
        elif intent == "Ambience_Vibe":
            return "RETRIEVE_AMBIENCE_AND_SETTING"
        elif intent == "Reservation_Booking":
            return "TRIGGER_TABLE_BOOKING_FLOW"
        elif intent == "Recommend_Positive":
            return "RETRIEVE_TOP_RATED_AND_DISH_LIKED"
        elif intent == "Service_Staff":
            return "RETRIEVE_SERVICE_FEEDBACK"
        else:
            return "RETRIEVE_NUTRITION_AND_GNN_RECOMMEND"


if __name__ == "__main__":
    clf = IntentSentimentClassifier()
    test_sentences = [
        "Quán này phục vụ chậm quá, tôi đợi hơn nửa tiếng rồi!",
        "Có món nào ngon rẻ cho học sinh sinh viên không em?",
        "Tư vấn cho anh món gà ít calo thanh đạm sau khi tập gym",
        "Tôi muốn tìm chỗ ngồi lãng mạn cho 2 người vào tối thứ 7",
        "Món mutton biryani ở đây có phải món ngon nhất không?",
        "Cho tôi đặt bàn 4 người lúc 19h tối nay nhé"
    ]
    print("=== KIỂM THỬ BỘ PHÂN LOẠI Ý ĐỊNH & CẢM XÚC (ZOMATO INTENT ROUTING) ===")
    for s in test_sentences:
        res = clf.predict(s)
        print(f"\n💬 Câu: \"{s}\"")
        print(f"   🎯 Intent: {res['primary_intent']:<22} | Cảm xúc: {res['sentiment']}")
        print(f"   ⚡ Routing Action: {res['routing_action']}")
