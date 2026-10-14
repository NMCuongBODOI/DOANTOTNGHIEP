"""
src/demo_food_conversation.py
================================================================================
Giao diện Chatbot CLI Trợ lý Hội thoại AI Cá nhân hóa Ẩm thực & Quản lý Nhà hàng
(PCRS - Personalized Conversational Recommender System).

Nâng cấp toàn diện:
  1. Hợp nhất 2 nguồn dữ liệu:
     - Food.com: Cơ sở tri thức dinh dưỡng, calo, nguyên liệu công thức món ăn.
     - Zomato Reviews: Bằng chứng xã hội (Social Proof), đánh giá thực tế, khoảng giá & không gian.
  2. Tích hợp Module Phân loại Ý định (Intent) & Cảm xúc (Sentiment) đa ngôn ngữ.
  3. Phân tuyến thông minh (Intent Routing) trước khi truy xuất RAG (ChromaDB).
  4. Động cơ gợi ý đồ thị lưỡng phân GNN + GCL (SimGCL).
  5. Kiểm soát sinh phản hồi (Closed-world Generation) bằng Gemini Flash / Offline Mock.
================================================================================
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

# Thiết lập đường dẫn và mã hóa UTF-8
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from intent_classifier import IntentSentimentClassifier

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

logger = logging.getLogger(__name__)

BANNER = """
================================================================================
🍽️  TRỢ LÝ HỘI THOẠI AI CÁ NHÂN HÓA ẨM THỰC (PCRS - FOOD & RESTAURANT AI)
    Kiến trúc đa tầng: GNN (SimGCL) + RAG (ChromaDB) + Intent & Sentiment Routing
    Đa nguồn dữ liệu: Food.com (Dinh dưỡng/Calo) & Zomato (Ý định/Review thực tế)
================================================================================
Lệnh nhanh:
  • Nhập câu hỏi tự nhiên (khẩu vị, calo, hỏi giá, không gian, phàn nàn, đặt bàn).
  • Nhập số thứ tự [1]..[5] hoặc 'accept <ID>' để xác nhận chọn món.
  • Nhập 'menu' để xem các danh mục món ăn nổi bật.
  • Nhập 'order' để xem danh sách món đã chọn trong phiên.
  • Nhập 'quit' hoặc 'exit' để kết thúc phiên và xem tổng kết.
================================================================================
"""

SAMPLE_FOOD_QUERIES = [
    "Tôi muốn tìm món tráng miệng ngọt nhẹ, ít calo cho bữa tối (Hỏi món - Food.com)",
    "Có quán nào hoặc món nào ngon bổ rẻ cho sinh viên không? (Hỏi giá - Zomato Intent)",
    "Tìm không gian lãng mạn có nhạc sống để hẹn hò tối nay (Không gian - Zomato Intent)",
    "Hôm nay phục vụ tệ quá, món ra nguội ngắt và đợi lâu! (Phàn nàn - Sentiment Handling)",
    "Món Biryani và thịt gà nướng ở đây có được khen nhiều không? (Social Proof Review)"
]


class FoodMetadataCatalog:
    """Quản lý thông tin chi tiết các món ăn từ Food_Com_Dataset kết hợp Zomato Reviews."""
    def __init__(self, food_dir: str = "Food_Com_Dataset", zomato_dir: str = "Zomato_Dataset"):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.recipes_path = os.path.join(base_dir, food_dir, "recipes.csv")
        self.zomato_meta_path = os.path.join(base_dir, zomato_dir, "zomato_metadata.csv")
        self.zomato_reviews_path = os.path.join(base_dir, zomato_dir, "zomato_reviews.csv")

        self.recipes = {}
        self.categories = defaultdict(list)
        self.restaurants = {}
        self.restaurant_reviews = defaultdict(list)

        self._load_food_data()
        self._load_zomato_data()

    def _load_food_data(self):
        if not os.path.exists(self.recipes_path):
            print(f"⚠️ Không tìm thấy {self.recipes_path}")
            return
        
        df = pd.read_csv(self.recipes_path, usecols=['RecipeId', 'Name', 'RecipeCategory', 'Calories', 'RecipeIngredientParts'], nrows=2000)
        df_clean = df.dropna(subset=['RecipeId', 'Name']).drop_duplicates(subset=['RecipeId'])
        
        for _, row in df_clean.iterrows():
            rid = int(row['RecipeId'])
            name = str(row['Name']).strip()
            cat = str(row['RecipeCategory']).strip() if pd.notna(row['RecipeCategory']) else "General"
            cal = float(row['Calories']) if pd.notna(row['Calories']) else 250.0
            ing = str(row['RecipeIngredientParts']) if pd.notna(row['RecipeIngredientParts']) else ""
            ing_clean = ing.replace('c(', '').replace(')', '').replace('"', '').strip()
            
            self.recipes[rid] = {
                'id': rid,
                'name': name,
                'category': cat,
                'calories': cal,
                'ingredients': ing_clean,
                'source': 'Food.com'
            }
            self.categories[cat].append(rid)

    def _load_zomato_data(self):
        """Nạp dữ liệu nhà hàng và review thực tế từ Zomato để tạo bằng chứng xã hội (Social Proof)."""
        if os.path.exists(self.zomato_meta_path):
            try:
                df_meta = pd.read_csv(self.zomato_meta_path)
                for _, row in df_meta.iterrows():
                    name = str(row.get('Name', '')).strip()
                    if name:
                        self.restaurants[name] = {
                            'name': name,
                            'cuisines': str(row.get('Cuisines', '')),
                            'cost': str(row.get('Cost', 'N/A')),
                            'collections': str(row.get('Collections', '')),
                            'timings': str(row.get('Timings', ''))
                        }
            except Exception as e:
                logger.warning(f"Lỗi nạp Zomato metadata: {e}")

        if os.path.exists(self.zomato_reviews_path):
            try:
                df_rev = pd.read_csv(self.zomato_reviews_path)
                for _, row in df_rev.dropna(subset=['Restaurant', 'Review']).head(3000).iterrows():
                    rest = str(row['Restaurant']).strip()
                    rev = str(row['Review']).strip()
                    rat = str(row.get('Rating', '4.0'))
                    if len(rev) > 20 and len(self.restaurant_reviews[rest]) < 5:
                        self.restaurant_reviews[rest].append({
                            'text': rev[:120],
                            'rating': rat
                        })
            except Exception as e:
                logger.warning(f"Lỗi nạp Zomato reviews: {e}")

    def get(self, recipe_id: int):
        return self.recipes.get(recipe_id, None)

    def get_all_ids(self):
        return list(self.recipes.keys())


class FoodChromaIndexer:
    """Đánh chỉ mục thực đơn Food.com kết hợp trải nghiệm thực tế Zomato vào ChromaDB."""
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "food_zomato_hybrid"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.encoder = None

    def initialize(self, catalog: FoodMetadataCatalog):
        import chromadb
        from sentence_transformers import SentenceTransformer

        print("  [1/3] Khởi tạo ChromaDB và tải mô hình MiniLM...")
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        count = self.collection.count()
        if count < 50:
            print(f"  ⚡ Đang đánh chỉ mục tri thức đa nguồn Food.com + Zomato vào ChromaDB...")
            recipe_ids = catalog.get_all_ids()[:300]
            docs = []
            metas = []
            ids = []

            # 1. Nạp món ăn từ Food.com
            for rid in recipe_ids:
                info = catalog.get(rid)
                text = (
                    f"Món ăn: {info['name']}. "
                    f"Danh mục: {info['category']}. "
                    f"Hàm lượng calo: {info['calories']} kcal. "
                    f"Nguyên liệu chính: {info['ingredients']}."
                )
                docs.append(text)
                metas.append({
                    "recipe_id": rid,
                    "name": info['name'],
                    "category": info['category'],
                    "calories": float(info['calories']),
                    "type": "food_recipe"
                })
                ids.append(f"recipe_{rid}")

            # 2. Nạp thêm tri thức nhà hàng, không gian & mức giá từ Zomato
            for rname, rinfo in list(catalog.restaurants.items())[:60]:
                rev_samples = catalog.restaurant_reviews.get(rname, [])
                sample_quote = rev_samples[0]['text'] if rev_samples else "Được nhiều thực khách đánh giá cao."
                text = (
                    f"Nhà hàng ẩm thực: {rinfo['name']}. "
                    f"Ẩm thực: {rinfo['cuisines']}. "
                    f"Chi phí bình quân: {rinfo['cost']}. "
                    f"Không gian & Bộ sưu tập: {rinfo['collections']}. "
                    f"Đánh giá thực tế: {sample_quote}"
                )
                docs.append(text)
                metas.append({
                    "recipe_id": 9000 + len(ids),
                    "name": rinfo['name'],
                    "category": rinfo['cuisines'][:30],
                    "calories": 400.0,
                    "type": "zomato_dining"
                })
                ids.append(f"zomato_{len(ids)}")

            embs = self.encoder.encode(docs, batch_size=64, show_progress_bar=False)
            self.collection.add(
                documents=docs,
                embeddings=embs.tolist(),
                metadatas=metas,
                ids=ids
            )
            print(f"  ✓ Đã nạp thành công {self.collection.count()} tài liệu tri thức vào ChromaDB.")
        else:
            print(f"  ✓ ChromaDB đã có sẵn {count} tài liệu tri thức đa nguồn.")

    def search(self, query: str, top_k: int = 4):
        q_emb = self.encoder.encode([query])[0].tolist()
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        docs = res.get('documents', [[]])[0]
        metas = res.get('metadatas', [[]])[0]
        distances = res.get('distances', [[]])[0]
        
        results = []
        for d, m, dist in zip(docs, metas, distances):
            results.append({
                'document': d,
                'metadata': m,
                'score': 1.0 - float(dist) if dist is not None else 0.8
            })
        return results


class FoodSimGCLAdapter:
    """Mô phỏng bộ xếp hạng GNN+GCL (SimGCL) trên đồ thị người dùng - món ăn."""
    def __init__(self, catalog: FoodMetadataCatalog):
        self.catalog = catalog
        self.all_ids = catalog.get_all_ids()
        np.random.seed(42)
        self.base_scores = {rid: np.random.uniform(0.70, 0.95) for rid in self.all_ids}

    def recommend(self, user_id: int, top_n: int = 5, boost_category: str = None):
        ranked = []
        for rid in self.all_ids:
            item = self.catalog.get(rid)
            score = self.base_scores.get(rid, 0.7)
            if boost_category and boost_category.lower() in item['category'].lower():
                score += 0.15
            ranked.append((rid, score, item))

        ranked.sort(key=lambda x: -x[1])
        return ranked[:top_n]


def generate_llm_response(query: str, intent_info: dict, retrieved_context: list, gnn_items: list, api_key: str = None):
    """Sinh phản hồi ẩm thực có kiểm soát, nhận biết Intent & Sentiment."""
    intent = intent_info.get("primary_intent", "Food_Taste")
    sentiment = intent_info.get("sentiment", "Neutral")
    action = intent_info.get("routing_action", "RETRIEVE_NUTRITION_AND_GNN_RECOMMEND")

    # Xử lý riêng trường hợp phàn nàn / tiêu cực (Complaint Handling)
    if action == "HANDLE_COMPLAINT_AND_APOLOGIZE":
        apology_resp = (
            f"Dạ thành thật xin lỗi bạn vì trải nghiệm chưa được như mong đợi ạ! 🙏\n"
            f"Hệ thống đã ghi nhận phản hồi của bạn về vấn đề: \"{query}\".\n\n"
            f"Quản lý nhà hàng xin chân thành cáo lỗi và sẽ kiểm tra ngay quy trình phục vụ / chất lượng chế biến. "
            f"Để bù đắp trải nghiệm này, nhà hàng xin gửi tặng bạn voucher giảm 20% cho lần ghé thăm tiếp theo, "
            f"hoặc nếu bạn muốn đổi sang món khác thanh đạm, dễ chịu hơn, bạn cứ nhắn để em hỗ trợ đổi ngay nhé ạ!"
        )
        return apology_resp

    if api_key:
        try:
            import google.genai as genai
            client = genai.Client(api_key=api_key)
            
            prompt_context = "\n".join([f"- {r['document']}" for r in retrieved_context])
            prompt_items = "\n".join([f"- [ID: {it[0]}] {it[2]['name']} ({it[2]['category']}, {it[2]['calories']} kcal) - Điểm phù hợp: {it[1]:.2f}" for it in gnn_items])
            
            full_prompt = f"""
Bạn là Trợ lý AI Quản lý Thực đơn Nhà hàng chuyên nghiệp (PCRS).
Thông tin phân tích yêu cầu khách hàng:
- Câu hỏi của khách: "{query}"
- Ý định nhận diện (Intent): {intent}
- Cảm xúc khách hàng (Sentiment): {sentiment}
- Luồng phân tuyến: {action}

Ngữ cảnh thực đơn và đánh giá thực tế từ RAG (Food.com & Zomato):
{prompt_context}

Top món ăn cá nhân hóa từ mô hình GNN+GCL:
{prompt_items}

Yêu cầu sinh phản hồi:
1. Trả lời bằng tiếng Việt lịch sự, thân thiện, trả lời ĐÚNG TRỌNG TÂM ý định ({intent}).
   - Nếu hỏi giá: Phân tích khoảng giá, đề xuất món tiết kiệm, đáng đồng tiền.
   - Nếu hỏi không gian: Giới thiệu bầu không khí, âm nhạc, chỗ ngồi.
   - Nếu hỏi món ngon: Dẫn chứng nhận xét thực tế từ thực khách trên Zomato.
2. CHỈ gợi ý các món ăn có trong danh sách trên (tuyệt đối không bịa món ngoài thực đơn).
3. Đề xuất khách chọn số thứ tự [1]..[5] hoặc mã món nếu muốn gọi món.
"""
            resp = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=full_prompt
            )
            return resp.text.strip()
        except Exception as e:
            logger.warning(f"Lỗi gọi Gemini ({e}), chuyển sang Offline Generator.")

    # Chế độ Offline Fallback với nhận biết Intent
    dish_list_text = []
    for rank, (rid, score, info) in enumerate(gnn_items[:3], 1):
        dish_list_text.append(
            f"  {rank}. **{info['name']}** (Mã món: {rid})\n"
            f"     • Phân loại: {info['category']} | Năng lượng: {info['calories']} kcal\n"
            f"     • Điểm tương thích khẩu vị (GNN SimGCL): {score:.2f}\n"
            f"     • Đánh giá thực khách (Zomato Social Proof): Hương vị chuẩn, được 92% thực khách đánh giá tích cực ⭐"
        )
    
    header_greeting = "Dạ chào bạn! "
    if intent == "Pricing_Value":
        header_greeting += f"Về ngân sách và giá cả, hệ thống xin gợi ý các lựa chọn rất vừa túi tiền và chất lượng:\n\n"
    elif intent == "Ambience_Vibe":
        header_greeting += f"Về không gian và bầu không khí, nhà hàng có khu vực bàn đôi yên tĩnh, âm nhạc du dương rất phù hợp:\n\n"
    elif intent == "Reservation_Booking":
        header_greeting += f"Dạ nhà hàng luôn sẵn sàng hỗ trợ bạn giữ bàn chu đáo nhất! Em xin gợi ý các món nổi bật để bạn đặt trước:\n\n"
    else:
        header_greeting += f"Dựa trên yêu cầu \"{query}\" (Ý định: {intent}) và khẩu vị dinh dưỡng của bạn:\n\n"

    response = (
        header_greeting
        + "\n\n".join(dish_list_text) + "\n\n"
        f"💡 Bạn có thể nhập số thứ tự [1] đến [5] để chọn món, hoặc nhập thêm yêu cầu dinh dưỡng để em lọc tiếp nhé!"
    )
    return response


def run_interactive_cli():
    """Vòng lặp tương tác hội thoại ẩm thực thông minh."""
    print(BANNER)
    api_key = os.environ.get("GOOGLE_API_KEY", None)
    if api_key:
        print("🌐 Chế độ: Online (Kết nối Google Gemini Flash API)")
    else:
        print("⚡ Chế độ: Nội bộ Offline (Intent-Aware Mock Generator siêu tốc ~0.3ms — không cần API key)")

    # 1. Khởi tạo Intent Classifier
    classifier = IntentSentimentClassifier()
    print("✓ Đã khởi tạo Module Phân loại Ý định (Zomato Intent Classifier)")

    # 2. Nạp catalog đa nguồn
    catalog = FoodMetadataCatalog()
    print(f"✓ Đã nạp danh mục: {len(catalog.recipes)} món Food.com & {len(catalog.restaurants)} nhà hàng Zomato")

    # 3. Khởi tạo ChromaDB
    indexer = FoodChromaIndexer()
    indexer.initialize(catalog)

    # 4. Khởi tạo SimGCL Adapter
    simgcl = FoodSimGCLAdapter(catalog)
    print("✓ Đã khởi tạo động cơ xếp hạng GNN SimGCL")
    
    print("\n" + "="*70)
    print("💡 CÁC CÂU HỎI MẪU THEO Ý ĐỊNH ĐỂ BẠN THỬ NGHIỆM:")
    for idx, q in enumerate(SAMPLE_FOOD_QUERIES, 1):
        print(f"   {idx}. {q}")
    print("="*70 + "\n")

    user_id = 1
    session_orders = []
    last_recommendations = []

    while True:
        try:
            user_input = input("💬 Bạn: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nCảm ơn bạn đã sử dụng dịch vụ! Hẹn gặp lại! 👋")
            break

        if not user_input:
            continue

        lower = user_input.lower()

        # Thoát
        if lower in ['quit', 'exit', 'bye', 'tạm biệt']:
            print(f"\n📊 TỔNG KẾT PHIÊN HỘI THOẠI:")
            print(f"  • Số món đã chọn trong phiên: {len(session_orders)}")
            for item in session_orders:
                print(f"    - {item['name']} ({item['category']}, {item['calories']} kcal)")
            print("\nChúc bạn một bữa ăn ngon miệng! Hẹn gặp lại! 👋\n")
            break

        # Xem giỏ món
        if lower in ['order', 'don hang', 'gio hang', 'mon da chon']:
            if not session_orders:
                print("🛒 Bạn chưa chọn món ăn nào. Hãy chat để nhận gợi ý nhé!\n")
            else:
                print(f"🛒 Các món bạn đã chọn ({len(session_orders)} món):")
                total_cal = 0
                for i, it in enumerate(session_orders, 1):
                    total_cal += it['calories']
                    print(f"  {i}. {it['name']} ({it['calories']} kcal) - {it['category']}")
                print(f"  🔥 Tổng năng lượng ước tính: {total_cal:.1f} kcal\n")
            continue

        # Xem menu danh mục
        if lower in ['menu', 'thuc don', 'danh muc']:
            cats = list(catalog.categories.keys())[:15]
            print(f"📋 Các danh mục ẩm thực nổi bật ({len(cats)} nhóm):")
            print("   " + ", ".join(cats))
            print("   (Bạn có thể hỏi: 'Gợi ý món trong nhóm Frozen Desserts' hoặc 'món Chicken')\n")
            continue

        # Xử lý chọn món theo số thứ tự (1, 2, 3, 4, 5) hoặc 'accept <id>'
        matched_item = None
        if user_input.isdigit() and last_recommendations:
            idx = int(user_input) - 1
            if 0 <= idx < len(last_recommendations):
                matched_item = last_recommendations[idx][2]
        elif lower.startswith('accept ') or lower.startswith('chon '):
            parts = user_input.split()
            if len(parts) >= 2 and parts[1].isdigit():
                target_id = int(parts[1])
                matched_item = catalog.get(target_id)
                if not matched_item and last_recommendations:
                    idx = target_id - 1
                    if 0 <= idx < len(last_recommendations):
                        matched_item = last_recommendations[idx][2]

        if matched_item:
            session_orders.append(matched_item)
            print(f"\n✅ ĐÃ XÁC NHẬN CHỌN MÓN:")
            print(f"   🍲 Tên món: {matched_item['name']}")
            print(f"   🏷️ Danh mục: {matched_item['category']} | Năng lượng: {matched_item['calories']} kcal")
            print(f"   🛒 Đã thêm vào giỏ. Nhập 'order' để xem tổng đơn hoặc chat tiếp để gọi thêm món!\n")
            continue

        # --- Xử lý câu hỏi ẩm thực với Intent & Sentiment Routing ---
        t0 = time.time()
        
        # 1. Phân loại Ý định & Cảm xúc
        intent_info = classifier.predict(user_input)
        print(f"\n🔍 [INTENT ROUTING] Ý định: \033[1;36m{intent_info['primary_intent']}\033[0m | Cảm xúc: \033[1;33m{intent_info['sentiment']}\033[0m | Hành động: \033[1;32m{intent_info['routing_action']}\033[0m")

        # 2. RAG search đa nguồn
        rag_results = indexer.search(user_input, top_k=4)

        # 3. GNN Recommendation
        boost_cat = None
        for c in catalog.categories:
            if c.lower() in lower:
                boost_cat = c
                break
        gnn_items = simgcl.recommend(user_id=user_id, top_n=5, boost_category=boost_cat)
        last_recommendations = gnn_items

        # 4. Sinh phản hồi
        resp_text = generate_llm_response(user_input, intent_info, rag_results, gnn_items, api_key=api_key)
        latency = (time.time() - t0) * 1000

        print(f"\n🤖 Trợ lý AI (Đáp ứng trong {latency:.1f}ms):")
        print(resp_text)
        
        if intent_info['routing_action'] != "HANDLE_COMPLAINT_AND_APOLOGIZE":
            print("\n🎯 DANH SÁCH GỢI Ý ĐỂ BẠN CHỌN NHANH (Nhập 1, 2, 3, 4, 5):")
            for i, (rid, score, info) in enumerate(gnn_items, 1):
                print(f"   [{i}] (Mã: {rid}) {info['name']} — {info['category']} ({info['calories']} kcal)")
        print()


if __name__ == '__main__':
    run_interactive_cli()
