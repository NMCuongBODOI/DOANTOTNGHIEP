"""
data_loader_zomato.py
Module nạp và tiền xử lý bộ dữ liệu Zomato Reviews & Restaurant Metadata cho các mô hình RecSys (CB, MF, LightGCN, SimGCL, RAG).
"""

import os
import sys
import math
import random
from collections import defaultdict
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


class DataLoaderZomato:
    def __init__(self, dataset_dir=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, min_rating=3.0, seed=42):
        if dataset_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dataset_dir = os.path.join(base_dir, "Zomato_Dataset")
        self.dataset_dir = dataset_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_rating = min_rating
        self.seed = seed

        self.user2id = {}
        self.id2user = {}
        self.item2id = {}
        self.id2item = {}
        self.item_texts = defaultdict(list)
        self.restaurant_meta = {}  # item_id -> dict info

    def load_file(self, filename):
        filepath = os.path.join(self.dataset_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file {filename} tại {self.dataset_dir}")

        pairs = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                raw_u = int(parts[0]) if parts[0].isdigit() else parts[0]
                if raw_u not in self.user2id:
                    u = len(self.user2id)
                    self.user2id[raw_u] = u
                    self.id2user[u] = raw_u
                else:
                    u = self.user2id[raw_u]

                if len(parts) >= 2:
                    raw_i = int(parts[1]) if parts[1].isdigit() else parts[1]
                    if raw_i not in self.item2id:
                        i = len(self.item2id)
                        self.item2id[raw_i] = i
                        self.id2item[i] = raw_i
                    else:
                        i = self.item2id[raw_i]
                    pairs.append((u, i))
        return pairs

    def load_metadata_and_texts(self):
        """Đọc file metadata và các câu review để trích xuất đặc trưng văn bản & thực thể."""
        meta_file = os.path.join(self.dataset_dir, "zomato_metadata.csv")
        reviews_file = os.path.join(self.dataset_dir, "zomato_reviews.csv")

        import pandas as pd
        if os.path.exists(meta_file):
            try:
                df_meta = pd.read_csv(meta_file)
                for _, row in df_meta.iterrows():
                    name = str(row.get('Name', '')).strip()
                    if not name:
                        continue
                    meta_info = {
                        "name": name,
                        "cuisines": str(row.get('Cuisines', '')),
                        "cost": str(row.get('Cost', '')),
                        "collections": str(row.get('Collections', '')),
                        "timings": str(row.get('Timings', ''))
                    }
                    if name in self.item2id:
                        i_id = self.item2id[name]
                        self.restaurant_meta[i_id] = meta_info
                        txt = f"{meta_info['cuisines']} {meta_info['collections']}"
                        self.item_texts[i_id].append(txt.lower())
            except Exception as e:
                print(f"[Zomato Loader] Cảnh báo đọc metadata: {e}")

        if os.path.exists(reviews_file):
            try:
                df_rev = pd.read_csv(reviews_file)
                for _, row in df_rev.iterrows():
                    rest_name = str(row.get('Restaurant', '')).strip()
                    rev_txt = str(row.get('Review', '')).strip()
                    if rest_name in self.item2id and rev_txt:
                        i_id = self.item2id[rest_name]
                        self.item_texts[i_id].append(rev_txt.lower())
            except Exception as e:
                print(f"[Zomato Loader] Cảnh báo đọc reviews: {e}")

    def build_tfidf_features(self, num_items):
        """Xây dựng ma trận TF-IDF 50 chiều biểu diễn nhà hàng/món ăn từ đánh giá & thể loại."""
        word_counts = defaultdict(int)
        for i_id in range(num_items):
            full_txt = " ".join(self.item_texts.get(i_id, []))
            words = set(w for w in full_txt.split() if len(w) > 3 and w.isalpha())
            for w in words:
                word_counts[w] += 1

        top_words = sorted(word_counts.keys(), key=lambda w: word_counts[w], reverse=True)[:50]
        if not top_words:
            top_words = ["food", "service", "ambience", "biryani", "chicken", "taste", "place", "good", "chinese", "dining"]

        word2id = {w: idx for idx, w in enumerate(top_words)}
        num_words = len(top_words)

        tfidf_matrix = np.zeros((num_items, num_words), dtype=np.float32)
        df = defaultdict(int)
        for i_id in range(num_items):
            full_txt = " ".join(self.item_texts.get(i_id, []))
            words = set(w for w in full_txt.split() if w in word2id)
            for w in words:
                df[w] += 1

        for i_id in range(num_items):
            full_txt = " ".join(self.item_texts.get(i_id, []))
            words = [w for w in full_txt.split() if w in word2id]
            if not words:
                continue
            tf = 1.0 / len(words)
            for w in words:
                w_idx = word2id[w]
                idf = math.log((num_items + 1.0) / (df[w] + 1.0)) + 1.0
                tfidf_matrix[i_id, w_idx] = tf * idf

        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tfidf_matrix = tfidf_matrix / norms
        return tfidf_matrix, top_words

    def prepare_data(self):
        train_file = os.path.join(self.dataset_dir, "train.txt")
        test_file = os.path.join(self.dataset_dir, "test.txt")

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(f"Không tìm thấy train.txt hoặc test.txt trong {self.dataset_dir}")

        train_pairs = self.load_file("train.txt")
        test_pairs = self.load_file("test.txt")

        num_users = len(self.user2id)
        num_items = len(self.item2id)

        # Đọc bổ sung metadata và review text nếu có
        self.load_metadata_and_texts()

        train_user_items = defaultdict(set)
        test_user_items = defaultdict(set)
        val_user_items = defaultdict(set)
        val_pairs = []

        for u, i in train_pairs:
            train_user_items[u].add(i)

        for u, i in test_pairs:
            test_user_items[u].add(i)

        edge_index = [[], []]
        for u, i in train_pairs:
            item_node_id = num_users + i
            edge_index[0].append(u)
            edge_index[1].append(item_node_id)
            edge_index[0].append(item_node_id)
            edge_index[1].append(u)

        tfidf_features, genre_list = self.build_tfidf_features(num_items)

        return {
            "num_users": num_users,
            "num_items": num_items,
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
            "test_pairs": test_pairs,
            "train_user_items": train_user_items,
            "val_user_items": val_user_items,
            "test_user_items": test_user_items,
            "edge_index": edge_index,
            "tfidf_features": tfidf_features,
            "genre_list": genre_list,
            "user2id": self.user2id,
            "item2id": self.item2id,
            "id2user": self.id2user,
            "id2item": self.id2item,
            "restaurant_meta": self.restaurant_meta
        }


if __name__ == "__main__":
    loader = DataLoaderZomato()
    data = loader.prepare_data()
    print("=== NẠP DỮ LIỆU ZOMATO THÀNH CÔNG ===")
    print(f"Users: {data['num_users']}")
    print(f"Items: {data['num_items']}")
    print(f"Train pairs: {len(data['train_pairs'])}")
    print(f"Test pairs: {len(data['test_pairs'])}")
    print(f"TF-IDF shape: {data['tfidf_features'].shape}")
    print(f"Features: {data['genre_list'][:10]}")
