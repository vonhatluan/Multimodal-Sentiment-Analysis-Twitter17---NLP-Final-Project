import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Đường dẫn tới model
model_path = "./my_fast_roberta_model"

# Kiểm tra xem model có tồn tại không
if not os.path.exists(model_path):
    print(f"Lỗi: Không tìm thấy mô hình tại {model_path}")
    print("Vui lòng đảm bảo đường dẫn mô hình là chính xác.")
    exit(1)

# 1. Load custom RoBERTa model
print("Đang tải mô hình RoBERTa tùy chỉnh...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
labels_roberta = ['Irrelevant', 'Negative', 'Neutral', 'Positive']
print("✓ Tải mô hình RoBERTa thành công!")

# 2. Load pre-trained DistilBERT
print("Đang tải mô hình DistilBERT...")
distilbert_pipeline = pipeline("sentiment-analysis", 
                               model="distilbert-base-uncased-finetuned-sst-2-english")
print("✓ Tải mô hình DistilBERT thành công!")

# 3. Initialize VADER
print("Đang khởi tạo VADER...")
vader = SentimentIntensityAnalyzer()
print("✓ Khởi tạo VADER thành công!\n")

# Các nhãn cảm xúc
labels = ['Irrelevant', 'Negative', 'Neutral', 'Positive']

# Biểu tượng cảm xúc
emotion_emoji = {
    'Irrelevant': '😐',
    'Negative': '😢',
    'Neutral': '😑',
    'Positive': '😊'
}

# ===== Phương pháp 1: Custom RoBERTa =====
def predict_sentiment_roberta(text):
    """Dự đoán cảm xúc bằng Custom RoBERTa"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        predicted_class_id = torch.argmax(logits, dim=-1).item()
    
    predicted_label = labels_roberta[predicted_class_id]
    confidence = float(probabilities[predicted_class_id])
    return predicted_label, confidence, dict(zip(labels_roberta, probabilities))

# ===== Phương pháp 2: DistilBERT =====
def predict_sentiment_distilbert(text):
    """Dự đoán cảm xúc bằng DistilBERT"""
    result = distilbert_pipeline(text[:512])[0]
    label = result['label']
    confidence = result['score']
    
    # Map DistilBERT labels to our format
    if label == 'POSITIVE':
        return 'Positive', confidence, {'POSITIVE': confidence, 'NEGATIVE': 1-confidence}
    else:
        return 'Negative', confidence, {'NEGATIVE': confidence, 'POSITIVE': 1-confidence}

# ===== Phương pháp 3: VADER =====
def predict_sentiment_vader(text):
    """Dự đoán cảm xúc bằng VADER"""
    scores = vader.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        label = 'Positive'
        confidence = scores['pos']
    elif compound <= -0.05:
        label = 'Negative'
        confidence = scores['neg']
    else:
        label = 'Neutral'
        confidence = scores['neu']
    
    return label, confidence, {'Positive': scores['pos'], 'Negative': scores['neg'], 'Neutral': scores['neu']}

# ===== Phương pháp 4: TextBlob =====
def predict_sentiment_textblob(text):
    """Dự đoán cảm xúc bằng TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        label = 'Positive'
        confidence = polarity
    elif polarity < -0.1:
        label = 'Negative'
        confidence = abs(polarity)
    else:
        label = 'Neutral'
        confidence = abs(polarity)
    
    return label, confidence, {'Polarity': polarity, 'Subjectivity': blob.sentiment.subjectivity}

# ===== Hàm tổng hợp để so sánh tất cả phương pháp =====
def analyze_sentiment_multi_methods(text):
    """Phân tích cảm xúc bằng 4 phương pháp"""
    if not text.strip():
        return "Vui lòng nhập nội dung để phân tích!", "", "", ""
    
    # Dự đoán từ 4 phương pháp
    roberta_label, roberta_conf, roberta_probs = predict_sentiment_roberta(text)
    distilbert_label, distilbert_conf, distilbert_probs = predict_sentiment_distilbert(text)
    vader_label, vader_conf, vader_probs = predict_sentiment_vader(text)
    textblob_label, textblob_conf, textblob_probs = predict_sentiment_textblob(text)
    
    # Kết quả RoBERTa chi tiết
    roberta_detail = f"{emotion_emoji.get(roberta_label, '')} {roberta_label}\nĐộ tin cậy: {roberta_conf:.2%}"
    
    # Kết quả DistilBERT chi tiết
    distilbert_detail = f"{emotion_emoji.get(distilbert_label, '')} {distilbert_label}\nĐộ tin cậy: {distilbert_conf:.2%}"
    
    # Kết quả VADER chi tiết
    vader_detail = f"{emotion_emoji.get(vader_label, '')} {vader_label}\nĐộ tin cậy: {vader_conf:.2%}"
    
    # Kết quả TextBlob chi tiết
    textblob_detail = f"{emotion_emoji.get(textblob_label, '')} {textblob_label}\nĐộ tin cậy: {textblob_conf:.2%}"
    
    return roberta_detail, distilbert_detail, vader_detail, textblob_detail

# Tạo Gradio interface
with gr.Blocks(title="Phân Tích Cảm Xúc - Đa Phương Pháp") as demo:
    gr.Markdown(
        """
        # 🎯 Ứng Dụng Phân Tích Cảm Xúc - Đa Phương Pháp (Multi-Method Sentiment Analysis)
        
        Nhập văn bản để phân tích cảm xúc bằng **4 phương pháp khác nhau**:
        
        1. **Custom RoBERTa** - Mô hình Transformer được fine-tune chuyên biệt
        2. **DistilBERT** - Mô hình tiên tiến từ HuggingFace
        3. **VADER** - Phương pháp dựa trên luật, tốt cho media xã hội
        4. **TextBlob** - Phương pháp dựa trên từ điển, đơn giản và nhanh
        
        So sánh kết quả từ các phương pháp khác nhau để có cái nhìn toàn diện!
        """
    )
    
    with gr.Row():
        # Input text
        input_text = gr.Textbox(
            label="📝 Nhập văn bản cần phân tích",
            placeholder="Ví dụ: Tôi rất yêu thích sản phẩm này!",
            lines=4
        )
    
    # Nút phân tích
    analyze_btn = gr.Button("🔍 Phân Tích Cảm Xúc (4 Phương Pháp)", variant="primary", size="lg")
    
    gr.Markdown("## 📊 Kết Quả Phân Tích")
    
    # Output từ 4 phương pháp
    with gr.Row():
        roberta_output = gr.Textbox(
            label="1️⃣ Custom RoBERTa",
            interactive=False,
            text_align="center"
        )
        distilbert_output = gr.Textbox(
            label="2️⃣ DistilBERT",
            interactive=False,
            text_align="center"
        )
    
    with gr.Row():
        vader_output = gr.Textbox(
            label="3️⃣ VADER (Rule-based)",
            interactive=False,
            text_align="center"
        )
        textblob_output = gr.Textbox(
            label="4️⃣ TextBlob",
            interactive=False,
            text_align="center"
        )
    
    # Ví dụ để test
    gr.Examples(
        examples=[
            ["This new Borderlands update is absolutely amazing! I love it."],
            ["My internet connection is so bad right now, I can't even play."],
            ["You are an idiot"],
            ["I don't care about the new update, it's just another patch."],
            ["This game is okay, nothing special."],
        ],
        inputs=input_text,
        label="💡 Ví Dụ Test"
    )
    
    # Xử lý sự kiện click nút
    analyze_btn.click(
        fn=analyze_sentiment_multi_methods,
        inputs=input_text,
        outputs=[roberta_output, distilbert_output, vader_output, textblob_output]
    )
    
    # Cũng có thể phân tích khi submit (Enter)
    input_text.submit(
        fn=analyze_sentiment_multi_methods,
        inputs=input_text,
        outputs=[roberta_output, distilbert_output, vader_output, textblob_output]
    )

if __name__ == "__main__":
    # Chạy ứng dụng
    demo.launch(
        share=False,
        server_name="127.0.0.1",  # Localhost
        server_port=7861,
        theme=gr.themes.Soft(),
        show_error=True
    )

# Truy cập: App sẽ mở tại http://127.0.0.1:7861 sau khi chạy:
# cd "d:\NLP -Tài liệu phân tích cảm xúc\Mô hình phân tích cảm xúc từ dữ liệu Twitter-17"; python webapp.py
# 
# Tính năng:
# - Phân tích cảm xúc bằng 4 phương pháp khác nhau
# - So sánh kết quả giữa các phương pháp
# - Hỗ trợ cả tiếng Anh lẫn tiếng Việt