@echo off
chcp 65001 > nul
cd /d "d:\NLP -Tài liệu phân tích cảm xúc\Mô hình phân tích cảm xúc từ dữ liệu Twitter-17"
echo ================================
echo Đang khởi động ứng dụng...
echo ================================
python webapp.py
pause
