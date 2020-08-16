Document Scanner có thể scan một văn bản và xuất ra file text nội dung của văn bản đó, thông qua các bước :
- Color picker : detect màu của văn bản, làm nổi bật chữ trên nền
- Wrap image : chọn khung văn bản, chỉ detect văn bản trong phần khung đó, bỏ qua phần nền không cần thiết
- Text detection : phát hiện nội dung văn bản bằng Tesseract, lưu văn bản scan được ở file text

Các file .py :
- colorPicker.py : chọn mặt nạ thích hợp cho ảnh
- detectingCharacters.py : phát hiện từng ký tự trong văn bản. Lưu ảnh văn bản đã được cắt trong thư mục 'Resource/Saved image', lưu nội dung văn bản trong 'Resource/Saved text'
- detectingWords.py : phát hiện từng từ trong văn bản. Lưu ảnh văn bản đã được cắt trong thư mục 'Resource/Saved image', lưu nội dung văn bản trong 'Resource/Saved text'
- usingWebcam.py : sử dụng webcam để scan văn bản, nhấn phím 's' để  lưu ảnh văn bản đã được cắt trong thư mục 'Resource/Saved image', lưu nội dung văn bản trong 'Resource/Saved text'
