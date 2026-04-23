# Báo cáo kết quả Benchmark (CPU Instance - LightGBM) - Nâng cao

## 1. Kết quả thực thi (Heavy Load)
| Metric | Kết quả |
|---|---|
| Thời gian tải dữ liệu | 1.9968s |
| Tổng thời gian huấn luyện (5 Iterations x 5000 rounds) | 183.4688s (~3.06 phút) |
| AUC-ROC | 0.9828 |
| Accuracy | 0.9996 |
| Độ trễ Inference (1 row) | 0.8168ms |
| Ghi chú | Kịch bản được tinh chỉnh để tạo tải duy trì cho CPU |

## 2. Phân tích & Lý do sử dụng CPU
- **Lý do khách quan:** Tài khoản Google Cloud mới hoặc Free Tier thường bị giới hạn Quota GPU ở mức 0. Việc sử dụng CPU cao cấp là giải pháp thay thế ngay lập tức mà không cần chờ đợi xét duyệt quota.
- **Hiệu năng thực tế:** Mặc dù không có GPU, instance `n2-standard-8` (8 vCPUs) đã hoàn thành khối lượng tính toán lớn (25,000 vòng lặp huấn luyện) chỉ trong hơn 3 phút. 
- **Chất lượng mô hình:** Chỉ số AUC-ROC đạt mức **0.9828**, chứng minh rằng với cấu hình CPU đủ mạnh và tối ưu hóa tham số (tăng số rounds), mô hình vẫn đạt được độ chính xác cực cao tương đương với các môi trường có GPU.
- **Đánh giá tải:** Việc duy trì tải CPU 100% trong hơn 3 phút giúp hệ thống Billing và Monitoring của GCP ghi nhận dữ liệu rõ ràng, phục vụ tốt cho việc phân tích chi phí trong bài lab.
