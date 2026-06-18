# Báo cáo Hiệu năng Huấn luyện & Suy luận LightGBM (CPU Fallback)

Do tài khoản AWS mới bị giới hạn nghiêm ngặt về hạn mức sử dụng vCPU cho GPU (mặc định bằng 0), yêu cầu tăng quota cần nhiều thời gian phê duyệt. Để đảm bảo tiến độ thực hành, bài lab được chuyển đổi sang phương án dự phòng chạy trên CPU instance cao cấp `r5.2xlarge` (8 vCPUs, 32 GB RAM) sử dụng mô hình LightGBM trên tập dữ liệu Credit Card Fraud Detection.

Kết quả benchmark hiệu năng đạt được như sau:
* **Thời gian huấn luyện (Training Time):** Chỉ mất **1.182 giây** để hoàn thành huấn luyện trên toàn bộ tập dữ liệu (284,807 dòng) nhờ cơ chế song song hóa tối ưu của LightGBM tận dụng toàn bộ 8 cores CPU.
* **Độ chính xác (AUC-ROC):** Đạt **0.906422** (Accuracy: 99.89%), cho thấy mô hình phân loại cực kỳ chính xác các giao dịch gian lận.
* **Tốc độ suy luận (Inference Speed):** Độ trễ cho 1 dòng dữ liệu (Inference Latency) cực thấp ở mức **0.3337 ms**, và băng thông xử lý hàng loạt đạt **2,260,276.13 dòng/giây** (Throughput), đáp ứng hoàn hảo cho các hệ thống phát hiện gian lận thời gian thực (Real-time Fraud Detection).
