# No-Context và Full TACS: kết quả đối chiếu

Hai checkpoint `clean_v3` đã được đánh giá trên đủ **10.000 ảnh test chuẩn
CIFAR-100**, giữ nguyên cấu hình và checkpoint đã chọn từ validation.

| Accuracy | Paper, Bảng 2 | Validation clean_v3 | Test chuẩn clean_v3 |
|---|---:|---:|---:|
| No-Context | 87,8% | 90,07% | **90,22%** |
| Full TACS | 89,2% | 90,19% | **90,01%** |
| TACS − No-Context | +1,4 điểm | +0,12 điểm | **−0,21 điểm** |

Trên test, TACS sửa đúng 137 ảnh nhưng làm sai thêm 158 ảnh. Khoảng tin cậy
bootstrap ghép cặp 97,5% cho chênh lệch accuracy là **[−0,59; +0,18] điểm**,
có chứa 0. NLL lần lượt là 0,33261 và 0,33000; chênh lệch NLL cũng chưa có
ý nghĩa thống kê theo phép đo đã đăng ký. Một seed chưa phản ánh biến thiên
giữa các lần train.

Cả hai mô hình có điểm số cao hơn mốc tương ứng của paper về mặt số học.
Chưa có ablation xác định thay đổi nào của recipe gây ra khoảng cách này.

## Những gì đã thay đổi

- Dùng official DINOv3 ViT-S/16 LVD-1689M và sửa trainer để mở đúng số block,
  tách LR backbone `1e-5` và head `3e-4`, áp dụng cosine scheduler.
- Học query-only 5 epoch với 4 block cuối, final norm và head. Cả hai nhánh
  nhận checkpoint này rồi học tiếp 10 epoch với toàn bộ 12 task block được mở.
- Dùng ảnh 224, batch 32, một seed và augmentation khác paper. Những thay đổi
  này có thể làm điểm tăng hoặc giảm, không mặc định làm điểm giảm.
- Nhánh TACS dùng nearest-16 shortlist và selector encoder đóng băng với
  projector học được; các thay đổi này không ảnh hưởng nhánh No-Context.

Test dùng checkpoint No-Context epoch học tiếp 9 và Full TACS epoch 7, đã
khóa trước khi đọc test. Pool 9.000 ảnh chỉ từ train, không dùng nhãn để
retrieval và luôn chặn context trùng pixel với query. Kết quả phụ sau lọc
12 ảnh test trùng với train hoặc lặp trong test: 9.988 ảnh, No-Context 90,27%,
Full TACS 90,06%. Kết quả phụ không thay thế điểm trên test chuẩn.

Lần đánh giá này thực hiện theo yêu cầu người dùng sau khi đã biết kết quả
validation, lưu riêng và giữ nguyên hồ sơ đánh giá có cổng validation trước đó.
Chỉ đánh giá test ảnh sạch. Demo chính đối chiếu Full TACS dự án với No-Context
paper, nêu khác biệt thiết lập và giữ mốc No-Context của dự án để tránh quy toàn
bộ chênh lệch cho TACS.

Nguồn: [paper, mục 4.2](../paper/TACS_CVPR_2026.pdf#page=5),
[Bảng 2](../paper/TACS_CVPR_2026.pdf#page=6),
[plan](../outputs/tacs_shortlist_fullft_clean_v3/plan.json),
[kết quả test JSON](../outputs/tacs_shortlist_fullft_clean_v3/evaluation/test_standard/test_result.json),
[bảng CSV](../outputs/tacs_shortlist_fullft_clean_v3/evaluation/test_standard/test_comparison.csv).
