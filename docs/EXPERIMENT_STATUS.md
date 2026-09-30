# Trạng thái thí nghiệm TACS-X

Cập nhật lần cuối: 2026-09-30

## Protocol và phạm vi kết quả

Các run được báo cáo dùng CIFAR-100 với phép chia xác định: 45.000 ảnh train,
5.000 ảnh validation và 10.000 ảnh test. Candidate pool cố định chỉ được tạo từ
train; checkpoint được chọn theo validation trước khi đánh giá test. Không dùng
nhãn hoặc dữ liệu test để tạo pool hay điều chỉnh selector.

Đây là mini reproduction có chủ đích: DINOv3 ViT-S/16 đóng băng, ảnh 224×224,
5 epoch, batch size 32, seed 42 và 4 ứng viên cho mỗi query (`C=4`). Các giá trị
mặc định khác của selector/policy là `tau=0.1` và `lambda_policy=1.0`. Mỗi run
ghi lại các sai khác này trong `config.yaml`; chỉ được so sánh trực tiếp các run
cùng ngân sách và protocol.

## Kết quả có artifact trong `outputs/`

| Run | Mode | Test accuracy | Best validation accuracy | Test loss | Ghi chú |
|---|---|---:|---:|---:|---|
| `cifar100_gumbel_c4_e5_actionfix` | `gumbel_only` | 44.64% | 44.78% | 2.2151 | Metadata dùng action thực tế sau action-fix. |
| `cifar100_policy_c4_e5_actionfix` | `policy_only` | 44.44% | 43.12% | 2.2418 | Cùng action cho task loss, reward và REINFORCE. |
| `cifar100_full_tacs_c4_e5_actionfix` | `full_tacs` | 44.69% | 43.98% | 2.2323 | Tốt nhất trong ba learned top-1 modes, nhưng chênh lệch nhỏ. |
| `cifar100_topk_k2_c4_e5` | `topk_tacs`, `K=2` | **64.78%** | 63.56% | 1.3708 | Gộp cố định hai context bằng trọng số khả vi. |
| `cifar100_adaptive_topk_c4_e5` | `adaptive_topk_tacs`, threshold 0.90 | **69.12%** | 67.86% | 1.1823 | Kết quả cao nhất trong các run hiện có. |

Mỗi năm run trên hiện có đủ `config.yaml`, `metrics.json`, `history.csv`,
`checkpoint_best.pt`, `selected_pairs.csv`, `retrieval_examples.csv` và 16 ảnh
truy hồi trong thư mục run tương ứng.

Các baseline cũ không có artifact thô trong repository, nên chỉ giữ các số đã
ghi nhận trước đây để làm bối cảnh, không đưa vào bảng có thể kiểm chứng ở trên:
No-Context 82.79%, random context 44.78% và DINO similarity 49.23%. Run
No-Context 30 epoch (84.61%) là một thí nghiệm riêng, không so sánh với ma trận
5 epoch. Không sử dụng các kết quả learned-selector trước action-fix.

## Diễn giải hiện tại

Với `C=4` và chỉ chọn một context, Gumbel, policy-only và full TACS đều thấp
hơn No-Context và chỉ xấp xỉ random context. Full TACS top-1 cao hơn
Gumbel/policy-only lần lượt 0.05 và 0.25 điểm phần trăm, chưa đủ để suy luận lợi
ích ổn định từ policy trong một seed.

Ngược lại, việc gộp context trong TACS-X cải thiện rõ rệt so với full TACS
top-1 dưới cùng ngân sách: `K=2` tăng **20.09** điểm phần trăm, còn adaptive
Top-K tăng **24.43** điểm phần trăm. Kết quả này là bằng chứng hỗ trợ extension
trên cấu hình mini hiện tại, không phải tái lập chính xác bài báo và chưa thể
khái quát khi chỉ có seed 42.

| Chẩn đoán test | Gumbel | Policy-only | Full TACS | Top-K `K=2` | Adaptive Top-K |
|---|---:|---:|---:|---:|---:|
| Entropy selector | 1.2612 | 1.1627 | 1.2055 | 1.3459 | 1.3585 |
| Tỷ lệ context khác lớp | 98.17% | 98.21% | 98.27% | 98.00% | 98.47% |
| Cosine query/context | 0.0441 | 0.0309 | 0.0402 | 0.0462 | 0.0483 |
| Mean reward | Không áp dụng | -1.3090 | -1.2574 | Không áp dụng | Không áp dụng |

Entropy vẫn cao so với mức cực đại `log(4)=1.3863`, và phần lớn context khác
lớp. Mức tăng của Top-K do đó cần được phân tích theo nhóm mẫu và không nên bị
diễn giải đơn giản thành selector đã tìm được các ảnh cùng lớp. `mean_reward`
trống ở các mode Top-K là đúng thiết kế: chúng tối ưu task loss qua tổng trọng
số khả vi, không gọi nhánh REINFORCE/reward.

## Hành vi adaptive Top-K

`adaptive_threshold=0.90` chọn hai context cho 148/10.000 mẫu test (1.48%) và
ba context cho 9.852/10.000 mẫu (98.52%). Không có mẫu nào chọn một hoặc đủ bốn
context. Vì vậy cấu hình adaptive hiện tại vận hành gần như Top-3; đây là điểm
cần được xác nhận qua ablation ngưỡng trước khi kết luận adaptive-K mang lại lợi
ích riêng so với việc dùng `K=3` cố định.

## Trạng thái hoàn tất

- Đã sửa metadata/renderer Top-K: `selected_pairs.csv` lưu đủ
  `topk_candidate_ids`, `topk_candidate_labels`, `topk_weights`, `effective_k`;
  các trường `candidate_id` và `candidate_label` vẫn là top-1 để tương thích
  ngược.
- Đã sửa action mismatch ở `policy_only`; Gumbel, policy-only và full TACS lưu
  action/context thực tế thay vì suy diễn bằng `argmax` khi xuất metadata.
- Đã chạy và lưu artifact đầy đủ cho ba run `*_actionfix` và hai run Top-K.
- Đã render 16 retrieval examples cho từng run hiện có.
- Test suite hiện pass **11/11**; gồm kiểm tra score shape, hard Gumbel,
  gradient policy, reward sign, candidate-pool leakage, tổng trọng số Top-K và
  metadata Top-K.

## Hoàn tất đợt thí nghiệm hiện tại

Không còn bước chạy hoặc tổng hợp nào cần thiết cho đợt thí nghiệm hiện tại.
`outputs/summary.csv`, `outputs/summary.json` và `outputs/ablation_table.md`
đã được tổng hợp lại từ đủ năm run có artifact trong `outputs/`:
Gumbel-only, policy-only, full TACS, Top-K `K=2` và adaptive Top-K.

Các số liệu trong bảng kết quả của tài liệu này khớp với `metrics.json` của
từng run và với các file summary đã cập nhật. Những thử nghiệm như nhiều seed,
ablation `K` hoặc threshold là hướng mở rộng tùy chọn cho một đợt nghiên cứu
khác, không phải việc còn thiếu của đợt này.

## Ghi chú triển khai

`dino_similarity` cache embedding toàn cục của DINO cho candidate pool cố định
và chỉ tải context được chọn theo batch; cache không dùng ảnh hay nhãn test. Các
mode selector học được vẫn xử lý ảnh candidate đã lấy mẫu vì đường truyền
straight-through cần pixel context. Chỉ metadata được sinh sau action-fix được
dùng để chẩn đoán action thực tế.
