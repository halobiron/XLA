# Trạng thái thí nghiệm TACS-X

Cập nhật lần cuối: 2026-11-14

## Kiểm toán nguyên nhân và bản sửa trainer v3 (2026-11-14)

Toàn bộ **23** `metrics.json` hiện có thiếu `trainer_version`: đây là kết quả
trainer cũ, không phải kết quả của bản sửa partial-unfreeze/LR ở commit `4968810`
hay bản sửa v3. Các bảng bên dưới là kết quả lịch sử, không được gán cho trainer mới.

Các lỗi đã xác nhận:

- **Vị trí patch hai ảnh không khớp thứ tự token.** Chuỗi là `[query patches,
  context patches]`, nhưng RoPE lấy từ grid `H x 2W` ở thứ tự theo hàng. Ví dụ
  2x2: vị trí đúng phải là `[0,1,4,5,2,3,6,7]`, code cũ gán `[0,1,2,3,4,5,6,7]`.
  `horizontal` sửa ánh xạ; `legacy` giữ nguyên để đọc checkpoint cũ; `per_image`
  là ablation tái sử dụng tọa độ gốc từng ảnh. Paper không chỉ rõ RoPE hai ảnh.
- **Đánh giá sai quy tắc inference.** Gumbel/Policy/Full vẫn lấy mẫu ngẫu nhiên
  ở validation/test. V3 mặc định argmax; `evaluation.selection: sampled` chỉ
  dùng cho ablation có chủ đích. Không thể mặc định argmax sẽ tăng accuracy.
- **Run unf4 cũ không phải fine-tune 4 block.** Đối chiếu checkpoint với trọng số
  official cho thấy cả 12 block và patch embedding đều thay đổi. Code trainer
  cũ bỏ qua N khi `freeze:false`, dùng `lr_head=3e-4` cho toàn bộ tham số và không
  áp dụng scheduler. Commit `4968810` đã sửa, nhưng chưa có artifact run mới.
- **Adaptive-K bỏ mất ứng viên vượt ngưỡng.** V3 giữ ứng viên đầu tiên đưa tổng
  xác suất tới ngưỡng. Kết quả adaptive cũ không mô tả đúng quy tắc coverage này.
- **DINO baseline không chạy được cùng protocol fine-tune.** Cache cũ yêu cầu
  task backbone đóng băng. V3 dùng retriever riêng đóng băng khi task backbone
  train được, để query/candidate retrieval luôn cùng không gian embedding.
- **Chế độ train tắt cả module không có tham số.** V3 chỉ đặt toàn bộ adapter
  đóng băng vào eval, không tắt RoPE augmentation/dropout của backbone train được.

Đối chứng chỉ đọc trên **64 ảnh validation**, không đọc tập test, không huấn luyện:

| Probe, cùng head No-Context và trọng số official | Accuracy | Cosine với query-only |
|---|---:|---:|
| Query-only | 79.69% | 1.000 |
| Query + bản sao query, layout legacy | 65.63% | 0.718 |
| Query + bản sao query, layout horizontal sửa thứ tự | 78.13% | 0.824 |
| Query + context ngẫu nhiên, legacy | 37.50% | 0.587 |
| Query + cùng context ngẫu nhiên, horizontal | 42.19% | 0.661 |

Nhánh query-only qua adapter khớp official `forward_features` với max absolute
error **0.0**. Backbone các checkpoint No-Context, Random, Full và Top-K K4 đều
khớp trọng số pretrained, xác nhận chúng hoàn toàn đóng băng. Sửa vị trí cải thiện
đặc trưng trong đối chứng này nhưng chưa loại bỏ tác hại của context ngẫu nhiên.
Đây là probe với head cũ trên đầu vào thay đổi, **không phải accuracy của mô hình
đã train lại**, và không đủ để phân bổ chính xác toàn bộ khoảng cách 38 điểm.

Kết luận: lỗi vị trí đã được chứng minh có ảnh hưởng; inference và trainer cũ
cũng sai protocol. Đóng băng backbone khi thay đổi đầu vào hai ảnh là giới hạn
quan trọng cần kiểm chứng bằng retraining, chưa phải tác động nhân quả đã định lượng.
Paper classification fine-tune toàn bộ mô hình; 5 epoch, 224px, partial fine-tune
và C=4 vẫn là mini reproduction, phải công bố sai khác.

Cách chạy đối chứng tiếp theo: dùng `configs/cifar100_unf4.yaml`, tên run mới
v3, trước hết No-Context và Random, sau đó DINO-similarity và Full. Dùng cùng
protocol và kiểm tra validation trước khi quyết định mở rộng; không tinh chỉnh
theo test. Chưa có kết quả nào chứng minh v3 vượt No-Context.

Chi tiết máy đọc được: `outputs/diagnostics/fidelity.json`. Tái chạy bằng:

```powershell
.venv\Scripts\python.exe scripts/diagnose_fidelity.py --samples 64
```

V3 cũng dùng cùng action cho task/reward/policy để reward trong metadata mô tả
đúng context đã lưu, tái sử dụng loss đã detach, tránh forward selector lặp, và
tách generator DataLoader khỏi RNG của model. Dùng chung action là lựa chọn
triển khai; paper mô tả hai đường tối ưu nhưng không bắt buộc cùng lần lấy mẫu.

Kiểm chứng bản sửa: **28 tests pass**, dùng dummy encoders, không tải model.
Smoke-test riêng với DINOv3 official xác nhận gradient qua Gumbel hữu hạn và
đầu ra paired legacy khớp code trước sửa (max absolute error 0.0).
Head phân loại được khởi tạo cùng seed giữa baseline và learned modes; thứ tự
DataLoader không bị thay đổi bởi RNG của selector hay số lần lấy mẫu policy.

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

`outputs/` hiện có **23 run**, chia thành ba nhóm. Mọi run dùng cùng protocol
(5 epoch, batch 32, DINOv3 ViT-S/16 đóng băng trừ khi ghi chú khác); các khác biệt
nằm ở cột Cấu hình. Số liệu lấy từ `outputs/summary.csv`.

### A. Ma trận chính (C=4, seed 42)

| Run | Mode | Test acc | Best val acc | Test loss | Ghi chú |
|---|---|---:|---:|---:|---|
| `cifar100_no_context_c4_e5` | `no_context` | **82.79%** | 82.04% | 0.6296 | Chỉ ảnh query; mốc so sánh cao nhất của mọi mode top-1/Top-K. |
| `cifar100_dino_sim_c4_e5` | `dino_similarity` | 49.23% | 48.02% | 1.9965 | Truy hồi cosine DINO đóng băng trong C=4; tốt nhất trong các mode top-1. |
| `cifar100_random_c4_e5` | `random_context` | 44.78% | 44.18% | 2.2398 | Chọn ngẫu nhiên 1 trong 4 ứng viên. |
| `cifar100_gumbel_c4_e5_actionfix` | `gumbel_only` | 44.64% | 44.78% | 2.2151 | Metadata dùng action thực tế sau action-fix. |
| `cifar100_policy_c4_e5_actionfix` | `policy_only` | 44.44% | 43.12% | 2.2418 | Cùng action cho task loss, reward và REINFORCE. |
| `cifar100_full_tacs_c4_e5_actionfix` | `full_tacs` | 44.69% | 43.98% | 2.2323 | Tốt nhất trong ba learned top-1 modes, chênh lệch nhỏ. |
| `cifar100_topk_k2_c4_e5` | `topk_tacs`, K=2 | 64.78% | 63.56% | 1.3708 | Gộp hai context bằng trọng số khả vi. |
| `cifar100_topk_k3_c4_e5` | `topk_tacs`, K=3 | 69.04% | 68.46% | 1.1800 | |
| `cifar100_topk_k4_c4_e5` | `topk_tacs`, K=4 | **70.99%** | 70.06% | 1.1088 | Cao nhất trong các run có context (dùng cả 4 ứng viên). |
| `cifar100_adaptive_topk_c4_e5` | `adaptive_topk_tacs`, thr 0.90 | 69.12% | 67.86% | 1.1823 | Gần như Top-3 (xem mục adaptive). |

### B. Ablation ngưỡng adaptive Top-K (C=4, seed 42)

| Run | Threshold | Test acc | Best val acc | Test loss |
|---|---:|---:|---:|---:|
| `cifar100_adaptive_thr50_c4_e5` | 0.50 | 48.01% | 45.90% | 2.0660 |
| `cifar100_adaptive_thr70_c4_e5` | 0.70 | 64.08% | 62.58% | 1.4192 |
| `cifar100_adaptive_topk_c4_e5` | 0.90 | 69.12% | 67.86% | 1.1823 |

### C. Biến thể bổ sung

| Run | Cấu hình khác | Test acc | Best val acc | Test loss |
|---|---|---:|---:|---:|
| `cifar100_no_context_c4_e5_unf4` | `no_context`, cấu hình ghi 4 block nhưng checkpoint thực tế thay đổi cả 12 block do lỗi trainer cũ | 48.63% | 48.00% | 1.9080 |
| `cifar100_no_context_c4_e5_s1` | seed 1 | 82.96% | 82.74% | 0.6294 |
| `cifar100_dino_sim_c4_e5_s1` | seed 1 | 49.96% | 49.74% | 1.9881 |
| `cifar100_full_tacs_c4_e5_s1` | seed 1 | 45.28% | 45.20% | 2.1946 |
| `cifar100_topk_k2_c4_e5_s1` | seed 1, K=2 | 65.17% | 64.56% | 1.3620 |
| `cifar100_random_c16_e5` | C=16 | 45.11% | 44.58% | 2.2406 |
| `cifar100_dino_sim_c16_e5` | C=16 | 53.00% | 52.76% | 1.7537 |
| `cifar100_gumbel_c16_e5` | C=16 | 45.54% | 44.66% | 2.2003 |
| `cifar100_policy_c16_e5` | C=16 | 42.80% | 42.94% | 2.2659 |
| `cifar100_full_tacs_c16_e5` | C=16 | 44.42% | 43.88% | 2.2197 |
| `cifar100_topk_k2_c16_e5` | C=16, K=2 | 62.92% | 62.88% | 1.3976 |

Lưu ý khi dùng nhóm C: các run `c16` chưa có bản `*_actionfix` tương ứng nên
không được xem là cùng chuẩn với nhóm A. Run `unf4` là cấu hình huấn luyện khác
(backbone được mở khóa), không so trực tiếp với ma trận backbone đóng băng.
Multi-seed chỉ có 2 seed (42, 1) và chỉ cho 5 cấu hình, nên chưa đủ để kết luận
thống kê.

Mỗi run có context có `config.yaml`, `metrics.json`, `history.csv`,
`checkpoint_best.pt`, `selected_pairs.csv`; các run Top-K và adaptive còn có
`retrieval_examples.csv` và ảnh truy hồi. Thư mục ảnh truy hồi không có ở mọi run: ví dụ `full_tacs_c16`, `topk_k2_c16`
và các run seed 1 có, còn `dino_sim_c16` thì không. `no_context` không có context nên chỉ có `config.yaml`,
`metrics.json`, `history.csv` và checkpoint.

Ba baseline (No-Context, random, DINO similarity) đã được chạy lại cùng
protocol và khớp kết quả ghi nhận trước đây. Run No-Context 30 epoch (84.61%) là
một thí nghiệm riêng, không so sánh với ma trận 5 epoch. Không sử dụng các kết
quả learned-selector trước action-fix.

## Diễn giải hiện tại

Với `C=4` và chỉ chọn một context, Gumbel, policy-only và full TACS đều thấp
hơn No-Context (82.79%) khoảng 38 điểm phần trăm và chỉ xấp xỉ random context
(44.78%); DINO similarity (49.23%) cao hơn chúng 4.5–4.8 điểm. Tức là trong cấu
hình mini này, mọi cách ghép một ảnh context khác lớp đều làm hại so với không
dùng context, và selector học được chưa vượt truy hồi tương tự đóng băng, trái
với Bảng 2/5 của bài báo (TACS > DINO-sim > No-Context trên CIFAR-100). Full TACS top-1 cao hơn
Gumbel/policy-only lần lượt 0.05 và 0.25 điểm phần trăm, chưa đủ để suy luận lợi
ích ổn định từ policy trong một seed.

Các mode Top-K (64.78%, 69.12%) cũng vẫn thấp hơn No-Context 18.0 và 13.7 điểm.
Ngược lại, việc gộp context trong TACS-X cải thiện rõ rệt so với full TACS
top-1 dưới cùng ngân sách: `K=2` tăng **20.09** điểm phần trăm, còn adaptive
Top-K tăng **24.43** điểm phần trăm. Kết quả này là bằng chứng hỗ trợ extension
trên cấu hình mini hiện tại, không phải tái lập chính xác bài báo và chưa thể
khái quát khi chỉ có seed 42.

| Chẩn đoán test | Random | DINO-sim | Gumbel | Policy-only | Full TACS | Top-K `K=2` | Adaptive Top-K |
|---|---:|---:|---:|---:|---:|---:|---:|
| Entropy selector | 1.3558 | 1.3838 | 1.2612 | 1.1627 | 1.2055 | 1.3459 | 1.3585 |
| Tỷ lệ context khác lớp | 98.83% | 96.57% | 98.17% | 98.21% | 98.27% | 98.00% | 98.47% |
| Cosine query/context | 0.0227 | 0.0762 | 0.0441 | 0.0309 | 0.0402 | 0.0462 | 0.0483 |
| Mean reward | Không áp dụng | Không áp dụng | Không áp dụng | -1.3090 | -1.2574 | Không áp dụng | Không áp dụng |

Entropy của random và DINO-sim là giá trị do code ghi cho các mode không học
selector, không phản ánh một chính sách được tối ưu; chỉ so sánh entropy giữa các
mode có selector học được. Ở DINO-sim, cosine query/context cao nhất (0.076) và
cross-class thấp nhất (96.57%) nhưng vẫn rất xa 0%: với CIFAR-100 và `C=4`, hầu
như không có ứng viên cùng lớp trong tập 4 ảnh.

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
- Đã chạy và lưu artifact đầy đủ cho ba baseline, ba run `*_actionfix` và hai run Top-K.
- Đã render 16 retrieval examples cho từng run có context.
- Test suite hiện pass **11/11**; gồm kiểm tra score shape, hard Gumbel,
  gradient policy, reward sign, candidate-pool leakage, tổng trọng số Top-K và
  metadata Top-K.

## Hoàn tất đợt thí nghiệm hiện tại

Không còn bước chạy hoặc tổng hợp nào bắt buộc cho đợt thí nghiệm hiện tại.
`outputs/summary.csv`, `outputs/summary.json` và `outputs/ablation_table.md`
chứa đủ 23 run có artifact trong `outputs/` (xem ba nhóm A/B/C ở trên).

Các số liệu trong bảng kết quả của tài liệu này khớp với `metrics.json` của
từng run và với các file summary đã cập nhật. Ablation `K` (2/3/4), ablation threshold (0.5/0.7/0.9), `C=16` và seed 1 cho
một số cấu hình đã được chạy; seed bổ sung cho các mode còn lại là hướng mở rộng
tùy chọn.

## Ghi chú triển khai

`dino_similarity` cache embedding toàn cục của DINO cho candidate pool cố định
và chỉ tải context được chọn theo batch; cache không dùng ảnh hay nhãn test. Các
mode selector học được vẫn xử lý ảnh candidate đã lấy mẫu vì đường truyền
straight-through cần pixel context. Chỉ metadata được sinh sau action-fix được
dùng để chẩn đoán action thực tế.
