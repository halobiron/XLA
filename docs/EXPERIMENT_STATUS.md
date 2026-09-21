# Trạng thái thí nghiệm TACS-X

Cập nhật lần cuối: 2026-09-19

## Những gì có giá trị khoa học

Tất cả kết quả dưới đây dùng phép chia CIFAR-100 xác định: 45.000 ảnh huấn
luyện, 5.000 ảnh xác thực và bộ kiểm thử chính thức gồm 10.000 ảnh được giữ
nguyên. Tập ứng viên cố định chỉ được tạo từ phần huấn luyện. Độ chính xác trên
tập kiểm thử chỉ được tính một lần, sau khi chọn checkpoint theo độ chính xác
trên tập xác thực.

Ngân sách tái lập quy mô nhỏ được chủ đích giảm xuống còn 5 epoch, batch size
32 và 4 ứng viên được lấy mẫu cho mỗi truy vấn (`C=4`). Các sai khác này được
lưu trong `config.yaml` của từng lượt chạy; chỉ so sánh các lượt chạy cùng ngân
sách này.

| Lượt chạy | Chế độ | Độ chính xác kiểm thử | Diễn giải |
|---|---|---:|---|
| `cifar100_no_context_e5` | Không ngữ cảnh | **82.79%** | Mốc tham chiếu chính không ngữ cảnh. |
| `cifar100_random_c4_e5` | Ngữ cảnh ngẫu nhiên | 44.78% | Ngữ cảnh tùy ý gây hại nghiêm trọng cho tác vụ. |
| `cifar100_dino_c4_e5` | Độ tương đồng DINO đóng băng | 49.23% | Cao hơn ngẫu nhiên 4.45 điểm, nhưng vẫn kém xa không ngữ cảnh. |
| `cifar100_gumbel_c4_e5_actionfix` | Bộ chọn chỉ Gumbel | **44.64%** | Lượt chạy lại sau action-fix; vẫn thấp hơn ngữ cảnh ngẫu nhiên và DINO similarity. |
| `cifar100_policy_c4_e5_actionfix` | Bộ chọn chỉ policy | **44.44%** | Lượt chạy lại sau action-fix; thấp hơn Gumbel và ngữ cảnh ngẫu nhiên. |
| `cifar100_full_tacs_c4_e5_actionfix` | TACS lai đầy đủ | **44.69%** | Lượt chạy lại sau action-fix; tốt nhất trong ba bộ chọn học được, nhưng chênh lệch rất nhỏ. |

Kết quả Không ngữ cảnh ban đầu với 30 epoch (`84.61%`) có giá trị như một lượt
chạy độc lập 30 epoch, nhưng **không thể so sánh** với ma trận 5 epoch ở trên.
Lượt chạy trước đó chọn checkpoint bằng tập kiểm thử là không hợp lệ và không
được báo cáo.

## Kết luận hiện tại

Trên CIFAR-100 với ngân sách 5 epoch/C=4, việc thêm một ảnh ứng viên không tự
động mang lại ích lợi. Chọn ngẫu nhiên gây hại; truy hồi cosine DINO đóng băng
cải thiện so với chọn ngẫu nhiên nhưng vẫn gây hại so với chỉ dùng ảnh truy vấn.
Điều này thiết lập động lực cho việc chọn theo tác vụ; nó chưa chứng minh rằng
TACS cải thiện tác vụ. Trong các bộ chọn được học của lượt action-fix, TACS đầy
đủ tốt nhất với 44.69%, chỉ cao hơn chỉ Gumbel 0.05 điểm phần trăm. Chỉ policy
kém nhất với 44.44%, nên thành phần policy không mang lại ích lợi đo lường được
trong lượt chạy ngắn này.

Các chẩn đoán truy hồi cũng phù hợp với kết quả này:

- Ngữ cảnh ngẫu nhiên: tỷ lệ chọn khác lớp 98.83%, cosine pixel trung bình giữa
  truy vấn/ngữ cảnh 0.0227.
- Độ tương đồng DINO: tỷ lệ chọn khác lớp 96.57%, cosine pixel trung bình giữa
  truy vấn/ngữ cảnh 0.0762.
- Entropy lựa chọn của độ tương đồng DINO gần `log(4)`, nên bốn ứng viên được
  lấy mẫu ít được phân tách bởi điểm cosine DINO thô.
- Chỉ Gumbel (action-fix): entropy lựa chọn 1.2612, tỷ lệ khác lớp 98.17%,
  cosine pixel trung bình truy vấn/ngữ cảnh 0.0441, phần thưởng trung bình
  không áp dụng.
- Chỉ policy: entropy lựa chọn 1.1627, tỷ lệ khác lớp 98.21%, cosine pixel
  trung bình truy vấn/ngữ cảnh 0.0309, phần thưởng trung bình -1.3090.
- TACS đầy đủ (action-fix): entropy lựa chọn 1.2055, tỷ lệ khác lớp 98.27%,
  cosine pixel trung bình truy vấn/ngữ cảnh 0.0402, phần thưởng trung bình
  -1.2574.

Cả hai lượt chạy dựa trên policy đều có phần thưởng trung bình âm, nghĩa là ngữ
cảnh được chọn làm tăng loss theo ngữ cảnh so với loss chỉ truy vấn tính trung
bình. Entropy bộ chọn vẫn cao ở mọi chế độ được học (mức tối đa cho bốn ứng
viên là `log(4) = 1.3863`), nên mức ưu tiên giữa các ứng viên vẫn yếu.

Lưu ý triển khai: sau các lượt chạy trên, đường `policy_only` đã được sửa để
dùng cùng action cho task loss, reward và REINFORCE; các chế độ Gumbel cũng đã
lưu action/ngữ cảnh thực tế thay vì suy ra bằng `argmax` khi xuất metadata.
Do đó, accuracy đã ghi nhận vẫn là kết quả quan sát được, nhưng selected-pair
diagnostics của các chế độ được học ở trên không thể coi là metadata chính xác
cho action thực tế nếu chúng được tạo trước bản sửa này.

## Trạng thái sau khi sửa

- [x] Đã xác minh test suite của dự án: 9/9 test pass.
- [x] Đã sửa action mismatch ở `policy_only` và metadata action/ngữ cảnh cho
  Gumbel, policy-only và full TACS.
- [x] Đã chạy lại và lưu đầy đủ artifact cho Gumbel, policy-only và full TACS
  sau action-fix: `config.yaml`, `metrics.json`, `history.csv`,
  `checkpoint_best.pt` và `selected_pairs.csv`.
- [x] Đã tạo `outputs/summary.csv`, `outputs/summary.json` và
  `outputs/ablation_table.md`; toàn bộ thư mục `outputs/` đã được nén trong
  `outputs.zip`.
- [ ] Artifact thô của các baseline cũ (No-Context, Random context và DINO
  similarity) không có trong repository; chỉ còn các số liệu đã ghi ở bảng
  trên. Không cần chạy lại chúng vì action-fix không ảnh hưởng các baseline.
- [x] Đã render 16 retrieval examples cho mỗi run action-fix. Ảnh và CSV mô tả
  các cặp được chọn nằm tại `outputs/<run_name>/retrieval_examples/` và
  `outputs/<run_name>/retrieval_examples.csv`. Các cặp được chọn có chủ đích
  gồm cả kết quả tốt/xấu theo reward (policy/full TACS), hoặc score cao/thấp
  (Gumbel), thay vì chỉ lấy một đoạn đầu của test set.

## Việc cần làm tiếp theo — extension Top-K TACS-X

Không chạy lại các baseline hoặc ba run `*_actionfix`: chúng đã có kết quả hợp
lệ trong ma trận ngân sách 5 epoch/C=4. Phần còn lại của dự án là đánh giá
extension Top-K TACS-X trên cùng ngân sách đó.

1. [x] **Đã sửa metadata và renderer cho Top-K.** `selected_pairs.csv` giờ
   lưu `topk_candidate_ids`, `topk_candidate_labels`, `topk_weights` và
   `effective_k`; các trường `candidate_id`/`candidate_label` cũ được giữ là
   top-1 để tương thích ngược. Renderer hiển thị query cùng mọi context đã đóng
   góp, kèm weight. Test suite xác minh weight tổng bằng 1 và evaluator lưu đủ
   danh sách Top-K.
2. **Chạy Top-K cố định với `K=2`**, giữ seed 42, C=4, 5 epoch và batch size
   32. Config mặc định đã đặt `topk.k: 2`:

   ```powershell
   .venv\Scripts\python.exe -m tacsx.cli --config configs/cifar100.yaml --mode topk_tacs --name cifar100_topk_k2_c4_e5 --candidates-per-query 4 --epochs 5 --batch-size 32
   ```

3. **Chạy adaptive Top-K** với `adaptive_threshold: 0.90` từ config, dùng đúng
   ngân sách trên:

   ```powershell
   .venv\Scripts\python.exe -m tacsx.cli --config configs/cifar100.yaml --mode adaptive_topk_tacs --name cifar100_adaptive_topk_c4_e5 --candidates-per-query 4 --epochs 5 --batch-size 32
   ```

4. **Tổng hợp và phân tích extension** sau khi hai run hoàn tất:

   ```powershell
   .venv\Scripts\python.exe -m tacsx.summarize --outputs outputs
   .venv\Scripts\python.exe scripts\render_retrieval_examples.py --outputs outputs --data-root data --limit 16
   ```

   So sánh `topk_tacs` và `adaptive_topk_tacs` với `full_tacs` dưới cùng ngân
   sách. Báo cáo rõ nếu Top-K không cải thiện accuracy; đó vẫn là kết quả hợp
   lệ của extension, không phải lý do để chỉnh theo test accuracy.

Khi báo cáo, sử dụng ba hàng `*_actionfix` ở bảng trên cho các chế độ học được;
không dùng các số learned-selector từ trước bản sửa action.

## Ghi chú thời gian chạy

`dino_similarity` lưu cache embedding toàn cục DINO đóng băng cho tập ứng viên
cố định một lần, rồi chỉ tải ảnh ngữ cảnh đã chọn cho mỗi batch. Cache này là
một tối ưu hóa chính xác cho tập ứng viên cố định, đóng băng; nó không dùng nhãn
hoặc ảnh kiểm thử. Các chế độ bộ chọn được học vẫn xử lý ảnh ứng viên đã lấy mẫu
vì đường truyền ngữ cảnh straight-through của chúng cần các pixel đó.
