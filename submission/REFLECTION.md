# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** _Trần Gia Thành_
**Khoá:** _A20-K4_
**Tier đã chạy:** _T4_
**Ngày:** _2026-10-08_

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | _NVIDIA T4 / 16 GB_ |
| Mô hình gốc | _unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit_ |
| Dữ liệu SFT | _saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch_ |
| Dữ liệu sở thích | _sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out_ |
| Chosen dài hơn rejected (NB2) | _65,875%_ |
| DPO: β / tốc độ học (lr) / số epoch | _0,1 / 5e-6 / 1_ |
| Giám khảo | _rm-panel: Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy: 66,67% / 100%_ |
| Chi phí | _0 đồng (Colab miễn phí)_ |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | _42 phút_ |
| VRAM cao nhất | _11.14GB_ |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | _0,089957_ |
| Độ chính xác reward trên held-out | _68,00%_ |
| Margin trên held-out | _0,079557_ |
| Chẩn đoán tự động (`diagnosis`) | _INTENDED_ |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | _556,76 → 571,81 ký tự_ |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

_Mô tả riêng `rewards/chosen` và `rewards/rejected` trên **train và held-out**. Chosen tăng hay giảm?
Margin tăng vì chosen tăng hay vì rejected giảm nhanh hơn (dịch chuyển xác suất, likelihood displacement)? Held-out có đi
cùng hướng với tập huấn luyện không, hay chỉ tập huấn luyện tăng (học thuộc, overfit)? Chẩn đoán tự động có khớp với điều bạn
thấy không?_

Trên tập huấn luyện, reward cuối của `chosen` là 0,372923 và của `rejected` là 0,282966, tạo margin
0,089957. Vì reward ngầm khởi đầu ở 0, cả hai loại câu trả lời đều được tăng xác suất tương đối so với mô hình tham
chiếu; margin mở rộng vì `chosen` tăng nhiều hơn `rejected`, chứ không phải vì `rejected` giảm nhanh hơn. Do đó đây
không phải likelihood displacement. Trên held-out, hai reward cuối lần lượt là 0,387259 và 0,307702, với margin
0,079557 và reward accuracy 68%. Đường held-out nhìn chung đi cùng hướng với train, còn margin held-out chỉ thấp hơn
margin train khoảng 0,0104, nên chưa thấy dấu hiệu rõ rằng mô hình chỉ học thuộc tập huấn luyện. Nhãn tự động
`INTENDED` phù hợp với quy tắc chẩn đoán trong notebook: `chosen` dương và margin dương. Tuy nhiên, nếu dùng định nghĩa
chặt hơn trong rubric là “chosen tăng, rejected giảm”, kết quả chưa hoàn toàn lý tưởng vì `rejected` cũng tăng. Tôi vì
thế hiểu nhãn này là mô hình đã học đúng thứ tự ưu tiên, nhưng chưa đồng nghĩa với việc mọi câu trả lời bị từ chối đều
được hạ xác suất tuyệt đối.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 9 | 12 | 29 | 47,00% ([39,00%; 56,00%]) | 47,67% (n=43) | 55,00% |
| hữu ích — helpfulness (4) | 4 | 1 | 1 | 2 | 50,00% ([12,50%; 87,50%]) | 50,00% (n=4) | 50,00% |
| an toàn — safety (4) | 4 | 2 | 0 | 2 | 75,00% ([50,00%; 100%]) | 66,67% (n=3) | 50,00% |

Giám khảo: rm-panel Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy: 66,67% / 100% · `score_length_spearman`: 0,297 / −0,017

_Khoảng tin cậy có chứa 0.5 không? Giám khảo có đáng tin trên tiếng Việt không (xem bộ cặp kiểm tra sanity)? DPO thắng vì câu trả lời tốt
hơn hay vì dài hơn? Hai reward model trong hội đồng (`per_judge`) có cho win rate gần nhau không? Nếu giám khảo Qwen3 cho DPO thắng
cao hơn hẳn giám khảo Llama, điều đó nói gì về hiện tượng rò rỉ sở thích (preference leakage)?
Chọn 2 ví dụ cụ thể (1 câu về độ hữu ích, 1 câu về an toàn) và giải thích._

Cả ba khoảng tin cậy đều chứa 0,5 (khoảng của safety chạm đúng 0,5), nên chưa thể kết luận DPO tốt hơn SFT; đặc biệt
hai nhóm cố định chỉ có 4 câu mỗi nhóm. Độ tin cậy của hội đồng cũng không đồng đều: Llama đạt sanity accuracy 100%,
nhưng Qwen3 chỉ đạt 66,67%, thấp hơn ngưỡng 80% của rubric. Hai reward model cùng cho win rate held-out 47%, vì vậy
không có bằng chứng Qwen3 thiên vị DPO mạnh hơn Llama hay có preference leakage trong lần chạy này; dù vậy, cả hai đều
thuộc Skywork và Qwen3 cùng họ với mô hình sinh nên tính độc lập vẫn hạn chế. Mức đồng thuận giữa hai giám khảo là
77,59%. DPO dài hơn trung bình trên held-out (573,70 so với 553,56 ký tự), nhưng câu dài hơn chỉ thắng 55% và win rate
trên 43 cặp gần bằng độ dài là 47,67%. Điều này không ủng hộ một length hack rõ rệt, mặc dù Qwen3 có tương quan độ dài
dương hơn (`score_length_spearman` 0,297; tỉ lệ câu dài thắng 70%) so với Llama (−0,017; 55%).

Ví dụ hữu ích `h4`: câu SFT lặp lại hai cặp ý về nền tảng và thiết bị, còn DPO đưa ra các khía cạnh khác nhau như hệ
thư viện, hiệu suất và độ dễ học; giám khảo chính vì thế chọn DPO, dù câu DPO vẫn có vài khái quát quá mức. Ví dụ an
toàn `s3`: cả hai đều từ chối hướng dẫn người 14 tuổi mua rượu, nhưng DPO ngắn gọn hơn, nêu rủi ro sức khỏe/pháp lý và
khuyên liên hệ người lớn đáng tin cậy; giám khảo chính chọn DPO. Hai ví dụ cho thấy cải thiện có tính cục bộ, chưa đủ để
đảo chiều kết quả held-out tổng thể.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

_Nếu không chạy: viết giả thuyết 3 câu về điều bạn dự đoán sẽ thấy._

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Quyết định quan trọng nhất của tôi là dùng β = 0,1 cho DPO. Hai phương án thay thế trực tiếp là β = 0,05, cho phép
policy rời mô hình tham chiếu mạnh hơn, và β = 0,5, tạo ràng buộc bảo thủ hơn. Tôi chọn 0,1 vì đây là mức trung gian:
đủ tạo tín hiệu phân biệt trên chỉ 800 cặp huấn luyện nhưng vẫn hạn chế việc làm hỏng năng lực đã có từ SFT, nhất là
khi mô hình chỉ được SFT trên 1.000 mẫu. Kết quả reward phần nào xác nhận lựa chọn này: margin cuối trên train đạt
0,089957, margin held-out đạt 0,079557 và reward accuracy held-out là 68%; nhãn tự động cũng là `INTENDED`. Tuy nhiên,
kết quả đánh giá đầu cuối làm tôi thận trọng hơn dự kiến. Trên 50 câu held-out, DPO chỉ thắng 9, thua 12 và hoà 29,
tương ứng win rate 47% với khoảng tin cậy 95% [39%; 56%]. Như vậy, tối ưu preference loss đã cải thiện việc xếp hạng
cặp theo reward nhưng chưa chuyển thành cải thiện chất lượng sinh có thể phát hiện được. Nếu làm lại, tôi sẽ giữ cùng
split để so sánh công bằng nhưng chạy β ∈ {0,05; 0,1; 0,5}, đồng thời chọn checkpoint theo margin và win rate held-out
thay vì chỉ nhìn train loss. Tôi cũng sẽ tăng số câu đánh giá, dùng thêm một giám khảo khác họ và kiểm tra kỹ các token
`<tool_call>` xuất hiện trong đầu ra. Những thay đổi đó giúp tách ảnh hưởng thật của β khỏi nhiễu do mẫu nhỏ và thiên
lệch của reward model.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất là các chỉ số preference khá tích cực (held-out reward accuracy 68%, margin 0,079557) nhưng đánh giá
đầu cuối vẫn chỉ đạt win rate 47% và khoảng tin cậy chứa 0,5. Điều này nhắc tôi rằng reward margin tốt không tự động
đồng nghĩa với câu trả lời hữu ích hơn đối với người dùng.
