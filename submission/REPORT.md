# Lab 21 — Evaluation Report

**Họ tên**: Phạm Cường Quốc  **MSSV**: 2A202602469  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4 16 GB (Colab Free) — chạy `fp16` vì T4 (Turing) không có bf16

> Mọi con số dưới đây lấy trực tiếp từ `results/` (`verdict.json`, `baselines_frozen.json`,
> `autopsy.json`, `runs.csv`, `mask_proof.json`, `template_check.json`, `token_stats.json`,
> `qualitative.json`).

---

## 0. Lựa chọn thí nghiệm và lý do

| | Lựa chọn | Lý do |
|---|---|---|
| Base model | `unsloth/Qwen3.5-4B` (mặc định tier T4) | Vừa VRAM T4 16 GB khi train LoRA 16-bit (đo được 8.78 GB peak); giữ mặc định để phép so sánh với các cấu hình đối chứng của NB4 dùng đúng cấu hình phần cứng đã được kiểm chứng. |
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (corpus mặc định) | Có thang chấm **khách quan** (so khớp từng trường), không cần LLM judge. Đây là lần chạy đầu với pipeline nên tôi giữ corpus mặc định để mọi checksum/cổng kiểm tra có ý nghĩa. |
| Report | Theo khung mẫu, bổ sung mục 0, 3.1 và phân tích lỗi ở mục 6 | — |

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (`data/train_seed.jsonl`) |
| Train / val | 225 / 25 (seed 42, `train_frac=0.9`) |
| Eval | 50 câu target · 15 câu regression — **toàn bộ tập** (`eval_limit: null`, `smoke_mode: false`) |
| `max_length` | 1024 (mặc định tier T4) — p95 đo được là **98** token, max 101, gợi ý 256 *(token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch → **30 step** (225 mẫu / batch hiệu dụng 16 ≈ 15 step/epoch) |
| LoRA `correct` | text-linear (12 module) · r=16 · α=32 · LR 1e-4 · 32,464,896 tham số train |

**Về `max_length`:** p95 = 98 token nên giá trị 256 đã thừa; tôi để 1024 của tier. Lệch
này **không ảnh hưởng kết quả**: mẫu dài nhất là 101 token nên không mẫu nào bị cắt, và
với `per_device_batch=1` không có padding nên cũng không tốn thêm compute. Nó chỉ là một
trần an toàn, không phải độ dài thật.

**Template có giữ khối `<think>` không?** **Có** — `template_check.json`:
`"verdict": "reasoning preserved — safe to train on traces"` (`open_tag_present: true`,
`body_present: true`). Trên corpus này điều đó không quan trọng: câu trả lời train là JSON
trần, và template Qwen3.5 đặt sẵn `<think>\n\n` rỗng ở cuối generation prompt (thấy trong
`masked_preview` bên dưới), nên không có trace suy luận nào trong vùng tính loss.
`valid_trace_rate = 0.0` vì vậy là đúng kỳ vọng, không phải dấu hiệu collapse.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39 / 94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn **được** tính loss (`supervised_preview`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn **bị che** (`masked_preview`):

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>

```

System prompt, ticket và header assistant đều bị che; chỉ JSON + `<|im_end|>` được học.
`supervised_fraction` = 0.41, xa ngưỡng 0.95 của lỗi "tính loss cả prompt". `<|im_end|>`
nằm trong loss nên model học được điểm dừng — khớp với `format = 1.0` ở mục 3.

---

## 3. Ba baseline (NB2) và bản fine-tune (NB5)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.0 | 3205.5 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.0 | 988.3 |
| (c) LoRA fine-tune (`correct`) | **0.970** | 0.6111 | 1.0 | 1386.4 |

*n = 50 câu target, 15 câu regression cho cả ba dòng.*

**(b) có thật sự mạnh hơn (a) không?** **Có**, cách biệt rất lớn: target 0.000 → 0.765,
format 0.0 → 1.0, latency giảm hơn 3 lần. Với prompt ngây thơ, base model không trả về
JSON hợp lệ lần nào (format 0.0) và viết dài (3.2 s/mẫu); prompt tối ưu liệt kê đủ khoá và
miền giá trị nên base model đã làm được 76.5% bài toán mà không cần train.

**Có sửa `OPTIMIZED_PROMPT` không?** **Không.** SHA `719e74d3b6232053` khớp bản gốc —
`make verify`: `baseline (b) prompt unmodified ✓`.

### 3.1 Khai báo về thứ tự đo

Lần chạy đầu tiên của tôi để nhầm `EVAL_LIMIT=8` (mặc định của ô Colab). Lần đó NB2 **đã
chạy trước khi train** (trên 8 câu: (a)=0.000, (b)=0.688). Sau khi `make verify` báo FAIL
`full eval set used`, tôi chạy lại **NB2 + NB5** với toàn bộ tập eval, dùng nguyên các
adapter đã train. Vì vậy bộ số n=50 của (a)/(b) ở trên được đo **sau** khi train.

Tôi cho rằng điều này không làm hỏng phép so sánh vì: (1) prompt (b) không đổi (SHA khớp);
(2) tập eval không đổi (`eval sets unmodified ✓` — checksum); (3) không có tham số nào
được chỉnh giữa hai lần — không có cơ hội "vô thức tối ưu" baseline theo kết quả
fine-tune. Nhưng tôi ghi rõ ở đây vì rubric yêu cầu đo (b) trước khi train.

---

## 4. Giải phẫu cấu hình sai (NB4)

Cả bốn run cùng `max_steps = 30` (`make verify`: `all runs share ONE step budget ✓`).

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | format | train s | VRAM GB | latency ms |
|---|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear (12) | 16 | 32,464,896 | 1e-4 | 0.6256 | **0.97** | 1.0 | 386.7 | 8.78 | 1386.4 |
| `attn_only` | q,v (2) | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5378 | **0.97** | 1.0 | 271.7 | 8.79 | 881.6 |
| `wrong_lr` | text-linear (12) | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.00** | 0.0 | 403.5 | 8.78 | 5195.1 |
| `qlora` | text-linear (12) | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.94** | 1.0 | 471.8 | 3.86 | 1740.8 |

**Biến duy nhất bị đổi ở mỗi run** (so với `correct`):
`attn_only` → vị trí gắn adapter (rank tự giải để giữ nguyên ngân sách tham số, lệch
8,192 tham số ≈ 0.03%) · `wrong_lr` → learning rate (÷10) · `qlora` → độ chính xác trọng
số base (4-bit thay 16-bit).

**Xếp hạng:**
- Theo **target** (thang đo thật): `correct` = `attn_only` (0.97) > `qlora` (0.94) ≫ `wrong_lr` (0.00)
- Theo **train loss**: `attn_only` (0.538) < `correct` (0.626) < `qlora` (0.706) < `wrong_lr` (1.570)

Hai thứ tự **khác nhau ở đầu bảng**: loss nói `attn_only` tốt hơn `correct`, target nói
hai run hoà.

**4.1 — `attn_only` vs `correct` (vị trí vs rank).**
Với cùng ~32.46M tham số, `attn_only` (q,v @ r=283) **hoà** `correct` (all-linear @ r=16)
trên target: 0.97 = 0.97, format đều 1.0. Nhưng train loss của `attn_only` thấp hơn rõ
(0.538 vs 0.626) — nếu chấm bằng loss tôi sẽ kết luận sai rằng "attention-only tốt hơn".
Loss thấp hơn mà target không cao hơn cho thấy phần chênh loss là khả năng khớp tập train
(r=283 trên 2 module cho mỗi module rất nhiều chiều để ghi nhớ 225 mẫu), không phải năng
lực tổng quát hoá. Về đòn bẩy: khi ngân sách tham số đã khớp, **vị trí không phải đòn bẩy
trên tác vụ này** — triage JSON là tác vụ hẹp, base đã làm được 76.5% chỉ bằng prompt, nên
cả hai cách đặt adapter đều đủ dung lượng để đạt trần ~0.97. Kết luận "all-linear tốt hơn"
của deck §11.2 vì vậy không được tái hiện ở đây; tôi không có bằng chứng để khẳng định nó
cho tác vụ này. Một khác biệt đo được là **latency khi phục vụ không merge**: `attn_only`
881.6 ms vs `correct` 1386.4 ms — adapter rải trên 12 module/layer tốn thêm nhiều phép
nhân phụ hơn adapter trên 2 module. Khác biệt này biến mất nếu merge (NB6).

**4.2 — `wrong_lr` (LR 1e-5 thay 1e-4).**
Chỉ khác một con số nhưng kết quả sụp hoàn toàn: final loss 1.570 (gấp ~2.5 lần
`correct`), target 0.00, format 0.0, latency 5195 ms. Bộ số này gần như trùng với
baseline (a) (0.000 / 0.0 / 3205 ms) — hợp lý, vì adapter được train và chấm với chính
system prompt ngây thơ `"Phân loại ticket sau."`; sau 30 step với LR thang full-FT,
adapter gần như chưa dịch chuyển model khỏi hành vi base, nên model vẫn viết tự do,
không ra JSON. Nếu chỉ nhìn loss mà không biết LR, tôi có thể kết luận sai rằng "LoRA học
kém / dữ liệu quá khó / cần rank cao hơn" và đi tăng rank — trong khi đòn bẩy thật là LR.
Đây là **biến có tác động lớn nhất** trong cả 4 run: 0.97 → 0.00.

**4.3 — `qlora` (4-bit).**
Tiết kiệm **4.92 GB** VRAM peak (8.78 → 3.86 GB, −56%). Cái giá: target giảm 0.03
(0.97 → 0.94), train chậm hơn 22% (471.8 vs 386.7 s) và suy luận chậm hơn 26%
(1740.8 vs 1386.4 ms) do chi phí giải lượng tử hoá. Số đo **ủng hộ có điều kiện**
khuyến nghị "không dùng QLoRA cho Qwen3.5": trên T4 16 GB, LoRA 16-bit đã vừa với
8.78 GB, nên QLoRA chỉ mang lại thiệt hại (chất lượng + tốc độ) mà không mở khoá được gì.
QLoRA chỉ đáng khi bản 16-bit **không vừa** (ví dụ 9B trên T4). Lưu ý: chênh 0.03 trên 50
câu là nhỏ (≈6 trường sai thêm trên 200 trường), một lần chạy chưa đủ để coi là chắc chắn.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.205` · `regression Δ = −0.180` · `valid_trace_rate = 0.0`

Lý do do cổng ghi: *"general capability regressed by 0.180 (tolerance 0.020)"*.

**Diễn giải.** Bản fine-tune làm đúng điều nó được train để làm: trên tập target nó vượt
baseline (b) +0.205 (0.765 → 0.970) và giữ format 1.0. Theo từng câu
(`results/per_item.json`), FT **thắng (b) ở 33/50 ticket, hoà 17, không thua câu nào**.
Nếu chỉ nhìn cột target, đây là một thành công rõ.

Nhưng cổng hồi quy đánh giá đồng thời năng lực chung, và ở đó model mất 0.180 điểm
(0.7911 → 0.6111) trên 15 câu hỏi phổ thông, gấp 9 lần ngưỡng 0.02. Dữ liệu từng câu cho
thấy **cơ chế** của hồi quy này: **14/15 câu trả lời phổ thông của FT bắt đầu bằng `{` và
chứa khoá `"intent"`**, so với 0/15 của (b). Model không chỉ "quên kiến thức" mà đã
**sụp định dạng** — nó coi mọi đầu vào là một ticket cần phân loại:

- *"1 km bằng bao nhiêu mét?"* → `{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh", …}` — không có câu trả lời.
- *"Một năm có bao nhiêu tháng?"* → `{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}` — đúng y schema triage.
- *"Viết một câu chúc mừng sinh nhật"* → `{"intent": "chuc_mung_sinh_nhat", "urgency": "trung_tinh", …}`.

Ở các câu khác, model vẫn còn kiến thức nhưng nhét nó vào JSON
(`{"intent": "trivia", "answer": "Hà Nội", …}`) — và cách chấm keyword-recall vẫn cho
điểm những câu này. Nghĩa là **con số −0.180 còn đánh giá thấp mức hồi quy về hành vi**:
dưới một thang đo khắt khe hơn (yêu cầu trả lời văn xuôi), gần như 15/15 câu đều hỏng.

Nguyên nhân hợp lý: 100% dữ liệu train là một định dạng đầu ra duy nhất (JSON 4 khoá,
~40 token), không có mẫu nào yêu cầu trả lời văn xuôi; và vì adapter được train với system
prompt ngắn `"Phân loại ticket sau."`, model học liên kết "bất kỳ input nào → JSON triage"
thay vì "ticket → JSON". 30 step với LR 1e-4 trên toàn bộ linear layer đủ để liên kết đó
lấn át hành vi trò chuyện của base model.

Một điểm cần trung thực: câu duy nhất FT **hơn** (b) ở regression (*"2 mũ 10"*, 0 → 1)
không phải do FT giỏi hơn, mà vì (b) giải từng bước dài dòng và bị cắt ở `max_new_tokens=96`
trước khi viết ra 1024; FT trả lời ngắn trong JSON nên kịp ghi đáp số.

Điều đó nói gì về bài toán: triage 4 trường là tác vụ prompt tốt đã giải phần lớn (0.765),
nên phần lợi cho fine-tune là +0.205 — có thật, nhưng chi phí ẩn (sụp định dạng ngoài miền,
latency +40%) lớn hơn. Fine-tune này **chưa nên deploy**. Hướng sửa đúng là trộn 1–5% dữ
liệu phổ thông trả lời văn xuôi (replay, deck §6.3) và đo lại — không phải nới ngưỡng.

---

## 6. Định tính — có cả ca THUA

Nguồn: `results/per_item.json` — sinh lại dự đoán (b) và (c) cho toàn bộ tập eval, cùng
prompt, cùng adapter, cùng greedy decode như NB2/NB5. Điểm trung bình tái lập **khớp chính
xác** số đã đóng băng: target (b) 0.765 / FT 0.970, regression (b) 0.7911 / FT 0.6111.

| # | Nguồn | Đầu vào (rút gọn) | Nhãn đúng / keyword | (b) base + optimized prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|---|
| 1 | target i=7 | "…máy xay sinh tố… Muốn đổi. Đã 3 ngày…" | doi_tra · trung_binh | van_chuyen · **cao** (0.50) | doi_tra · trung_binh (**1.00**) | ✅ FT thắng — (b) nhầm intent và thổi phồng urgency |
| 2 | target i=30 | "…đèn bàn LED… **Hoàn lại.** Sớm nhé. Lần cuối mua ở đây." | doi_tra · trung_binh · tieu_cuc | **hoan_tien** · **cao** (0.50) | doi_tra · trung_binh (**1.00**) | ✅ FT thắng — học được quy ước "Hoàn lại" = trả hàng, không phải hoàn tiền |
| 3 | target i=3 | "…bình giữ nhiệt… Chưa thấy tiền. **Khi nào tiện.**" | urgency **thap** | urgency **trung_binh** (0.75) | urgency **trung_binh** (0.75) | ❌ FT sai — **cùng lỗi với (b)**, fine-tune không sửa được |
| 4 | regression #2 | "1 km bằng bao nhiêu mét?" | `1000` | "…**1 km = 1000 m**." (1.0) | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, …}` (**0.0**) | ❌ **FT thua (b)** — trả về JSON triage thay vì câu trả lời |
| 5 | regression #9 | "Một năm có bao nhiêu tháng?" | `12` | "Một năm bình thường có **12 tháng**." (1.0) | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}` (**0.0**) | ❌ **FT thua (b)** — sụp định dạng hoàn toàn |
| 6 | regression #3 | "Viết một câu chúc mừng sinh nhật bằng tiếng Việt." | `sinh nhật` | "Chúc bạn một ngày sinh nhật thật vui vẻ…" (1.0) | `{"intent": "chuc_mung_sinh_nhat", "urgency": "trung_tinh", …}` (**0.0**) | ❌ **FT thua (b)** — không làm được tác vụ sinh văn bản tự do |

**Tổng hợp theo từng câu:**
- **Target:** FT > (b) ở 33 câu, bằng ở 17, **thua ở 0 câu**. Lỗi phổ biến nhất của (b) là
  nhầm `doi_tra` ↔ `hoan_tien` (11 lần) và thổi phồng urgency `trung_binh` → `cao` (9 lần);
  FT sửa hết các lỗi này.
- **Regression:** FT thua (b) ở 5/15 câu (#2, #3, #9, #11, #14), thắng 1 câu (#6, do (b) bị
  cắt token — xem mục 5). **Mọi ca FT thua (b) đều nằm ngoài miền train.**

**Mẫu chung ở các ca FT sai — có hai mẫu, đều rõ:**

1. **Ngoài miền → sụp định dạng.** Mọi câu FT thua (b) đều là câu hỏi phổ thông, và FT
   đều trả về JSON (14/15). Đây là chi phí thật của fine-tune trên dữ liệu một định dạng.
2. **Trong miền → lỗi "Khi nào tiện" kế thừa từ base.** Cả **6/6** câu target FT sai (i = 3,
   5, 12, 39, 41, 46) đều sai **urgency**, đều đoán `trung_binh` thay vì `thap`, và đều chứa
   cụm **"Khi nào tiện"** — đó cũng là toàn bộ 6 câu có cụm này trong tập eval. Dữ liệu từng
   câu cho thấy **(b) cũng sai urgency ở cả 6 câu này** (4 câu đoán `trung_binh`, 2 câu đoán
   `cao`; 0/6 đoán `thap`). Trong khi đó, trong dữ liệu train cụm "Khi nào tiện" xuất hiện 35
   lần và **cả 35 đều là `thap`**. Vậy đây là một tiên nghiệm sai của base model ("khi
   nào…" đọc như đang hỏi thời hạn ⇒ không thể là "thấp"), và 30 step fine-tune chỉ kéo được
   nó từ `cao` về `trung_binh` chứ chưa đến `thap`. Fine-tune sửa được những lỗi mà prompt
   gây ra (nhầm intent), nhưng chưa ghi đè được một tiên nghiệm ngôn ngữ mạnh — đây là lỗi
   **có hệ thống**, có thể sửa bằng thêm step hoặc nêu quy tắc urgency ngay trong prompt.

---

## 7. Kết luận & điều tôi học được

### Kết luận

**Tôi không deploy bản fine-tune này ở dạng hiện tại.** Lý do không phải vì nó yếu ở bài
toán chính — ngược lại, nó thắng baseline (b) 0.970 vs 0.765, thắng 33/50 ticket và không
thua ticket nào. Lý do là **cái giá nó phải trả để có +0.205 đó**:

1. **Sụp định dạng ngoài miền.** 14/15 câu hỏi phổ thông bị trả lời bằng JSON; regression
   giảm 0.180, gấp 9 lần ngưỡng 0.02. Chuỗi nhân quả rõ: 100% dữ liệu train là cùng một
   định dạng đầu ra → model không có tín hiệu nào cho thấy còn tồn tại kiểu trả lời khác →
   cộng thêm system prompt ngắn `"Phân loại ticket sau."` dùng khi train, model học
   "mọi input → JSON triage" thay vì "ticket → JSON triage".
2. **Phần lớn giá trị đã có sẵn từ prompt.** Chỉ đổi prompt ngây thơ sang prompt tối ưu đã
   đưa target từ 0.000 lên 0.765 mà không tốn một step train nào. Fine-tune chỉ đóng góp
   phần còn lại, chủ yếu là sửa nhầm `doi_tra` ↔ `hoan_tien` (11 lỗi của (b)).
3. **Latency +40%** (1386 vs 988 ms) khi phục vụ adapter chưa merge.

Phán quyết này **phụ thuộc vào cách triển khai**, và tôi nói rõ ranh giới: nếu model đứng
sau một bộ định tuyến chỉ chuyển cho nó ticket đã được lọc, thì sụp định dạng ngoài miền
không bao giờ bị kích hoạt, và +0.205 target (sau khi merge để xoá chi phí latency) là
một cải thiện đáng giá. Nhưng cổng hồi quy của lab giả định model là một trợ lý dùng chung,
và trong giả định đó, một model trả `{"intent": "hoi_thong_tin", …}` cho câu "1 km bằng
bao nhiêu mét?" là một lỗi sản phẩm. Tôi không nới ngưỡng để đổi kết luận.

**Đòn bẩy thật sự trong lab, xếp theo bằng chứng đo được:**

| Hạng | Đòn bẩy | Bằng chứng |
|---|---|---|
| 1 | **Learning rate** | đổi đúng một con số (1e-4 → 1e-5): target 0.97 → **0.00** |
| 2 | **Độ phủ dữ liệu** | một định dạng duy nhất → 14/15 câu phổ thông thành JSON; quy ước "Khi nào tiện" nhất quán 35/35 vẫn chưa ghi đè được tiên nghiệm của base |
| 3 | **Prompt** | (a) → (b): target 0.000 → 0.765, không train |
| 4 | **Mask + template** | điều kiện cần (supervised_fraction 0.41, `<|im_end|>` trong loss → format 1.0); không chạy đối chứng `everything` nên không lượng hoá được |
| — | Vị trí adapter | **không phải đòn bẩy ở đây**: cùng ngân sách tham số, attention-only hoà all-linear (0.97 = 0.97) |
| — | QLoRA | chỉ là đánh đổi: −4.92 GB VRAM, đổi lấy −0.03 target và chậm hơn ~25%; không cần trên T4 cho model 4B |

Bước tiếp theo nếu được làm tiếp không phải là chỉnh rank hay vị trí, mà là **sửa dữ
liệu**: trộn 1–5% câu hỏi phổ thông có câu trả lời văn xuôi (replay, deck §6.3), train lại,
và kiểm tra xem regression có về trong ngưỡng trong khi target vẫn giữ được trên 0.765.

### Ba điều tôi học được

**1. Trước lab tôi nghĩ "fine-tune xong mà loss giảm là thành công". Giờ tôi không còn
tin vậy.** Ba bằng chứng từ chính lần chạy của tôi: `attn_only` có loss thấp nhất (0.538)
nhưng chỉ hoà `correct` trên target; `wrong_lr` vẫn chạy đủ 30 step, không lỗi, loss vẫn
giảm — nhưng target 0.00, hành vi y hệt base model; và `correct` thắng target rõ ràng mà vẫn
FAIL vì hỏng năng lực chung. Nếu chỉ nhìn đường loss, tôi đã sai cả ba lần. Từ giờ, câu đầu
tiên tôi hỏi sau khi train là "output thật trông thế nào, và nó thắng cái gì?".

**2. Đọc từng câu đã thay đổi kết luận của tôi hai lần.** Lần thứ nhất: con số
"regression −0.18" khiến tôi nghĩ model *quên vài kiến thức*. Đọc `per_item.json` mới thấy
nó **trả JSON cho 14/15 câu hỏi phổ thông** — một kiểu hỏng khác hẳn, và nặng hơn con số,
vì thang keyword-recall vẫn cho điểm câu trả lời bị nhét trong JSON. Lần thứ hai: tôi đã
viết trong bản nháp report rằng 6 lỗi urgency là do fine-tune *chưa học kịp* quy ước
"Khi nào tiện". Khi có dự đoán từng câu của (b), tôi thấy base model cũng sai đúng 6 câu
đó, không câu nào đoán `thap` — tức đây là tiên nghiệm sai **kế thừa từ base**, và
fine-tune đã kéo nó từ `cao` về `trung_binh` chứ không gây ra nó. Tôi đã phải sửa lại
giả thuyết của chính mình. Bài học: con số tổng cho biết *có vấn đề*, chỉ dữ liệu từng câu
mới cho biết *vấn đề là gì*.

**3. Cổng kiểm tra tự động bắt được lỗi mà tôi bỏ qua.** Lần chạy đầu tôi để nguyên
`EVAL_LIMIT=8` (mặc định của ô Colab) và đã có ngay một bộ số trông rất đẹp —
target 0.938 trên 8 câu. Chính `make verify` báo FAIL và buộc tôi chạy lại trên đủ 50 câu.
Con số đổi (0.938 → 0.970 cho FT, 0.688 → 0.765 cho (b)) và kết luận về độ lớn của hồi quy
cũng đổi (−0.125 → −0.180). Một con số trên 8 câu nghe có vẻ chắc chắn nhưng chỉ cần sai
thêm một câu regression là lệch 0.125. Tôi hiểu ra vì sao lab đặt checksum, SHA prompt và
kiểm tra `EVAL_LIMIT` thành cổng cứng: chúng bảo vệ phép so sánh khỏi chính sự bất cẩn của
người làm thí nghiệm.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử
- **Replay data** (ưu tiên số 1): trộn 3–5% câu hỏi phổ thông có câu trả lời văn xuôi vào
  tập train, chạy lại NB3 + NB5, đo xem regression có về trong ngưỡng 0.02 không và target
  còn giữ được bao nhiêu trong +0.205.
- **Thang regression khắt khe hơn**: phạt câu trả lời dạng JSON cho câu hỏi phổ thông, để
  con số phản ánh đúng mức sụp định dạng thay vì chỉ đếm từ khoá.
- **Kiểm tra giả thuyết "Khi nào tiện"**: thêm 1 epoch, và song song thêm quy tắc
  "Khi nào tiện → thap" vào prompt (b) — nếu (b) đạt được bằng prompt thì đó lại thêm một
  bằng chứng rằng bài toán này không cần fine-tune.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] **B5 HuggingFace Hub** — adapter `correct` (LoRA all-linear r=16, 30 step), công khai:
  **https://huggingface.co/pcq301/qwen3.5-4b-lora-cskh-triage-lab21**
  (base `unsloth/Qwen3.5-4B`; gồm `adapter_model.safetensors` + `adapter_config.json`)

**Artefact bổ sung ngoài pipeline:** `results/per_item.json` — dự đoán từng câu của (b) và
(c) trên toàn bộ tập target + regression, sinh trên Colab T4 sau NB5 bằng đúng hàm
`generate.generate_batch` và prompt của NB2/NB5; tái lập khớp chính xác các điểm đã đóng băng.
