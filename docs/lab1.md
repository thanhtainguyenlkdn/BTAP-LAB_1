# PHÂN TÍCH Ý TƯỞNG, MỤC TIÊU VÀ QUY TRÌNH THỰC HIỆN BÀI TOÁN PHÂN LOẠI ẢNH FASHION-MNIST BẰNG CNN (PYTORCH)

## Giới thiệu

Tài liệu trình bày quy trình xây dựng bài toán phân loại hình ảnh thời
trang bằng mạng nơ-ron tích chập (CNN) với PyTorch. Nội dung được sắp
xếp theo trình tự: **Preview → Design → Implementation → Training →
Evaluation → Inference → Conclusion**. Mục tiêu là làm rõ bài toán cần
giải quyết, lý do lựa chọn phương pháp, cách triển khai từng giai đoạn
và kết quả cần đánh giá.

------------------------------------------------------------------------

# PHẦN I. PREVIEW -- TỔNG QUAN BÀI TOÁN

## 1. WHAT -- Bài toán là gì?

Xây dựng mô hình học sâu sử dụng **Convolutional Neural Network (CNN)**
bằng PyTorch để nhận diện và phân loại hình ảnh các mặt hàng thời trang.

-   **Đầu vào:** Ảnh thời trang grayscale, kích thước `28 × 28` pixel,
    một kênh màu.
-   **Mô hình:** CNN được xây dựng và huấn luyện bằng PyTorch.
-   **Đầu ra:** Một trong 10 nhãn thời trang tương ứng với hình ảnh đầu
    vào.

### 10 lớp cần phân loại

    Nhãn Tên lớp       Ý nghĩa
  ------ ------------- -------------------
       0 T-shirt/top   Áo phông, áo kiểu
       1 Trouser       Quần dài
       2 Pullover      Áo len
       3 Dress         Váy
       4 Coat          Áo khoác
       5 Sandal        Dép sandal
       6 Shirt         Áo sơ mi
       7 Sneaker       Giày thể thao
       8 Bag           Túi xách
       9 Ankle boot    Bốt cổ thấp

Ví dụ: Khi đưa ảnh giày thể thao vào mô hình, CNN trích xuất các đặc
trưng của ảnh và dự đoán nhãn `Sneaker`.

## 2. WHY -- Tại sao thực hiện bài toán?

Bài toán giúp vận dụng kiến thức học sâu vào phân loại ảnh, đồng thời
tìm hiểu toàn bộ quy trình từ xử lý dữ liệu đến đánh giá và sử dụng mô
hình.

Các mục tiêu chính:

-   **Ứng dụng CNN:** Hiểu cách mạng tích chập trích xuất đặc trưng hình
    ảnh.
-   **Xử lý dữ liệu:** Tải, kiểm tra, trực quan hóa, chuẩn hóa và chia
    dữ liệu.
-   **Huấn luyện:** Sử dụng hàm mất mát, thuật toán tối ưu và các kỹ
    thuật hạn chế overfitting.
-   **Đánh giá:** Kiểm tra độ chính xác và hiệu suất phân loại của từng
    lớp.
-   **Phân tích lỗi:** Xác định những nhóm thời trang mô hình thường
    nhận diện nhầm.
-   **Dự đoán:** Đưa ảnh mới vào mô hình và nhận kết quả cùng xác suất
    dự đoán.

Một khó khăn đáng chú ý là một số loại áo có hình dạng tương đối giống
nhau, chẳng hạn `T-shirt/top`, `Pullover`, `Coat` và `Shirt`. Mô hình
cần học được những đặc trưng phù hợp để phân biệt các lớp này.

## 3. WHEN -- Khi nào thực hiện?

Bài toán được triển khai theo thứ tự:

  -----------------------------------------------------------------------
  Giai đoạn               Thời điểm               Mục đích
  ----------------------- ----------------------- -----------------------
  Khám phá dữ liệu        Trước huấn luyện        Hiểu cấu trúc và đặc
                                                  điểm dữ liệu

  Thiết kế mô hình        Trước huấn luyện        Xác định kiến trúc CNN
                                                  và thông số

  Huấn luyện              Sau khi chuẩn bị dữ     Giúp CNN học đặc trưng
                          liệu, mô hình           

  Validation              Trong từng Epoch        Theo dõi khả năng tổng
                                                  quát hóa

  Đánh giá Test           Sau huấn luyện          Kiểm tra kết quả cuối
                                                  cùng

  Inference               Sau khi chọn mô hình    Dự đoán ảnh riêng lẻ
                          tốt nhất                
  -----------------------------------------------------------------------

**Lưu ý:** Không sử dụng tập Test để cập nhật trọng số hoặc lựa chọn mô
hình trong quá trình huấn luyện. Tập Test được giữ riêng để đánh giá
cuối cùng.

## 4. WHERE -- Thực hiện ở đâu?

Dự án sử dụng Python, PyTorch và Jupyter Notebook.

  Công cụ/thư viện   Vai trò
  ------------------ --------------------------------------------
  Python             Ngôn ngữ lập trình
  PyTorch            Xây dựng và huấn luyện CNN
  NumPy, Pandas      Xử lý và thống kê dữ liệu
  Matplotlib         Hiển thị ảnh và biểu đồ
  Scikit-learn       Tính Accuracy, Precision, Recall, F1-score
  Jupyter Notebook   Thực hiện và kiểm tra từng giai đoạn

Mã nguồn được chia thành các tệp chức năng trong thư mục `src/` và ba
Notebook tương ứng với ba giai đoạn chính.

## 5. WHICH -- Sử dụng những gì?

### 5.1. Bộ dữ liệu Fashion-MNIST

Sử dụng bộ dữ liệu **Fashion-MNIST** của Zalando Research.

-   Tổng cộng: **70.000 ảnh**.
-   Train ban đầu: **60.000 ảnh**.
-   Test: **10.000 ảnh**.
-   Kích thước: `28 × 28` pixel.
-   Ảnh grayscale, có một kênh màu.
-   Có 10 lớp; mỗi lớp có 6.000 ảnh Train ban đầu và 1.000 ảnh Test.

### 5.2. Các thành phần kỹ thuật

  -----------------------------------------------------------------------
  Thành phần                          Chức năng
  ----------------------------------- -----------------------------------
  CNN                                 Trích xuất đặc trưng hình ảnh

  Conv2D                              Phát hiện đặc trưng thông qua phép
                                      tích chập

  BatchNorm                           Chuẩn hóa, hỗ trợ ổn định quá trình
                                      học

  ReLU                                Hàm kích hoạt tạo tính phi tuyến

  MaxPooling                          Giảm kích thước bản đồ đặc trưng

  Dropout                             Hạn chế overfitting

  CrossEntropyLoss                    Đo sai lệch giữa dự đoán và nhãn
                                      thật

  AdamW                               Cập nhật trọng số mô hình

  ReduceLROnPlateau                   Điều chỉnh Learning Rate khi kết
                                      quả không cải thiện

  Early Stopping                      Dừng huấn luyện khi mô hình không
                                      còn cải thiện
  -----------------------------------------------------------------------

## 6. HOW -- Quy trình tổng quát

Toàn bộ bài toán gồm ba giai đoạn:

1.  **Data Exploration -- Khám phá dữ liệu:** Tải dữ liệu → Kiểm tra →
    Thống kê → Trực quan hóa.
2.  **Model Training -- Xây dựng và huấn luyện:** Tiền xử lý → Chia dữ
    liệu → Xây dựng CNN → Huấn luyện.
3.  **Evaluation & Inference -- Đánh giá và dự đoán:** Nạp mô hình →
    Đánh giá Test → Phân tích lỗi → Dự đoán ảnh.

Mục tiêu cuối cùng là tạo ra mô hình có khả năng phân loại 10 nhóm thời
trang với kết quả tốt trên dữ liệu chưa dùng để huấn luyện.

------------------------------------------------------------------------

# PHẦN II. DESIGN -- THIẾT KẾ GIẢI PHÁP

## 1. Thiết kế đầu vào và đầu ra

Ảnh đầu vào là ảnh grayscale kích thước `28 × 28`. Trước khi chuẩn hóa,
mỗi pixel có giá trị từ 0 đến 255.

Quy trình xử lý ảnh:

1.  Nhận ảnh đầu vào.
2.  Chuyển ảnh thành Tensor.
3.  Chuẩn hóa giá trị pixel.
4.  Đưa ảnh qua các lớp CNN.
5.  Trích xuất đặc trưng.
6.  Phân loại thành 10 lớp.
7.  Trả về nhãn dự đoán.

## 2. Thiết kế cách chia dữ liệu

Từ 60.000 ảnh Train ban đầu, chia ngẫu nhiên thành Train và Validation
theo tỷ lệ 90:10. Giữ riêng 10.000 ảnh Test.

  Tập dữ liệu         Số ảnh Vai trò
  ------------- ------------ --------------------------------------------
  Train               54.000 Huấn luyện và cập nhật trọng số
  Validation           6.000 Theo dõi Loss, Accuracy và chọn checkpoint
  Test                10.000 Đánh giá cuối cùng sau huấn luyện
  **Tổng**        **70.000** 

Trong code, việc chia Train và Validation được thực hiện bằng cách xáo
trộn chỉ số với `seed = 42`, sau đó lấy 10% làm Validation. Đây là phép
chia ngẫu nhiên thông thường, không phải Stratified Split.

## 3. Thiết kế kiến trúc CNN

Kiến trúc gồm **hai khối trích xuất đặc trưng** và **một khối phân
loại**.

### 3.1. Khối trích xuất đặc trưng

**CNN Block 1**

-   Đầu vào: `1 × 28 × 28`.
-   Conv2D: 1 kênh đầu vào, 32 bộ lọc.
-   BatchNorm và ReLU.
-   Conv2D tiếp theo: 32 bộ lọc.
-   MaxPooling: giảm kích thước từ `28 × 28` xuống `14 × 14`.
-   Dropout2D: tỷ lệ `0.25`.

**CNN Block 2**

-   Đầu vào: `32 × 14 × 14`.
-   Conv2D: 64 bộ lọc.
-   BatchNorm và ReLU.
-   Conv2D tiếp theo: 64 bộ lọc.
-   MaxPooling: giảm kích thước từ `14 × 14` xuống `7 × 7`.
-   Dropout2D: tỷ lệ `0.25`.

### 3.2. Khối phân loại

Sau khi trích xuất đặc trưng:

1.  **Flatten:** Chuyển Tensor `64 × 7 × 7` thành vector gồm `3.136` giá
    trị.
2.  **Linear:** Chuyển `3.136` đặc trưng thành 256 neuron.
3.  **BatchNorm1D và ReLU:** Chuẩn hóa và kích hoạt.
4.  **Dropout:** Tỷ lệ `0.5`.
5.  **Linear cuối:** Chuyển 256 neuron thành 10 đầu ra tương ứng với 10
    lớp.

Kiến trúc có khoảng **0,87 triệu tham số có thể học**.

### 3.3. Luồng xử lý của CNN

``` text
Ảnh đầu vào: 1 × 28 × 28
          |
          v
CNN Block 1
Conv → BatchNorm → ReLU
Conv → BatchNorm → ReLU
MaxPool → Dropout
          |
          v
Đặc trưng: 32 × 14 × 14
          |
          v
CNN Block 2
Conv → BatchNorm → ReLU
Conv → BatchNorm → ReLU
MaxPool → Dropout
          |
          v
Đặc trưng: 64 × 7 × 7
          |
          v
Flatten: 3.136 giá trị
          |
          v
Fully Connected: 3.136 → 256 → 10
          |
          v
Dự đoán 1 trong 10 lớp
```

## 4. Thiết kế phương pháp huấn luyện

  Thông số                         Giá trị
  -------------------- -------------------
  Epoch tối đa                          30
  Batch Size                            64
  Learning Rate                      0.001
  Optimizer                          AdamW
  Weight Decay                      0.0001
  Loss Function           CrossEntropyLoss
  Scheduler              ReduceLROnPlateau
  Scheduler patience                     2
  Early Stopping                   5 Epoch
  Seed                                  42

Ý nghĩa các lựa chọn:

-   **AdamW:** Dùng gradient để cập nhật trọng số nhằm giảm Loss.
-   **CrossEntropyLoss:** Đo sai lệch giữa kết quả dự đoán và nhãn thật.
-   **ReduceLROnPlateau:** Giảm Learning Rate khi Validation Loss không
    cải thiện theo điều kiện đã đặt.
-   **Dropout và Weight Decay:** Hạn chế mô hình phụ thuộc quá mức vào
    dữ liệu Train.
-   **Early Stopping:** Dừng huấn luyện nếu Validation Loss không cải
    thiện trong 5 Epoch liên tiếp.

Các thông số được khai báo trong `config.py` và sử dụng trong Notebook
huấn luyện.

------------------------------------------------------------------------

# PHẦN III. IMPLEMENTATION -- TRIỂN KHAI BÀI TOÁN

Dự án có ba Notebook, mỗi Notebook đảm nhiệm một nhiệm vụ riêng.

# GIAI ĐOẠN 1. DATA EXPLORATION -- KHÁM PHÁ DỮ LIỆU

**Notebook:** `01_data_exploration.ipynb`

## Bước 1. Chuẩn bị môi trường

**Mục tiêu:** Khởi tạo môi trường và các thư viện cần thiết.

Công việc:

-   Import NumPy để xử lý mảng dữ liệu.
-   Import Pandas để thống kê dữ liệu.
-   Import Matplotlib để hiển thị ảnh và biểu đồ.
-   Import các module `config`, `data`, `utils`.
-   Cố định Seed = 42 để kết quả có thể tái lập.

**Kết quả mong muốn:** Môi trường sẵn sàng để tải và phân tích dữ liệu.

## Bước 2. Tải và đọc dữ liệu

**Mục tiêu:** Đưa Fashion-MNIST vào chương trình.

Quy trình:

1.  Kiểm tra dữ liệu CSV có sẵn trong thư mục `data/` hay không.
2.  Nếu chưa có, thử tải dữ liệu từ Kaggle bằng `kagglehub`.
3.  Nếu không tải được, sử dụng `torchvision.datasets.FashionMNIST`.
4.  Đọc hình ảnh và nhãn tương ứng.

Các dữ liệu thu được:

-   `X_train`, `y_train`: Ảnh và nhãn huấn luyện.
-   `X_test`, `y_test`: Ảnh và nhãn kiểm tra.

**Kết quả mong muốn:** Có đầy đủ hình ảnh và nhãn để phân tích.

## Bước 3. Kiểm tra phân bố dữ liệu

**Mục tiêu:** Xác định số lượng ảnh trong từng lớp có cân bằng hay
không.

Công việc:

-   Đếm số ảnh của 10 lớp.
-   Tính tỷ lệ phần trăm của từng lớp.
-   Tính độ lệch chuẩn số lượng ảnh giữa các lớp.
-   Vẽ biểu đồ phân bố dữ liệu.

**Kết quả cần nhận xét:** Trong dữ liệu ban đầu, mỗi lớp có 6.000 ảnh
Train và 1.000 ảnh Test. Do dữ liệu cân bằng, Accuracy có thể được dùng
làm chỉ số đánh giá chính; tuy nhiên vẫn cần xem Precision, Recall và
F1-score để đánh giá từng lớp.

## Bước 4. Trực quan hóa hình ảnh

**Mục tiêu:** Quan sát dữ liệu thực tế để hiểu đặc điểm hình ảnh.

Thực hiện:

-   Hiển thị 16 ảnh ngẫu nhiên để quan sát sự đa dạng.
-   Hiển thị 10 ảnh đại diện, mỗi lớp một ảnh.

**Kết quả mong muốn:** Nhận biết hình dạng, đặc điểm và sự khác biệt
giữa các lớp.

## Bước 5. Thống kê pixel và ảnh trung bình

**Mục tiêu:** Xác định đặc điểm giá trị pixel để chuẩn bị tiền xử lý.

Công việc:

-   Kiểm tra giá trị pixel nhỏ nhất và lớn nhất.
-   Tính Mean và Standard Deviation.
-   Tính ảnh trung bình của từng lớp.
-   So sánh ảnh trung bình giữa những lớp dễ nhầm lẫn.

Công thức chuẩn hóa:

\[ x' = `\frac{x/255 - 0.2860}{0.3530}`{=tex} \]

Trong đó:

-   (x): Giá trị pixel ban đầu.
-   (255): Đưa pixel về khoảng từ 0 đến 1.
-   (0.2860): Mean.
-   (0.3530): Standard Deviation.

**Kết quả mong muốn:** Biết cách chuẩn hóa dữ liệu và nhận ra những lớp
có đặc điểm hình ảnh gần giống nhau.

## Bước 6. Tổng kết khám phá dữ liệu

Sau Notebook 01, cần rút ra các nhận xét:

-   Dữ liệu có 70.000 ảnh và 10 lớp.
-   Ảnh có kích thước `28 × 28` pixel.
-   Dữ liệu ban đầu cân bằng giữa các lớp.
-   Pixel có giá trị từ 0 đến 255.
-   Một số lớp áo có hình dạng gần giống nhau.

**Quyết định:** Sử dụng CNN để trích xuất đặc trưng và phân loại 10 lớp;
áp dụng chuẩn hóa và Data Augmentation để hỗ trợ huấn luyện.

------------------------------------------------------------------------

# GIAI ĐOẠN 2. MODEL TRAINING -- XÂY DỰNG VÀ HUẤN LUYỆN CNN

**Notebook:** `02_model_training.ipynb`

Đây là giai đoạn giúp CNN học đặc trưng từ hình ảnh và điều chỉnh trọng
số để phân loại tốt hơn.

## Bước 1. Chuẩn bị dữ liệu huấn luyện

**Mục tiêu:** Chia dữ liệu và chuẩn bị DataLoader.

Từ 60.000 ảnh Train ban đầu:

-   54.000 ảnh dùng để Train.
-   6.000 ảnh dùng để Validation.
-   10.000 ảnh Test được giữ riêng.

### Data Augmentation

Áp dụng cho tập Train:

-   **Random Horizontal Flip:** Lật ngang ảnh ngẫu nhiên với xác suất
    50%.
-   **Random Rotation:** Xoay ảnh ngẫu nhiên trong khoảng từ -10° đến
    +10°.

Data Augmentation giúp mô hình tiếp xúc với các biến thể hình ảnh khác
nhau, hạn chế việc học thuộc những đặc điểm cố định của ảnh huấn luyện.

**Lưu ý:** Chỉ áp dụng tăng cường dữ liệu cho Train. Validation và Test
sử dụng ảnh không tăng cường.

### Batch Processing

Sử dụng DataLoader với:

``` python
BATCH_SIZE = 64
```

Điều này có nghĩa mô hình xử lý 64 ảnh trong một Batch trước khi chuyển
sang Batch tiếp theo.

**Kết quả mong muốn:** Có ba DataLoader riêng biệt cho Train, Validation
và Test.

## Bước 2. Xây dựng mô hình CNN

**Mục tiêu:** Tạo kiến trúc có khả năng tự học đặc trưng hình ảnh.

Sử dụng class `FashionCNN` trong file `model.py`.

Luồng xử lý:

``` text
Input: 1 × 28 × 28
        |
        v
CNN Block 1
Conv → BN → ReLU → Conv → BN → ReLU
MaxPool → Dropout
        |
        v
32 × 14 × 14
        |
        v
CNN Block 2
Conv → BN → ReLU → Conv → BN → ReLU
MaxPool → Dropout
        |
        v
64 × 7 × 7
        |
        v
Flatten: 3.136 đặc trưng
        |
        v
Fully Connected: 3.136 → 256 → 10
        |
        v
10 giá trị đầu ra
```

**Kết quả mong muốn:** Mô hình nhận ảnh đầu vào và trả về 10 giá trị đầu
ra, tương ứng với 10 lớp.

## Bước 3. Khởi tạo hàm mất mát và thuật toán tối ưu

**Mục tiêu:** Xác định cách đo sai lệch và cập nhật trọng số.

### Hàm mất mát

``` python
criterion = nn.CrossEntropyLoss()
```

CrossEntropyLoss đo mức độ sai lệch giữa kết quả dự đoán và nhãn thật.

### Thuật toán tối ưu

``` python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)
```

AdamW sử dụng gradient để cập nhật trọng số, giúp mô hình cải thiện khả
năng dự đoán.

### Learning Rate Scheduler

``` python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)
```

Khi Validation Loss không cải thiện theo điều kiện của Scheduler,
Learning Rate sẽ giảm một nửa.

**Kết quả mong muốn:** Mô hình có đầy đủ thành phần để bắt đầu huấn
luyện.

## Bước 4. Tiến hành huấn luyện

**Mục tiêu:** Giúp CNN học đặc trưng từ 54.000 ảnh Train.

Một Epoch được thực hiện theo trình tự:

1.  **Lấy Batch ảnh:** Lấy 64 ảnh và nhãn tương ứng từ DataLoader.
2.  **Forward Propagation -- Lan truyền thuận:** Đưa ảnh qua CNN để tạo
    kết quả dự đoán.
3.  **Tính Loss:** So sánh dự đoán với nhãn thật bằng CrossEntropyLoss.
4.  **Backward Propagation -- Lan truyền ngược:** Tính Gradient để xác
    định mức độ ảnh hưởng của trọng số đến sai số.
5.  **Cập nhật trọng số:** AdamW điều chỉnh trọng số nhằm giảm Loss.
6.  **Lặp lại:** Tiếp tục với các Batch còn lại cho đến khi hoàn thành
    tập Train.
7.  **Validation:** Đưa 6.000 ảnh Validation qua mô hình để tính
    Validation Loss và Accuracy, không cập nhật trọng số.

Sau mỗi Epoch, ghi nhận bốn chỉ số:

-   Train Loss.
-   Train Accuracy.
-   Validation Loss.
-   Validation Accuracy.

**Kết quả mong muốn:** Loss giảm và Accuracy tăng khi mô hình học hiệu
quả.

## Bước 5. Theo dõi và kiểm soát quá trình huấn luyện

**Mục tiêu:** Phát hiện mô hình học chưa hiệu quả hoặc có dấu hiệu
overfitting.

### Early Stopping

-   Theo dõi Validation Loss.
-   Nếu Validation Loss không cải thiện sau 5 Epoch liên tiếp thì dừng
    huấn luyện.
-   Tránh tiếp tục huấn luyện khi mô hình không còn cải thiện.

### Model Checkpoint

-   Khi Validation Loss đạt kết quả tốt hơn, lưu trọng số mô hình.
-   File được lưu tại `models/best_model.pth`.
-   Sử dụng lại mô hình có kết quả Validation tốt nhất ở giai đoạn đánh
    giá.

## Bước 6. Trực quan hóa kết quả huấn luyện

**Mục tiêu:** Theo dõi quá trình học thông qua biểu đồ.

Vẽ hai biểu đồ:

1.  **Loss:** So sánh Train Loss và Validation Loss.
2.  **Accuracy:** So sánh Train Accuracy và Validation Accuracy.

  -----------------------------------------------------------------------
  Biểu hiện                           Ý nghĩa
  ----------------------------------- -----------------------------------
  Train Loss giảm                     Mô hình học tốt dữ liệu Train

  Validation Loss giảm                Khả năng dự đoán trên Validation
                                      cải thiện

  Train Accuracy tăng                 Khả năng dự đoán trên Train tốt hơn

  Validation Accuracy tăng            Khả năng tổng quát hóa cải thiện

  Train Loss giảm nhưng Validation    Dấu hiệu overfitting
  Loss tăng kéo dài                   
  -----------------------------------------------------------------------

**Kết quả mong muốn:** Xác định mô hình học tốt hay không và lựa chọn
checkpoint tốt nhất.

------------------------------------------------------------------------

# GIAI ĐOẠN 3. EVALUATION & INFERENCE -- ĐÁNH GIÁ VÀ DỰ ĐOÁN

**Notebook:** `03_evaluation_and_inference.ipynb`

Sau huấn luyện, cần kiểm tra khả năng nhận diện của mô hình trên dữ liệu
chưa được dùng để huấn luyện.

## Bước 1. Nạp mô hình tốt nhất

**Mục tiêu:** Khôi phục trọng số tốt nhất đã lưu ở Notebook 02.

Thực hiện:

-   Đọc file `best_model.pth`.
-   Khởi tạo lại kiến trúc `FashionCNN`.
-   Nạp trọng số đã huấn luyện.
-   Chuyển mô hình sang chế độ `eval()`.

**Kết quả mong muốn:** Có mô hình đã huấn luyện và sẵn sàng đánh giá.

## Bước 2. Đánh giá trên tập Test

**Mục tiêu:** Kiểm tra khả năng dự đoán trên 10.000 ảnh Test.

Đưa ảnh Test qua mô hình, lấy nhãn dự đoán và so sánh với nhãn thật.

### Các chỉ số đánh giá

  -----------------------------------------------------------------------
  Chỉ số                              Ý nghĩa
  ----------------------------------- -----------------------------------
  Accuracy                            Tỷ lệ dự đoán đúng trên tổng số ảnh

  Precision                           Trong các ảnh được dự đoán thuộc
                                      một lớp, có bao nhiêu ảnh đúng

  Recall                              Trong các ảnh thực sự thuộc một
                                      lớp, mô hình nhận diện đúng bao
                                      nhiêu

  F1-score                            Trung bình điều hòa giữa Precision
                                      và Recall
  -----------------------------------------------------------------------

Công thức Accuracy:

\[ Accuracy = `\frac{\text{Số dự đoán đúng}}{\text{Tổng số ảnh}}`{=tex}
\]

Ví dụ: Nếu mô hình dự đoán đúng 9.300 trong tổng số 10.000 ảnh:

\[ Accuracy = `\frac{9300}{10000}`{=tex} = 93% \]

**Kết quả mong muốn:** Xác định tỷ lệ nhận diện đúng trên tập dữ liệu
độc lập.

## Bước 3. Phân tích Confusion Matrix

**Mục tiêu:** Xác định lớp nào được dự đoán đúng và lớp nào thường bị
nhầm lẫn.

Cách đọc Confusion Matrix:

-   **Hàng:** Nhãn thật.
-   **Cột:** Nhãn dự đoán.
-   **Đường chéo chính:** Số lượng dự đoán đúng.
-   **Các ô ngoài đường chéo:** Số lượng dự đoán nhầm.

Ví dụ: Ảnh có nhãn thật `Shirt` nhưng được dự đoán thành `T-shirt/top`
là một trường hợp nhầm lẫn giữa hai lớp.

**Kết quả mong muốn:** Xác định những cặp lớp khó phân biệt và điểm yếu
của mô hình.

## Bước 4. Kiểm tra hình ảnh dự đoán sai

**Mục tiêu:** Quan sát trực tiếp các trường hợp mô hình nhận diện nhầm.

Chương trình lấy tối đa 16 ảnh dự đoán sai. Mỗi ảnh hiển thị:

-   **T (True Label):** Nhãn thật.
-   **P (Predicted Label):** Nhãn mô hình dự đoán.

**Kết quả mong muốn:** Hiểu nguyên nhân mô hình dự đoán sai, nhất là với
các lớp có hình dạng tương đồng.

## Bước 5. Dự đoán một hình ảnh riêng lẻ (Inference)

**Mục tiêu:** Kiểm tra khả năng nhận diện một hình ảnh cụ thể.

Quy trình:

1.  Chọn một ảnh từ tập Test.
2.  Đưa ảnh về khoảng giá trị phù hợp.
3.  Chuẩn hóa ảnh.
4.  Đưa ảnh qua mô hình CNN.
5.  Tính xác suất dự đoán cho 10 lớp bằng Softmax.
6.  Chọn lớp có xác suất cao nhất.
7.  Hiển thị nhãn dự đoán và biểu đồ xác suất.

Ví dụ minh họa giả định:

  Lớp              Xác suất
  -------------- ----------
  Sneaker               95%
  Ankle boot             3%
  Sandal                 1%
  Các lớp khác           1%

Mô hình chọn `Sneaker` vì đây là lớp có xác suất dự đoán cao nhất.

**Kết quả mong muốn:** Mô hình có thể dự đoán một ảnh riêng lẻ và thể
hiện phân phối xác suất trên 10 lớp.

------------------------------------------------------------------------

# PHẦN IV. OUTPUT -- KẾT QUẢ ĐẦU RA CỦA DỰ ÁN

Sau khi hoàn thành ba giai đoạn, dự án có thể tạo ra các sản phẩm sau:

  ----------------------------------------------------------------------------------
  Tệp kết quả                        Nội dung                Mục đích
  ---------------------------------- ----------------------- -----------------------
  `best_model.pth`                   Trọng số CNN tốt nhất   Sử dụng lại mô hình

  `01_class_distribution.png`        Biểu đồ phân bố lớp     Phân tích dữ liệu

  `01_sample_images.png`             Ảnh mẫu                 Quan sát dữ liệu

  `02_training_history.png`          Biểu đồ Loss và         Theo dõi huấn luyện
                                     Accuracy                

  `02_training_history.csv`          Chỉ số qua từng Epoch   Phân tích quá trình học

  `classification_report_test.csv`   Báo cáo phân loại       Đánh giá từng lớp

  `confusion_matrix_test.csv`        Bảng nhầm lẫn           Phân tích lỗi

  `03_confusion_matrix_test.png`     Biểu đồ Confusion       Trực quan hóa kết quả
                                     Matrix                  

  `03_misclassified.png`             Ảnh dự đoán sai         Kiểm tra lỗi

  `03_prediction_example.png`        Ví dụ dự đoán           Minh họa Inference
  ----------------------------------------------------------------------------------

Ngoài ra, chương trình có thể lưu hình ảnh và bảng kết quả trong các thư
mục `outputs/figures/` và `outputs/tables/`.

------------------------------------------------------------------------

# PHẦN V. EVALUATION -- ĐÁNH GIÁ TOÀN BỘ BÀI TOÁN

Cần đánh giá dự án dựa trên các tiêu chí:

  Tiêu chí        Câu hỏi cần trả lời
  --------------- ----------------------------------------------
  Dữ liệu         Dữ liệu đã được xử lý và chia đúng chưa?
  Kiến trúc       CNN có trích xuất được đặc trưng không?
  Huấn luyện      Loss có giảm và Accuracy có tăng không?
  Overfitting     Train và Validation có chênh lệch lớn không?
  Tổng quát hóa   Mô hình dự đoán tốt trên tập Test không?
  Phân loại       Lớp nào nhận diện tốt, lớp nào dễ nhầm?
  Inference       Mô hình có dự đoán được ảnh riêng lẻ không?

### Những điểm cần lưu ý

-   Accuracy cao không có nghĩa là tất cả các lớp đều được nhận diện tốt
    như nhau.
-   Cần xem F1-score để nhận biết lớp có hiệu suất thấp.
-   Cần kiểm tra Confusion Matrix để biết mô hình thường nhầm ở đâu.
-   Không nên kết luận mô hình không overfitting chỉ dựa vào Accuracy;
    cần xem cả Loss.
-   Chỉ sử dụng Test để đánh giá sau huấn luyện, không dùng Test để lựa
    chọn checkpoint.

------------------------------------------------------------------------

# PHẦN VI. CONCLUSION -- KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 1. Kết luận

Dự án triển khai quy trình phân loại hình ảnh thời trang bằng mạng CNN
sử dụng PyTorch. Qua bài tập, có thể thực hiện các công việc:

-   Khám phá và phân tích dữ liệu hình ảnh.
-   Tiền xử lý, chia tập và chuẩn hóa dữ liệu.
-   Thiết kế kiến trúc mạng CNN.
-   Huấn luyện bằng Backpropagation và AdamW.
-   Theo dõi quá trình học thông qua Loss và Accuracy.
-   Đánh giá bằng Accuracy, Precision, Recall, F1-score và Confusion
    Matrix.
-   Dự đoán hình ảnh và phân tích xác suất.

## 2. Hướng phát triển

Có thể cải thiện mô hình bằng cách:

-   Thử nghiệm kiến trúc CNN sâu hơn.
-   Tối ưu phương pháp Data Augmentation.
-   Thử nghiệm các Scheduler khác nhau.
-   Điều chỉnh Batch Size và Learning Rate.
-   Phân tích những lớp có F1-score thấp để cải thiện khả năng phân
    loại.

------------------------------------------------------------------------

# PHẦN VII. TỔNG KẾT TƯ DUY THỰC HIỆN BÀI TOÁN

## 1. Tư duy từ đầu đến cuối

  ------------------------------------------------------------------------
                          Bước Câu hỏi cần trả lời   Cách thực hiện
  ---------------------------- --------------------- ---------------------
         1\. Xác định bài toán Cần giải quyết vấn đề Phân loại 10 loại mặt
                               gì?                   hàng thời trang

          2\. Khám phá dữ liệu Dữ liệu có đặc điểm   70.000 ảnh, kích
                               gì?                   thước `28 × 28`, 10
                                                     lớp

        3\. Thiết kế giải pháp Sử dụng phương pháp   CNN, chuẩn hóa và
                               nào?                  Data Augmentation

          4\. Xây dựng mô hình Cần những thành phần  Conv2D, BatchNorm,
                               nào?                  ReLU, MaxPool,
                                                     Flatten, Linear

                5\. Huấn luyện Làm sao để mô hình    Forward → Loss →
                               học?                  Backward → Optimizer

                  6\. Theo dõi Mô hình học có tốt    Train Loss,
                               không?                Validation Loss,
                                                     Train Accuracy,
                                                     Validation Accuracy

                  7\. Đánh giá Mô hình dự đoán tốt   Test Accuracy,
                               đến đâu?              Precision, Recall,
                                                     F1, Confusion Matrix

                  8\. Ứng dụng Mô hình có nhận diện  Inference và phân
                               ảnh không?            phối xác suất
  ------------------------------------------------------------------------

## 2. Cách ghi nhớ các giai đoạn

  Giai đoạn          Câu hỏi trọng tâm
  ------------------ ---------------------------------------------
  Preview            Bài toán là gì và tại sao cần thực hiện?
  Design             Thiết kế giải pháp như thế nào?
  Data Exploration   Dữ liệu có đặc điểm gì?
  Model Training     Làm sao để mô hình học được?
  Evaluation         Mô hình dự đoán tốt đến đâu?
  Inference          Mô hình có nhận diện được ảnh mới không?
  Conclusion         Đã đạt được gì và có thể cải thiện điều gì?

## 3. Ý tưởng cốt lõi

Không phải chỉ viết CNN rồi huấn luyện là hoàn thành bài toán. Cần hiểu
dữ liệu trước, lựa chọn kiến trúc phù hợp, thiết kế quá trình huấn
luyện, theo dõi khả năng học, đánh giá bằng dữ liệu độc lập và cuối cùng
kiểm tra khả năng dự đoán thực tế.

Đó là quy trình xây dựng một dự án Machine Learning/Deep Learning hoàn
chỉnh từ đầu đến cuối.

------------------------------------------------------------------------

## LƯU Ý QUAN TRỌNG VỀ CHIA DỮ LIỆU

Cần phân biệt hai lần thống kê/chia dữ liệu:

-   **Ban đầu:** Fashion-MNIST có 60.000 ảnh Train và 10.000 ảnh Test.
    Khi thống kê phân bố dữ liệu ban đầu, mỗi lớp có 6.000 ảnh Train và
    1.000 ảnh Test.
-   **Sau khi chia Train:** 60.000 ảnh Train ban đầu được chia tiếp
    thành 54.000 ảnh Train và 6.000 ảnh Validation.

Vì vậy, không nhầm bảng phân bố dữ liệu ban đầu với bảng phân chia
Train/Validation/Test dùng trong quá trình huấn luyện.
