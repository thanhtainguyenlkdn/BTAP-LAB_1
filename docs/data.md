# DATA – HƯỚNG DẪN XỬ LÝ DỮ LIỆU FASHION-MNIST

## 1. Mục đích

Tệp `src/data.py` có nhiệm vụ quản lý toàn bộ quá trình chuẩn bị dữ liệu cho dự án phân loại hình ảnh Fashion-MNIST bằng mô hình CNN. Cụ thể, tệp này chịu trách nhiệm tìm kiếm và tải bộ dữ liệu, đọc hình ảnh và nhãn, chuyển đổi dữ liệu về định dạng phù hợp, chia tập Train, Validation và Test, sau đó tạo các DataLoader để cung cấp dữ liệu cho mô hình trong quá trình huấn luyện và đánh giá.

Fashion-MNIST là bộ dữ liệu hình ảnh thời trang gồm tổng cộng 70.000 ảnh trắng đen (grayscale), mỗi ảnh có kích thước `28 × 28` pixel. Trong đó, 60.000 ảnh được sử dụng làm tập Train ban đầu và 10.000 ảnh được sử dụng làm tập Test. Bộ dữ liệu có 10 lớp đối tượng khác nhau, được đánh số từ 0 đến 9. Mỗi ảnh chứa 784 giá trị pixel, ban đầu nằm trong khoảng từ 0 đến 255.

Việc chuẩn bị dữ liệu đúng cách rất quan trọng vì chất lượng và định dạng dữ liệu đầu vào ảnh hưởng trực tiếp đến khả năng học và độ chính xác của mô hình CNN.

## 2. Chuẩn bị thư mục và môi trường

Trước khi thực hiện các bước xử lý dữ liệu, cần kiểm tra cấu trúc thư mục của dự án để bảo đảm các tệp được đặt đúng vị trí. Cấu trúc thư mục được tổ chức như sau:

```text
project/
├── data/
├── models/
├── outputs/
│   ├── figures/
│   └── tables/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation_and_inference.ipynb
└── src/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── model.py
    ├── trainer.py
    ├── evaluate.py
    └── utils.py
```

Trong cấu trúc trên, thư mục `data/` dùng để lưu dữ liệu Fashion-MNIST, thư mục `models/` lưu trọng số mô hình sau huấn luyện, còn `outputs/` lưu các kết quả như biểu đồ và bảng thống kê. Thư mục `notebooks/` chứa ba notebook phục vụ cho quá trình khám phá dữ liệu, huấn luyện và đánh giá mô hình. Các chương trình xử lý chính được đặt trong thư mục `src/`.

Tiếp theo, cần cài đặt các thư viện Python cần thiết để chạy dự án. Mở Terminal tại thư mục gốc và thực hiện lệnh:

```bash
pip install numpy pandas matplotlib torch torchvision scikit-learn tqdm ipython
```

Các thư viện này đảm nhiệm những chức năng khác nhau. `NumPy` và `Pandas` hỗ trợ xử lý dữ liệu, `Matplotlib` dùng để trực quan hóa hình ảnh và biểu đồ, còn `PyTorch` và `torchvision` phục vụ xây dựng, huấn luyện mô hình CNN và xử lý dữ liệu hình ảnh. Ngoài ra, `scikit-learn` hỗ trợ tính toán các chỉ số đánh giá, `tqdm` hiển thị tiến trình thực hiện và `ipython` hỗ trợ môi trường Notebook.

Nếu chương trình sử dụng KaggleHub để tải dữ liệu từ Kaggle, cần cài đặt thêm:

```bash
pip install kagglehub
```

Lưu ý, việc tải dữ liệu từ Kaggle có thể yêu cầu tài khoản hoặc token Kaggle đã được cấu hình. Trong trường hợp không thể sử dụng KaggleHub, chương trình vẫn có phương án dự phòng thông qua `torchvision`.

## 3. Cách chương trình tìm kiếm và tải dữ liệu

Sau khi chuẩn bị môi trường, bước tiếp theo là kiểm tra nguồn dữ liệu. Trong tệp `src/data.py`, chương trình được thiết kế để tìm dữ liệu theo thứ tự ưu tiên, nhằm tận dụng các tệp đã có sẵn và hạn chế việc tải lại nhiều lần.

**Bước 1: Kiểm tra dữ liệu CSV có sẵn.**

Đầu tiên, chương trình tìm kiếm các tệp dữ liệu trong thư mục `data/`. Đối với tập Train, chương trình kiểm tra tên `fashion-mnist_train.csv` hoặc `train.csv`. Đối với tập Test, chương trình kiểm tra `fashion-mnist_test.csv` hoặc `test.csv`.

Nếu tìm thấy đầy đủ cả hai tệp, chương trình sẽ sử dụng trực tiếp dữ liệu này mà không cần tải lại từ Internet.

**Bước 2: Tải dữ liệu bằng KaggleHub.**

Nếu chưa tìm thấy đầy đủ hai tệp CSV, chương trình sẽ thử tải bộ dữ liệu từ Kaggle thông qua KaggleHub. Dataset được sử dụng là `zalando-research/fashionmnist`.

Sau khi tải thành công, chương trình tìm các tệp CSV phù hợp trong dữ liệu vừa tải và sao chép chúng vào thư mục `data/` để thuận tiện cho những lần sử dụng tiếp theo.

**Bước 3: Sử dụng torchvision khi không tìm được CSV.**

Trong trường hợp không thể tìm thấy dữ liệu CSV hoặc quá trình tải qua KaggleHub không thực hiện được, chương trình chuyển sang phương án dự phòng bằng `torchvision.datasets.FashionMNIST`.

Thư viện `torchvision` sẽ hỗ trợ tải và truy cập trực tiếp bộ Fashion-MNIST. Nhờ đó, quá trình chuẩn bị dữ liệu vẫn có thể tiếp tục mà không phụ thuộc hoàn toàn vào nguồn CSV.

Như vậy, chương trình có ba phương án lấy dữ liệu theo thứ tự: sử dụng CSV có sẵn, tải từ KaggleHub và cuối cùng là tải bằng torchvision.

## 4. Nạp dữ liệu thô để khám phá

Sau khi xác định được nguồn dữ liệu, chúng ta tiến hành nạp dữ liệu thô để kiểm tra và khám phá đặc điểm của bộ Fashion-MNIST.

Mở Notebook `01_data_exploration.ipynb` và thực hiện các lệnh sau:

```python
from src import data

X_train, y_train, X_test, y_test = data.load_raw_arrays()
```

Ở đây, hàm `load_raw_arrays()` có nhiệm vụ đọc dữ liệu và trả về bốn mảng tương ứng với hình ảnh và nhãn của hai tập Train và Test.

Cụ thể, `X_train` chứa hình ảnh của tập Train, còn `y_train` chứa nhãn tương ứng với từng hình ảnh. Tương tự, `X_test` và `y_test` lần lượt chứa hình ảnh và nhãn của tập Test.

Với bộ Fashion-MNIST đầy đủ, kích thước các mảng thường nhận được như sau:

| Biến | Nội dung | Kích thước |
|---|---|---|
| `X_train` | Ảnh Train | `(60000, 28, 28)` |
| `y_train` | Nhãn Train | `(60000,)` |
| `X_test` | Ảnh Test | `(10000, 28, 28)` |
| `y_test` | Nhãn Test | `(10000,)` |

Sau khi nạp dữ liệu, cần kiểm tra kích thước, kiểu dữ liệu và giá trị pixel để bảo đảm dữ liệu được đọc chính xác.

```python
print(X_train.shape, X_train.dtype)
print(y_train.shape, y_train.dtype)
print(X_test.shape, X_test.dtype)
print(y_test.shape, y_test.dtype)
print("Pixel min/max:", X_train.min(), X_train.max())
```

Trong đó, `shape` cho biết kích thước của mảng, `dtype` cho biết kiểu dữ liệu và `min()` và `max()` giúp xác định giá trị pixel nhỏ nhất và lớn nhất.

Thông thường, hình ảnh được trả về dưới dạng `uint8`, với giá trị pixel từ 0 đến 255. Đây là dữ liệu thô, chưa được chuẩn hóa, phù hợp để thống kê và hiển thị bằng Matplotlib.

Nếu kết quả không đúng với kích thước dự kiến, cần kiểm tra lại tệp CSV, nguồn dữ liệu hoặc quá trình đọc dữ liệu trong `data.py` trước khi tiếp tục.

## 5. Xác định ý nghĩa của 10 nhãn

Sau khi nạp dữ liệu, cần xác định ý nghĩa của các nhãn để hiểu mô hình sẽ thực hiện phân loại những đối tượng nào.

Trong tệp `src/config.py`, danh sách tên lớp được khai báo trong biến `CLASS_NAMES` theo thứ tự:

| Nhãn | Tên lớp | Ý nghĩa |
|---|---|---|
| 0 | `T-shirt/top` | Áo phông/áo kiểu |
| 1 | `Trouser` | Quần dài |
| 2 | `Pullover` | Áo len |
| 3 | `Dress` | Váy |
| 4 | `Coat` | Áo khoác |
| 5 | `Sandal` | Dép sandal |
| 6 | `Shirt` | Áo sơ mi |
| 7 | `Sneaker` | Giày thể thao |
| 8 | `Bag` | Túi xách |
| 9 | `Ankle boot` | Bốt cổ thấp |

Mỗi hình ảnh trong bộ dữ liệu được gắn với một nhãn số từ 0 đến 9. Ví dụ, nếu một hình ảnh có nhãn bằng 7 thì đối tượng tương ứng là `Sneaker`, tức giày thể thao.

Việc khai báo `CLASS_NAMES` giúp chương trình chuyển đổi nhãn số thành tên lớp khi hiển thị hình ảnh, thống kê hoặc trình bày kết quả dự đoán.

Cần giữ nguyên thứ tự của 10 lớp vì nếu thay đổi thứ tự, tên lớp hiển thị có thể không còn tương ứng với nhãn thực tế trong dữ liệu.

## 6. Kiểm tra phân bố dữ liệu theo từng lớp

Sau khi xác định các nhãn, bước tiếp theo là kiểm tra số lượng hình ảnh thuộc từng lớp. Việc này giúp đánh giá dữ liệu có cân bằng hay không và xác định liệu có lớp nào xuất hiện quá ít hoặc quá nhiều hay không.

Trong Notebook `01_data_exploration.ipynb`, thực hiện:

```python
import numpy as np
from src import config, utils

counts = np.bincount(y_train, minlength=config.NUM_CLASSES)

for name, count in zip(config.CLASS_NAMES, counts):
    print(f"{name:<12}: {count:>6,} ảnh")

fig = utils.plot_class_distribution(
    y_train,
    save_name="01_class_distribution"
)
```

Đầu tiên, `np.bincount()` được sử dụng để đếm số lần xuất hiện của từng nhãn trong mảng `y_train`. Tham số `minlength=config.NUM_CLASSES` bảo đảm kết quả có đủ 10 vị trí tương ứng với 10 lớp, kể cả khi một nhãn không xuất hiện trong dữ liệu.

Tiếp theo, vòng lặp `for` kết hợp với `zip()` giúp hiển thị tên lớp và số lượng hình ảnh tương ứng. Nhờ đó, chúng ta có thể quan sát số lượng ảnh của từng nhóm một cách trực tiếp.

Cuối cùng, hàm `utils.plot_class_distribution()` tạo biểu đồ phân bố dữ liệu và lưu kết quả vào:

```text
outputs/figures/01_class_distribution.png
```

Đối với bộ Fashion-MNIST chuẩn, tập Train ban đầu có 6.000 ảnh cho mỗi lớp và tập Test có 1.000 ảnh cho mỗi lớp. Nếu kết quả thu được đúng như vậy, có thể nhận xét rằng dữ liệu được phân bố cân bằng về số lượng giữa các lớp.

Điều này giúp Accuracy trở thành một chỉ số tổng quan hữu ích khi đánh giá mô hình. Tuy nhiên, vẫn cần xem thêm Precision, Recall và F1-score của từng lớp để hiểu rõ khả năng phân loại đối với từng loại trang phục.

## 7. Hiển thị hình ảnh mẫu

Sau khi kiểm tra số lượng hình ảnh, chúng ta cần quan sát trực tiếp các mẫu dữ liệu. Đây là bước giúp hiểu rõ đặc điểm hình ảnh, hình dáng của từng loại trang phục và mức độ khác biệt giữa các lớp.

### 7.1. Hiển thị 16 hình ảnh ngẫu nhiên

Đầu tiên, chương trình chọn ngẫu nhiên 16 hình ảnh từ tập Train:

```python
idx = np.random.choice(len(X_train), size=16, replace=False)
images = [X_train[i] for i in idx]
titles = [config.CLASS_NAMES[y_train[i]] for i in idx]

fig = utils.plot_image_grid(
    images, titles, ncols=8,
    save_name="01_sample_images"
)
```

Ở bước này, `np.random.choice()` được sử dụng để chọn 16 vị trí khác nhau trong tập dữ liệu. Tham số `replace=False` bảo đảm không chọn trùng một hình ảnh.

Sau đó, chương trình lấy hình ảnh tương ứng từ `X_train` và sử dụng nhãn trong `y_train` để xác định tên lớp. Hai danh sách `images` và `titles` được truyền vào hàm `plot_image_grid()` để hiển thị hình ảnh theo dạng lưới.

Tham số `ncols=8` quy định mỗi hàng có 8 hình ảnh. Kết quả giúp quan sát nhanh nhiều mẫu trang phục khác nhau và kiểm tra xem hình ảnh có được đọc đúng hay không.

### 7.2. Hiển thị hình ảnh đại diện cho từng lớp

Ngoài việc quan sát các hình ảnh ngẫu nhiên, chúng ta cần hiển thị một hình ảnh đại diện cho mỗi lớp để so sánh đặc điểm của 10 nhóm đối tượng.

Thực hiện lệnh:

```python
fig = utils.plot_samples_per_class(
    X_train, y_train,
    save_name="01_samples_per_class"
)
```

Hàm `plot_samples_per_class()` sử dụng hình ảnh và nhãn để tìm một mẫu thuộc từng lớp, sau đó sắp xếp thành lưới gồm 10 hình ảnh.

Kết quả được lưu tại:

```text
outputs/figures/01_samples_per_class.png
```

Thông qua hình ảnh này, chúng ta có thể nhận biết sự khác nhau giữa các nhóm như quần, váy, áo khoác, túi xách và giày. Đây cũng là cơ sở để nhận xét về những lớp có hình dáng tương đối giống nhau và có khả năng bị mô hình phân loại nhầm.

## 8. Thống kê giá trị pixel và hình ảnh trung bình

Sau khi quan sát hình ảnh mẫu, bước tiếp theo là tìm hiểu đặc điểm phân bố giá trị pixel trong toàn bộ tập Train.

Thực hiện các lệnh:

```python
print("Pixel min/max:", X_train.min(), X_train.max())
print("Mean thô:", X_train.mean())
print("Std thô:", X_train.std())
```

Trong đó, `min()` và `max()` xác định giá trị pixel nhỏ nhất và lớn nhất. `mean()` tính giá trị pixel trung bình, còn `std()` tính độ lệch chuẩn của các giá trị pixel.

Các thông số này giúp hiểu được phạm vi và mức độ phân tán của dữ liệu hình ảnh trước khi đưa vào mô hình CNN.

Tiếp theo, chúng ta tính hình ảnh trung bình của từng lớp:

```python
mean_images = [
    X_train[y_train == i].mean(axis=0)
    for i in range(config.NUM_CLASSES)
]

fig = utils.plot_image_grid(
    mean_images, config.CLASS_NAMES,
    ncols=5, save_name="01_mean_images"
)
```

Đối với mỗi nhãn từ 0 đến 9, biểu thức `y_train == i` được sử dụng để chọn tất cả hình ảnh thuộc lớp tương ứng. Sau đó, `mean(axis=0)` tính giá trị trung bình tại từng vị trí pixel của các hình ảnh đó.

Kết quả thu được là một hình ảnh trung bình đại diện cho mỗi lớp. Các hình ảnh này giúp quan sát hình dáng chung của từng nhóm trang phục, nhưng không phải là hình ảnh của một sản phẩm thực tế.

Ngoài ra, Fashion-MNIST có một số lớp trang phục tương đối giống nhau. Vì vậy, chương trình tiếp tục so sánh bốn lớp gồm T-shirt/top, Pullover, Coat và Shirt:

```python
confusing = [0, 2, 4, 6]

mean_imgs = [
    X_train[y_train == i].mean(axis=0)
    for i in confusing
]

titles = [config.CLASS_NAMES[i] for i in confusing]

utils.plot_image_grid(
    mean_imgs, titles, ncols=4,
    save_name="01_confusing_classes"
)
```

Bốn lớp này có thể có hình dáng tổng thể khá giống nhau. Sự khác biệt thường thể hiện ở phần cổ áo, tay áo, độ dài và cấu trúc của trang phục.

Việc so sánh giúp nhận biết những trường hợp khó phân biệt, từ đó hiểu rõ hơn những thách thức mà mô hình CNN phải giải quyết trong quá trình phân loại.

## 9. Chuẩn hóa dữ liệu hình ảnh

Sau khi hoàn thành việc khám phá dữ liệu thô, chúng ta cần chuẩn hóa hình ảnh trước khi đưa vào mô hình CNN.

Trong tệp `src/config.py`, hai thông số chuẩn hóa được khai báo:

```python
MEAN = 0.2860
STD = 0.3530
```

Đầu tiên, giá trị pixel ban đầu nằm trong khoảng 0–255 được chuyển về khoảng 0–1 bằng cách chia cho 255. Sau đó, chương trình thực hiện chuẩn hóa theo công thức:

\[
x_{norm} = \frac{x/255 - 0.2860}{0.3530}
\]

Trong công thức trên, `x` là giá trị pixel ban đầu, còn `x_norm` là giá trị sau chuẩn hóa. Hai thông số `MEAN` và `STD` lần lượt biểu thị giá trị trung bình và độ lệch chuẩn được sử dụng trong quá trình biến đổi dữ liệu.

Việc chuẩn hóa giúp đưa dữ liệu đầu vào về thang đo phù hợp hơn, hỗ trợ quá trình tối ưu trọng số của mô hình ổn định và hiệu quả hơn.

Cần lưu ý rằng hàm `load_raw_arrays()` chỉ trả về dữ liệu thô để phục vụ khám phá và thống kê. Việc chuyển đổi và chuẩn hóa dữ liệu dành cho huấn luyện được thực hiện trong `FashionMNISTCSVDataset` hoặc thông qua transform của torchvision.

Do đó, không nên chuẩn hóa cùng một dữ liệu hai lần vì sẽ làm sai lệch giá trị đầu vào của mô hình.

## 10. Chia dữ liệu thành Train, Validation và Test

Sau khi chuẩn hóa, dữ liệu cần được chia thành các tập riêng biệt để phục vụ những mục đích khác nhau trong quá trình xây dựng mô hình.

Trong `src/data.py`, hàm `get_dataloaders()` thực hiện việc chia 60.000 ảnh Train ban đầu thành hai tập Train và Validation. Tỷ lệ chia được xác định bởi biến `VAL_SPLIT = 0.1`, nghĩa là sử dụng 10% dữ liệu ban đầu cho Validation.

Với cấu hình hiện tại, số lượng dữ liệu được phân chia như sau:

| Tập dữ liệu | Số lượng ảnh | Mục đích |
|---|---:|---|
| Train | 54.000 | Cập nhật trọng số mô hình |
| Validation | 6.000 | Theo dõi chất lượng và lựa chọn mô hình tốt nhất |
| Test | 10.000 | Đánh giá mô hình sau huấn luyện |

Tập Train được sử dụng để mô hình học các đặc trưng hình ảnh và điều chỉnh trọng số thông qua quá trình lan truyền ngược. Tập Validation không trực tiếp cập nhật trọng số mà được dùng để theo dõi khả năng hoạt động của mô hình sau mỗi epoch, hỗ trợ điều chỉnh quá trình huấn luyện và lựa chọn checkpoint tốt nhất.

Trong khi đó, tập Test được giữ riêng để đánh giá kết quả cuối cùng. Việc không sử dụng tập Test để điều chỉnh mô hình giúp kết quả đánh giá phản ánh khách quan hơn khả năng dự đoán trên dữ liệu chưa được dùng trong quá trình huấn luyện.

Quá trình chia dữ liệu sử dụng `np.random.RandomState(seed).permutation(...)`, với giá trị seed mặc định bằng 42. Việc cố định seed giúp quá trình chia dữ liệu có thể tái lập, tức là khi chạy lại với cùng cấu hình, chương trình sẽ tạo ra cách chia tương ứng như trước.

Lưu ý, phương pháp này là chia ngẫu nhiên theo chỉ số, không phải Stratified Split.

## 11. Data Augmentation và tạo DataLoader

Sau khi chia dữ liệu, bước tiếp theo là tạo DataLoader để cung cấp từng nhóm hình ảnh cho mô hình trong quá trình huấn luyện và đánh giá.

Trong Notebook `02_model_training.ipynb`, có thể tạo các DataLoader bằng lệnh:

```python
train_loader, val_loader, test_loader = data.get_dataloaders(
    batch_size=config.BATCH_SIZE,
    val_split=config.VAL_SPLIT,
    use_augmentation=True
)
```

Hàm `get_dataloaders()` nhận các tham số cấu hình và trả về ba DataLoader tương ứng với tập Train, Validation và Test.

Với cấu hình hiện tại, Batch Size của tập Train là 64. Điều này có nghĩa là mỗi lần mô hình lấy một nhóm gồm 64 hình ảnh để thực hiện tính toán và cập nhật trọng số. Đối với Validation và Test, Batch Size được sử dụng gấp đôi, tương ứng là 128.

Ngoài ra, chương trình thiết lập `shuffle=True` cho Train để xáo trộn thứ tự dữ liệu sau mỗi epoch. Điều này giúp mô hình không học theo một thứ tự hình ảnh cố định. Đối với Validation và Test, `shuffle=False` được sử dụng để giữ nguyên thứ tự dữ liệu khi đánh giá.

### 11.1. Tăng cường dữ liệu (Data Augmentation)

Trong quá trình huấn luyện, chương trình có sử dụng Data Augmentation để tạo ra những biến thể hình ảnh từ dữ liệu ban đầu. Mục đích là giúp mô hình tiếp xúc với nhiều dạng hình ảnh khác nhau, từ đó hạn chế việc học thuộc các mẫu Train và cải thiện khả năng tổng quát hóa.

Hai phép biến đổi được sử dụng cho tập Train gồm:

**Lật ngang ngẫu nhiên:** Hình ảnh được lật theo chiều ngang với xác suất 50%. Nhờ đó, mô hình có thể học cách nhận diện đối tượng ngay cả khi hình ảnh bị đảo hướng trái – phải.

**Xoay ngẫu nhiên:** Hình ảnh được xoay một góc ngẫu nhiên trong khoảng từ -10° đến +10°. Phép biến đổi này giúp mô hình làm quen với những hình ảnh có hướng nghiêng nhẹ.

Các phép tăng cường chỉ được áp dụng cho tập Train. Tập Validation và Test không sử dụng Data Augmentation để bảo đảm quá trình theo dõi và đánh giá được thực hiện trên dữ liệu không bị biến đổi ngẫu nhiên.

Cần phân biệt rằng Data Augmentation không làm tăng số lượng bản ghi gốc trong bộ dữ liệu. Thay vào đó, hình ảnh được biến đổi khi được lấy ra trong quá trình huấn luyện.

### 11.2. Vai trò của DataLoader

DataLoader giúp chia dữ liệu thành từng batch, quản lý thứ tự lấy mẫu và cung cấp dữ liệu cho mô hình theo từng lần xử lý.

Thay vì đưa toàn bộ 54.000 hình ảnh vào mô hình cùng một lúc, chương trình chỉ lấy từng nhóm 64 ảnh để huấn luyện. Cách thực hiện này giúp quản lý bộ nhớ hiệu quả hơn và phù hợp với cơ chế tối ưu theo mini-batch của PyTorch.

Sau khi tạo thành công ba DataLoader, dữ liệu đã sẵn sàng để chuyển sang giai đoạn xây dựng và huấn luyện mô hình CNN.

## 12. Kết quả đầu ra của giai đoạn xử lý dữ liệu

Sau khi hoàn thành Notebook `01_data_exploration.ipynb`, cần kiểm tra lại các kết quả thu được để bảo đảm dữ liệu đã được đọc và phân tích đúng cách.

Các kết quả chính bao gồm thông tin kích thước và kiểu dữ liệu của tập Train, Test; số lượng hình ảnh thuộc từng lớp; biểu đồ phân bố dữ liệu; hình ảnh mẫu ngẫu nhiên; hình ảnh đại diện cho từng lớp; các thống kê về giá trị pixel và hình ảnh trung bình.

Ngoài ra, kết quả so sánh những lớp dễ nhầm lẫn cũng giúp nhận biết đặc điểm của dữ liệu và giải thích những khó khăn mà mô hình có thể gặp phải khi phân loại.

Các biểu đồ và hình ảnh trực quan được lưu trong thư mục:

```text
outputs/figures/
```

Sau khi hoàn thành giai đoạn này, chúng ta đã nắm được cấu trúc, số lượng, nhãn và đặc điểm cơ bản của bộ dữ liệu Fashion-MNIST. Những dữ liệu đã xử lý và các DataLoader sẽ được tạo lại khi Notebook `02_model_training.ipynb` thực hiện quá trình huấn luyện.

Đây là bước chuẩn bị quan trọng trước khi chuyển sang xây dựng kiến trúc CNN, lựa chọn các tham số huấn luyện và đánh giá khả năng phân loại hình ảnh của mô hình.
