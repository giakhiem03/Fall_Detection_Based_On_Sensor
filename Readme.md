*1. Các thư viện và module sử dụng:*

+ numpy và tensorflow: Dùng để xử lý các phép toán số học và thực hiện các tác vụ học máy.

+ random và os: Được dùng để thiết lập hạt giống ngẫu nhiên và quản lý môi trường hệ thống.

+ KFold từ sklearn.model_selection: Dùng để chia bộ dữ liệu thành k phần để thực hiện cross-validation.

+ confusion_matrix từ sklearn.metrics: Dùng để tính toán ma trận nhầm lẫn (confusion matrix) nhằm đánh giá mô hình.

+ SMOTE từ imblearn.over_sampling: Giúp xử lý mất cân bằng lớp trong dữ liệu bằng cách tạo ra các mẫu tổng hợp cho lớp thiểu số.

+ matplotlib.pyplot và seaborn: Được sử dụng để vẽ các biểu đồ, đặc biệt là ma trận nhầm lẫn.

+ EarlyStopping từ tensorflow.keras.callbacks: Dùng để dừng quá trình huấn luyện nếu độ lỗi không cải thiện, giúp tránh overfitting.

*2. Chi tiết các bước trong mã:*

Hàm set_seed(): Đảm bảo tính tái lập kết quả bằng cách thiết lập hạt giống ngẫu nhiên cho các thư viện liên quan như numpy, tensorflow, và random.

Hàm k_fold_cross_validation(X, y, k=5)

Thực hiện cross-validation với k-fold (mặc định là 5 folds).

Hàm load_fall_times(fall_times_file) đọc dữ liệu từ những cú ngã thực của người thử nghiệm (3.1 giây)

Hàm compute_fs1_features(window_data) tính toán các đặc trưng FS-1 cho một cửa sổ dữ liệu. Các đặc trưng FS-1 này bao gồm:

Max: Giá trị lớn nhất trong dữ liệu.

Min: Giá trị nhỏ nhất trong dữ liệu.

Mean: Trung bình của dữ liệu, được tính theo công thức động (cập nhật dần sau mỗi giá trị).

Variance: Phương sai của dữ liệu, cũng tính theo công thức động.

Hàm sliding_window_with_labels(data, fall_times_df, folder_name, file_name, window_size=450, window_jump=50) sử dụng kỹ thuật cửa sổ trượt (sliding window) để chia dữ liệu thành các cửa sổ có kích thước cố định (9 giây) và tính toán các đặc trưng FS-1 cho mỗi cửa sổ. Đồng thời, gán nhãn cho mỗi cửa sổ là Fall (1) hoặc ADL (0).

Chia dữ liệu thành các phần và huấn luyện mô hình MLP trên từng fold, đồng thời áp dụng SMOTE để cân bằng dữ liệu.

Sau mỗi fold, chương trình tính toán các chỉ số đánh giá mô hình như độ chính xác (accuracy), độ nhạy (sensitivity), và độ đặc hiệu (specificity).

Ma trận nhầm lẫn được hiển thị dưới dạng heatmap cho mỗi fold, và cuối cùng, tính toán và vẽ ma trận nhầm lẫn trung bình từ tất cả các fold.

*3. Quy trình huấn luyện và đánh giá mô hình:*

Chuẩn bị dữ liệu: Dữ liệu đầu vào được chia thành tập huấn luyện và tập kiểm tra dựa trên chỉ số fold.

Cân bằng dữ liệu: Dùng SMOTE để tạo ra các mẫu tổng hợp cho lớp thiểu số trong dữ liệu huấn luyện.

Xây dựng mô hình: Mô hình MLP gồm nhiều lớp Dense, BatchNormalization, và Dropout để học dữ liệu.
Huấn luyện mô hình: Mô hình được huấn luyện trên tập huấn luyện đã cân bằng dữ liệu, với EarlyStopping để dừng huấn luyện nếu mô hình không cải thiện sau một số epoch.

Đánh giá mô hình: Sau khi huấn luyện xong, mô hình được kiểm tra trên tập kiểm tra và tính toán các chỉ số như độ chính xác, độ nhạy, độ đặc hiệu. Đồng thời, vẽ ma trận nhầm lẫn cho từng fold và tính toán các giá trị trung bình.

*4. Hiển thị kết quả:*

Biểu đồ ma trận nhầm lẫn được vẽ cho từng fold, cùng với các chỉ số đánh giá cho từng fold.
Sau khi hoàn thành cross-validation, chương trình tính toán các chỉ số trung bình (accuracy, sensitivity, specificity) và hiển thị chúng cùng với độ lệch chuẩn.

Cuối cùng, ma trận nhầm lẫn trung bình từ tất cả các fold được vẽ để đánh giá tổng thể hiệu suất mô hình.
*5. Cách import các thư viện: *

+ pip install tensorflow
+ pip install numpy
+ pip install scikit-learn
+ pip install imbalanced-learn
+ pip install matplotlib
+ pip install seaborn
+ pip install catboost
