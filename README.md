Chào mọi người! Mình chia sẻ ngắn gọn trong bài này giải pháp của mình trong cuộc thi phân loại sắc thái bình luận tại AIVIVN.COM.
https://forum.machinelearningcoban.com/t/chia-se-model-sentiment-analysis-aivivn-com-top-5/4537
<h1>Sơ lược giải pháp:</h1>
Giải pháp của mình tập trung vào data hơn mô hình. Với bài toán này, mình tập trung tiền xử lý dữ liệu, loại bỏ nhiễu, gán nhãn lại các mislabel data. Lý do tập trung vào data hơn vì mình quan sát dữ liệu thấy có khá nhiều nhiễu, gán nhãn sai và lấy từ các trang thương mại điện tử nên từ ngữ lộn xộn, thường không theo văn phong chuẩn mực, cần phải có bước chuẩn hóa. Mô hình mình sử dụng là SVM và feature quen thuộc TF-IDF (5-gram). Lý do sử dụng SVM vì mình thấy SVM khá phù hợp với các bài toán có ít dữ liệu nhưng nhiều features. Mô hình này vẫn cho kết quả khá tốt, thời gian train/predict khá nhanh (train dưới 1 phút với macbook 2015 của mình). Cuối cùng là giải thích về việc dùng Error Analysis để gán lại các Mislabel data.
<h1>Chi tiết cách thực hiện:</h1>
<h2>1. Tiền xử lý dữ liệu:</h2>
<ul>
<li>Dữ liệu bình luận (văn nói) nên người dùng thường không quan tâm đến chữ hoa thường khi gõ, đưa hết về lower case.</li>
<li>Loại bỏ những ký tự kéo dài: Ví dụ: Áo đẹp quáaaaaaaa--> Áo đẹp quá.</li>

 <li>Tiếng Việt có 2 cách bỏ dấu nên đưa về 1 chuẩn. Ví dụ, chữ "Hòa" và "Hoà" đều được chấp nhận trong tiếng Việt. Ngoài ra còn một số trường hợp lỗi font chữ cũng cần chuẩn hóa lại. (các trường hợp dính chữ như: "Giao hàngnhanh" xử ý đc sẽ tốt hơn).</li>
<li>Chuẩn hóa một số sentiment word: "okie"-->"ok", "okey"-->"ok", authentic--> "chuẩn chính hãng",vv...</li>
<li>Emoj quy về 2 loại: emojs mang ý nghĩa tích cực (positive): '💯','💗' và emojs mang nghĩa tiêu cực (nagative): '👎','😣'.</li>
<li>Người dùng đánh giá 1,2 sao (*) quy hết về 1star, trên 3 sao quy hết về 5tar.</li>
<li>Loại bỏ dấu câu (puntuations) và các ký tự nhiễu.</li>
<li>Xử lý vấn đề phủ định, TF-IDF không xử lý được vấn đề phủ định trong bài toán sentiment. Ví dụ: <em>Cái áo này rất đẹp</em> và <em>Cái áo này chẳng đẹp</em> sẽ không khác nhau nhiều khi chọn feature tf-idf, giải pháp của mình là biến <em>chẳng đẹp</em> thành <em>not-positive</em>, hay <em>không tệ</em> thành <em>not-nagative</em> bằng cách dùng từ điển tâm lý và từ điển phủ định. Từ điển tâm lý mình lấy từ VietSentwordnet 1.0 chỉ lấy những từ có score >0.5 bổ sung 1 số sentiment words đặc thù từ tập train.</li>
<li>Augmentation data bằng cách thêm vào các sample của chính tập train nhưng không dấu. (Bình luận không dấu khá phổ biến).</li>
<li>Ngoài ra, mình bổ sung vào tập train các sample mới lấy từ chính 2 từ điển positive và nagative. Các từ vựng trong từ điển tích cực gán nhãn 0, các từ vựng từ từ điển tiêu cực gán nhãn 1.</li>
</ul>
<h2>2. Lựa chọn mô hình/tunning:</h2>
Mình chọn SVM, tunning một chút parameter.
feature TF-IDF, 5 gram, sử dụng thư viện SKlearn. Stopword xem ra không hữu dụng lắm.
<h2>3. Sử dụng Error Analysis để gán lại nhãn:</h2>
Bằng cách train 2 lần: Lần 1 chia tập train/test theo tỉ lệ 7/3 và lần 2 train overfitting, mình phát hiện ra các trường hợp gán nhãn sai và gán lại nhãn, lặp đi lặp lại quá trình này vài chục lần mình đã gán nhãn lại được khá nhiều data. Cách làm của mình dựa trên ý tưởng của Overfitting, nếu đã dạy mô hình tập dữ liệu A rồi test trên chính tập A đó mà mô hình chỉ đạt độ chính xác thấp chứng tỏ dữ liệu chưa phổ quát, quá ít dữ liệu hoặc gán nhãn sai. VD: Train 7/3 đạt 89%, train overfit đạt chỉ 94% thì chứng tỏ có nhiều data gán nhãn sai. Mình gán lại nhãn  đến khi độ chính xác khi train overfit đạt khoảng 98% thì dừng lại, lúc này độ chính xác của train 7/3 đạt khoảng 94%. Việc gán lại nhãn, loại bỏ nhiễu với train data là một phần của data science và hoàn toàn hợp lệ. (Tất nhiên là không động chút nào đến test data).
<h2>Kết quả</h2>
Mình chạy Crossvalidation 5 folds để đánh giá model công bằng hơn. Kết quả cuối cùng CV5fold với <strong>tập dữ liệu đã gán lại nhãn</strong> là <strong>94.4%</strong>. Kết quả submit tại AIVIVN.COM của mình đạt <strong>89.57%</strong>. 
<h2>CODE</h2>
Mô hình của mình đặt tại <a href="https://github.com/swordmanager/sentiment_analysis_nal">Github</a>
