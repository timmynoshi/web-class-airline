from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType

# Khởi tạo SparkSession
spark = (SparkSession.builder
         .appName("use_trained_model")
         .config("spark.driver.memory", "4g")  # Tăng bộ nhớ driver
         .config("spark.executor.memory", "4g")  # Tăng bộ nhớ executor
         .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
         .getOrCreate())


# Tải mô hình đã huấn luyện từ HDFS
model_path = "hdfs://localhost:9000/thanhtin/rf_model"
rf_model = RandomForestClassificationModel.load(model_path)

# Khởi tạo Flask và kích hoạt CORS
app = Flask(__name__)
CORS(app)

# Định nghĩa ánh xạ cho các giá trị chuỗi sang số
gender_mapping = {"Male": 0.0, "Female": 1.0}
customer_type_mapping = {"Loyal Customer": 0.0, "disloyal Customer": 1.0}
type_of_travel_mapping = {"Business travel": 0.0, "Personal Travel": 1.0}
class_mapping = {"Eco": 0.0, "Eco Plus": 1.0, "Business": 2.0}

# Định nghĩa schema cho dữ liệu đầu vào
schema = StructType([
    StructField("Age", IntegerType(), True),
    StructField("Flight Distance", FloatType(), True),
    StructField("Inflight wifi service", IntegerType(), True),
    StructField("Departure/Arrival time convenient", IntegerType(), True),
    StructField("Ease of Online booking", IntegerType(), True),
    StructField("Gate location", IntegerType(), True),
    StructField("Food and drink", IntegerType(), True),
    StructField("Online boarding", IntegerType(), True),
    StructField("Seat comfort", IntegerType(), True),
    StructField("Inflight entertainment", IntegerType(), True),
    StructField("On-board service", IntegerType(), True),
    StructField("Leg room service", IntegerType(), True),
    StructField("Baggage handling", IntegerType(), True),
    StructField("Checkin service", IntegerType(), True),
    StructField("Inflight service", IntegerType(), True),
    StructField("Cleanliness", IntegerType(), True),
    StructField("Departure Delay in Minutes", FloatType(), True),
    StructField("Arrival Delay in Minutes", FloatType(), True),
    StructField("Gender", FloatType(), True),
    StructField("Customer Type", FloatType(), True),
    StructField("Type of Travel", FloatType(), True),
    StructField("Class", FloatType(), True)
])

# Định nghĩa route cho trang chủ để nhập thông tin
@app.route('/')
def home():
    return render_template('form.html')

# Định nghĩa route cho API dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu JSON từ yêu cầu
        data = request.get_json()
        print("Received data:", data)

        # Xử lý các giá trị chuỗi và tạo dữ liệu đầu vào cho mô hình
        input_data = {
            "Age": int(data['Age']),
            "Flight Distance": float(data['Flight Distance']),
            "Inflight wifi service": int(data['Inflight wifi service']),
            "Departure/Arrival time convenient": int(data['Departure/Arrival time convenient']),
            "Ease of Online booking": int(data['Ease of Online booking']),
            "Gate location": int(data['Gate location']),
            "Food and drink": int(data['Food and drink']),
            "Online boarding": int(data['Online boarding']),
            "Seat comfort": int(data['Seat comfort']),
            "Inflight entertainment": int(data['Inflight entertainment']),
            "On-board service": int(data['On-board service']),
            "Leg room service": int(data['Leg room service']),
            "Baggage handling": int(data['Baggage handling']),
            "Checkin service": int(data['Checkin service']),
            "Inflight service": int(data['Inflight service']),
            "Cleanliness": int(data['Cleanliness']),
            "Departure Delay in Minutes": float(data['Departure Delay in Minutes']),
            "Arrival Delay in Minutes": float(data['Arrival Delay in Minutes']),
            "Gender": gender_mapping.get(data['Gender'], 0.0),
            "Customer Type": customer_type_mapping.get(data['Customer Type'], 0.0),
            "Type of Travel": type_of_travel_mapping.get(data['Type of Travel'], 0.0),
            "Class": class_mapping.get(data['Class'], 0.0)
        }

        # Chuyển dữ liệu thành DataFrame của Spark
        input_df = spark.createDataFrame([input_data], schema=schema)

        # Tạo cột `features`
        feature_columns = list(input_data.keys())
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        input_df = assembler.transform(input_df)

        # Thực hiện dự đoán
        predictions = rf_model.transform(input_df)
        prediction = predictions.select("prediction").collect()[0][0]

        # Trả về kết quả dưới dạng JSON
        result = "Customer is Satisfied" if prediction == 1 else "Customer is Neutral or Dissatisfied"
        return jsonify({"result": result})
    except Exception as e:
        print("Error occurred:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
