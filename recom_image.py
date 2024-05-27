from flask import Flask, render_template, render_template_string, request, jsonify, send_file
from flask_cors import CORS
import pymongo
from bson import ObjectId  # Import ObjectId from bson module
import cv2
import numpy as np
import tempfile
import os
import signal
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["PlateformIOTdb"]
collection = db["sensors"]

# Create a temporary directory to store images
temp_dir = tempfile.TemporaryDirectory()

def get_image_difference(img1, img2):
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)

    if image1 is None or image2 is None:
        return "One or both of the images could not be read."

    if image1.shape != image2.shape:
        return "Images have different dimensions. They are not the same."

    difference = cv2.absdiff(image1, image2)
    gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = sum(cv2.contourArea(cnt) for cnt in contours)
    area_threshold = 0.1 * image1.shape[0] * image1.shape[1]

    if total_area > area_threshold:
        return "There is a big change between the images."
    elif total_area > 0:
        return "There is a slight change between the images."
    else:
        return "The images are the same."

def get_image_object(object_id):
    # Retrieve document containing binary image data using ObjectId
    document = collection.find_one({"_id": object_id})

    if document:
        # Extract values field
        values = document.get("values")
        image_paths = []

        if values:
            # Iterate over each value
            for i, value in enumerate(values):
                # Check if the value is a dictionary
                if isinstance(value, dict) and "value" in value:
                    # Extract binary data from the value
                    binary_data = value["value"]
                    
                    # Check if the binary data is bytes
                    if isinstance(binary_data, bytes):
                        # Create a temporary file path
                        temp_file_path = os.path.join(temp_dir.name, f"image_{i}.jpg")
                        
                        # Write binary data to the temporary file
                        with open(temp_file_path, 'wb') as f:
                            f.write(binary_data)
                        
                        # Append the temporary file path to the list
                        image_paths.append(temp_file_path)
                    else:
                        print("Binary data is not bytes.")
                else:
                    print(f"No 'value' field found in document {i+1}.")
        else:
            print("No 'values' field found in the document.")
    else:
        print(f"No document found with ID {object_id} in the collection.")
    
    return image_paths

@app.route('/image_difference/<string:document_id>', methods=['GET'])
def compare_images(document_id):
    # Convert the ObjectId string to ObjectId object
    object_id = ObjectId(document_id)
    images = get_image_object(object_id)

    if len(images) < 2:
        return jsonify({"error": "Not enough images for comparison."}), 400

    # Move the first image to the front
    images = [images[0]] + images[1:]

    comparison_results = []
    first_image = images[0]

    # Modify this loop to include identifiers like "Image 1", "Image 2", etc.
    for i, img in enumerate(images[1:], start=1):
        result = get_image_difference(first_image, img)
        comparison_results.append((f"Image {i}", img, result))

    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Comparison Results</title>
    <style>
        body {
            background-color: #000418;
            color: white;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            text-align: center;
        }
        ul {
            list-style-type: none;
            padding: 0;
        }
        li {
            margin-bottom: 10px;
        }
        .image-container {
            display: inline-block;
            margin-right: 20px;
            text-align: center;
        }
        .image-container img {
            width: 200px; /* Set the width of the images */
            height: auto; /* Maintain aspect ratio */
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Image Comparison Results</h1>

        <ul>
            {% for img_num, img_path, result in results %}
                <li>{{ img_num }}: {{ result }}</li>
                <div class="image-container">
                    <img src="/images/{{ images[0] }}" alt="Original Image">
                    <p>Original Image</p>
                </div>
                <div class="image-container">
                    <img src="/images/{{ img_path }}" alt="Image {{ img_num }}">
                    <p>{{ img_num }}</p>
                </div>
            {% endfor %}
        </ul>
    </div>
</body>
</html>
    """

    return render_template_string(html_template, images=images, results=comparison_results)

@app.route('/images/<path:image_name>')
def get_image(image_name):
    # Construct the full path to the image
    image_path = os.path.join(temp_dir.name, image_name)
    # Check if the image exists
    if os.path.exists(image_path):
        # Stream the image file
        return send_file(image_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404

# HTML template as a string for recommendations
template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sensor Recommendations</title>
    <style>
        body {
            background-color: #000418; /* Blue background color */
            color: white;
            font-family: Arial, sans-serif;
        }
        .container {
            margin: 20px;
            padding: 20px;
            background-color: darkBlue;
            border-radius: 10px;
        }
        .sensor {
            margin-bottom: 20px;
        }
        .sensor h2 {
            margin-top: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        {% for sensor_name, recs in recommendations.items() %}
        <div class="sensor">
            <h2>Sensor: {{ sensor_name }}</h2>
            <p>{{ recs.recommendation_1 }}</p>
            <ul>
                {% for rec in recs.recommendation_2 %}
                <li>{{ rec }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endfor %}
    </div>
</body>
</html>
'''

# Route to get recommendations based on space ID
@app.route('/get-recommendations/<string:space_id>')
def get_recommendations(space_id):
    # Convert space_id string to ObjectId
    try:
        space_id_obj = ObjectId(space_id)
    except:
        return jsonify({"error": "Invalid space ID format"})

    # Retrieve sensor list for the given space ID
    sensor_list = get_sensor_list(space_id_obj)

    if not sensor_list:
        return jsonify({"error": "No sensors found for the given space ID."})

    # Perform recommendation logic for each sensor
    recommendations = {}
    for sensor_id in sensor_list:
        sensor_name = get_sensor_name(ObjectId(sensor_id))
        sensor_type = get_sensor_type(ObjectId(sensor_id))
        sensor_data = get_sensor_data(ObjectId(sensor_id))
        
        # Generate recommendations only if recommendation_1 is not None
        if sensor_type == "chart":
            freq_recommendation = generate_frequency_recommendation(sensor_data)
            if freq_recommendation is not None:
                outlier_recommendation = detect_outliers(sensor_data)
                anomaly_recommendation = detect_anomalies(sensor_data)
                recommendations[sensor_name] = {
                    "recommendation_1": freq_recommendation,
                    "recommendation_2": generate_recommendations(outlier_recommendation, anomaly_recommendation)
                }

    return render_template_string(template, recommendations=recommendations)

def get_sensor_list(space_id):
    # Find the space in MongoDB
    space = db["spaces"].find_one({"_id": space_id})
    if not space:
        return None

    # Extract sensor IDs from the space document
    sensor_list = space.get("sensorList", [])

    if not sensor_list:
        return None

    return sensor_list

def get_sensor_name(sensor_id):
    # Find the sensor in MongoDB
    sensor = db["sensors"].find_one({"_id": sensor_id})
    if not sensor:
        return None

    # Retrieve sensor name
    name = sensor.get("name", "")

    return name

def get_sensor_type(sensor_id):
    # Find the sensor in MongoDB
    sensor = db["sensors"].find_one({"_id": sensor_id})
    if not sensor:
        return None

    # Retrieve sensor type
    type = sensor.get("type", "")

    return type

def get_sensor_data(sensor_id):
    # Find the sensor in MongoDB
    sensor = db["sensors"].find_one({"_id": sensor_id})
    if not sensor:
        return None

    # Retrieve sensor values
    values = sensor.get("values", [])

    return values

def detect_outliers(sensor_data):
    if not sensor_data:
        return {"outlier": False}

    values = np.array([entry['value'] for entry in sensor_data])
    mean = np.mean(values)
    std_dev = np.std(values)

    # Define threshold for outliers
    threshold = 3 * std_dev

    # Detect outliers
    outliers = np.abs(values - mean) > threshold

    return {"outlier": np.any(outliers)}

def detect_anomalies(sensor_data):
    if not sensor_data:
        return {"anomaly": False}

    values = np.array([entry['value'] for entry in sensor_data])
    mean = np.mean(values)
    std_dev = np.std(values)

    # Define threshold for anomalies
    threshold_upper = mean + 3 * std_dev
    threshold_lower = mean - 3 * std_dev

    # Detect anomalies
    anomalies = (values > threshold_upper) | (values < threshold_lower)

    return {"anomaly": np.any(anomalies)}

def generate_frequency_recommendation(sensor_data):
    if not sensor_data:
        return None

    mean, std_dev, diff_values = calculate_statistics(sensor_data)
    recommendation = generate_recommendation(mean, std_dev, diff_values)

    return recommendation

def generate_recommendations(outlier_recommendation, anomaly_recommendation):
    recommendations = []
    
    # Check for anomalies
    if anomaly_recommendation["anomaly"]:
        recommendations.append("Recommendation 2: Investigate potential anomalies in the sensor data.")

    # Check for outliers
    if outlier_recommendation["outlier"]:
        recommendations.append("Recommendation 2: Investigate potential issues with the sensor or the environment due to outliers.")

    # If no anomaly or outlier and sensor is working normally
    if not anomaly_recommendation["anomaly"] and not outlier_recommendation["outlier"]:
        recommendations.append("Recommendation 2: Sensor is functioning normally.")

    return recommendations

def calculate_statistics(sensor_data):
    values = [entry['value'] for entry in sensor_data]
    mean = np.mean(values)
    std_dev = np.std(values)
    diff_values = np.diff(values)
    return mean, std_dev, diff_values

def generate_recommendation(mean, std_dev, diff_values):
    # Analyze patterns in the sensor data
    max_diff = np.max(np.abs(diff_values))

    # Check if values are nearly constant
    is_constant = max_diff < 0.1 * mean  # Adjust threshold as needed

    # Check if values are increasing or decreasing
    is_increasing = np.all(diff_values > 0)
    is_decreasing = np.all(diff_values < 0)

    # Generate recommendations based on patterns
    recommendation = ""

    # If values are nearly constant, recommend reducing data collection frequency
    if is_constant:
        recommendation = 'Recommendation 1: Reduce data collection frequency. Sensor values are nearly constant.'

    # If values are increasing, recommend increasing data collection frequency
    if is_increasing:
        recommendation = 'Recommendation 1: Increase data collection frequency. Sensor values are increasing.'

    # If values are decreasing, recommend increasing data collection frequency
    if is_decreasing:
        recommendation = 'Recommendation 1: Increase data collection frequency. Sensor values are decreasing.'

    # If values have high variance, recommend periodic data collection
    if std_dev > 10:  # Adjust threshold as needed
        recommendation = 'Recommendation 1: Implement periodic data collection to capture variations.'

    return recommendation

def signal_handler(sig, frame):
    print("Ctrl+C detected. Exiting gracefully.")
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    app.run(debug=True)
