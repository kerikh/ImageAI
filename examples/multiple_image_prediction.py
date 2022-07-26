from imageai.Prediction import ImagePrediction
import os

execution_path = os.getcwd()

multiple_prediction = ImagePrediction()
multiple_prediction.setModelTypeAsResNet()
multiple_prediction.setModelPath(os.path.join(execution_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
multiple_prediction.loadModel()

all_files = os.listdir(execution_path)
all_images_array = [
    each_file
    for each_file in all_files
    if (each_file.endswith(".jpg") or each_file.endswith(".png"))
]

results_array = multiple_prediction.predictMultipleImages(all_images_array, result_count_per_image=5)

for each_result in results_array:
    predictions, percentage_probabilities = each_result["predictions"], each_result["percentage_probabilities"]
    for index in range(len(predictions)):
        print(f"{predictions[index]} : {percentage_probabilities[index]}")
    print("-----------------------")