# HousingPriceEstimate
Novel Housing Price Estimator 

<img src="https://user-images.githubusercontent.com/73147643/147277252-0a22b542-588c-4cee-bb75-1ebbc9d6f8b7.jpg" height="300" />

## Reference:
Cite here: Nouriani, A. and Lemke, L., 2022. Vision-based housing price estimation using interior, exterior & satellite images. Intelligent Systems with Applications, 14, p.200081.

## Example input data

<img src="https://user-images.githubusercontent.com/73147643/147276357-43d9e8ba-d956-4531-b3ac-385429826443.jpg" height="150" /><img src="https://user-images.githubusercontent.com/73147643/147276358-34cf86bb-d42c-446e-844e-6ef479df556f.jpg" height="150" /><img src="https://user-images.githubusercontent.com/73147643/147276359-fdba0382-0804-4fb7-b458-9354cf3981d1.jpg" height="150" /><img src="https://user-images.githubusercontent.com/73147643/147276360-40938c47-c751-413b-8238-f37fdcbf8b5c.jpg" height="150" /><img src="https://user-images.githubusercontent.com/73147643/147276361-32fe0fc2-f0ff-4b3e-b895-415081440d7f.jpg" height="150" /><img src="https://user-images.githubusercontent.com/73147643/147276362-c73a4b7a-82a2-4b03-b998-1dad44ec9584.jpg" height="150" /><img src="https://user-images.githubusercontent.com/73147643/147276363-177ac0d5-c274-44c4-a8fa-c80ab7d9dbe9.jpg" height="150" />

## Instructions
- run main.py to see final results
- Regression.py produces and compares different regression models for price estimation
- googlenet_feature_extract.py extracts googleNet two last FC layers for luxury level classification
- lux_level_classification.py classifies luxury levels of images using KNN and SVM methods
- vgg_lux_level_classification.py classifies luxury levels of images using transfer learning and fine-tuned modified vgg16  
- room_classifier.py classifies images to different categories of satellite, bathroom, bedroom, dining room, kitchen, living room, front view
- data_process.py are used for moving and sorting images to the relevent folders 
- text_data.xlsx contains meta data of the houses


## Access to our dataset and saved trained models
You should ask permisson to get access to:
https://drive.google.com/drive/folders/1kKHnRHu9LuhcKsWNuKFHtFZIAu741_Eg?usp=sharing
