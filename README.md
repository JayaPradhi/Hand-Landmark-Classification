# Hand-Landmark-Classification


1.Dataset Gathering with OpenCV: OpenCV is an open-source computer vision and machine learning software library. In this step, using it along with a webcam to capture custom frames. These frames likely contain images of hands in various positions or gestures. After capturing these frames, storing the data in a repository, which could be a folder on your computer or a database.

![image](https://github.com/JayaPradhi/Tamil_sign_language/assets/127920413/3910af09-8229-4b80-b27d-cb0ffd54d8fa)


2.Feature Extraction with Matiapip Hand Landmarks Detector: Matiapip is likely a library or tool for hand landmark detection. Hand landmark detection involves identifying key points or landmarks on a hand, such as fingertips, knuckles, and the palm. By using this tool, able to extract these features from the captured images of hands. Once extracted, these features are likely represented as numerical values or coordinates. After extracting the features, saving them in a pickle file, which is a way to serialize Python objects for storage or transmission.


![image](https://github.com/JayaPradhi/Tamil_sign_language/assets/127920413/cba5cace-1770-4758-86c7-1cb047d23e51)



3.Random Forest Model: Random forest is a popular machine learning algorithm that's effective for classification tasks. In this step, building a random forest model and training it using the features extracted from the hand landmarks. The model consists of 100 decision trees, which work together to make predictions. Each decision tree learns to classify the hand landmarks based on the features extracted from them. After training the model, achieved a classification accuracy rate of 95%. This means that when the model is given new hand landmark data, it can correctly classify the landmarks with 95% accuracy.

![image](https://github.com/JayaPradhi/Tamil_sign_language/assets/127920413/b1d9daad-5810-420f-a119-c28feb65484e)

