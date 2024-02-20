#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <sstream>

// This Support Vector Machine is named Blaze :)
int main() {

    // Size of the linear set
    const int linearSize = 8;

    // Number of training iterations (notable mentions: 1, 100, 500, 1000, 10000, 100000, 1000000)
    const int maxIter = 50;
    int iterCount = 0;

    bool displayOverTime = true;                                                                    // Set this to false for an instant result

    // Set up the training data
    int labels[linearSize] = {1, 1, -1, -1, -1, 1, 1, 1};
    float trainingData[linearSize][2] = { {501, 10}, {501, 30}, {255, 10}, {501, 255}, {10, 501}, {255, 40}, {10, 230}, {100, 260}};

    cv::Mat trainingDataMat(linearSize, 2, CV_32F, trainingData);
    cv::Mat labelsMat(linearSize, 1, CV_32SC1, labels);

    // Set up the parameters
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);                                                        // Most common type of n-classification (where n >= 2)
    svm->setKernel(cv::ml::SVM::LINEAR);
    

    // Training the support vector machine (moved so data can be seen over time)
    // svm->train(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat);                           // Row_Sample means that every row is a sample

    // Visual Representation data (Not important for machine learning aspect)
    int width = 512, height = 512;
    cv::Mat display = cv::Mat::zeros(width, height, CV_8UC3);

    // Defining colors 
    cv::Vec3b red(0, 0, 255), blue(255, 0, 0);

    if(displayOverTime) {
        // Training the support vector machine over time
        for(iterCount; iterCount < maxIter; iterCount++) {
            svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, iterCount, 1e-9));  // Training terminates after limited iterations with a tolerance of 1e-6
            svm->train(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat);

            // Predicting classification regions
            for(int i = 0; i < display.rows; i++) {
                for(int j = 0; j < display.cols; j++) {
                    cv::Mat sampleData = (cv::Mat_<float>(1, 2) << j, i);
                    float response = svm->predict(sampleData);                                    // Predicts the response for the provided data

                    if(response == 1)
                        display.at<cv::Vec3b>(i, j) = red;

                    else if(response == -1)
                        display.at<cv::Vec3b>(i, j) = blue;
                }
            }

            // Show Training Data
            int thickness = -1;

            cv::circle(display, cv::Point(501, 10), 5, cv::Scalar(0, 0, 0), thickness);
            cv::circle(display, cv::Point(501, 30), 5, cv::Scalar(0, 0, 0), thickness);
            cv::circle(display, cv::Point(255, 10), 5, cv::Scalar(255, 255, 255), thickness);
            cv::circle(display, cv::Point(501, 255), 5, cv::Scalar(255, 255, 255), thickness);
            cv::circle(display, cv::Point(10, 501), 5, cv::Scalar(255, 255, 255), thickness);
            cv::circle(display, cv::Point(255, 40), 5, cv::Scalar(0, 0, 0), thickness);
            cv::circle(display, cv::Point(10, 230), 5, cv::Scalar(0, 0, 0), thickness);
            cv::circle(display, cv::Point(100, 260), 5, cv::Scalar(0, 0, 0), thickness);

            // Showing support vectors
            thickness = 2;
            cv::Mat supportVectors = svm->getUncompressedSupportVectors();

            for(int i = 0; i < supportVectors.rows; i++) {
                float* v = supportVectors.ptr<float>(i);
                cv::circle(display, cv::Point((int) v[0], (int) v[1]), 6, cv::Scalar(0, 200, 0), thickness);
            }

            // Display iteration count
            std::stringstream ss;
            ss << "Iteration: " << iterCount + 1;
            cv::putText(display, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);


            // Show the display to the user
            cv::imshow("Classified Data :)", display);
            cv::waitKey(1);
        }
    } else {
        // Training the support vector machine over time
        svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, maxIter, 1e-9));  // Training terminates after limited iterations with a tolerance of 1e-6
        svm->train(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat);

        // Predicting classification regions
        for(int i = 0; i < display.rows; i++) {
            for(int j = 0; j < display.cols; j++) {
                cv::Mat sampleData = (cv::Mat_<float>(1, 2) << j, i);
                float response = svm->predict(sampleData);                                    // Predicts the response for the provided data

                if(response == 1)
                    display.at<cv::Vec3b>(i, j) = red;

                else if(response == -1)
                    display.at<cv::Vec3b>(i, j) = blue;
            }
        }

        // Show Training Data
        int thickness = -1;

        cv::circle(display, cv::Point(501, 10), 5, cv::Scalar(0, 0, 0), thickness);
        cv::circle(display, cv::Point(501, 30), 5, cv::Scalar(0, 0, 0), thickness);
        cv::circle(display, cv::Point(255, 10), 5, cv::Scalar(255, 255, 255), thickness);
        cv::circle(display, cv::Point(501, 255), 5, cv::Scalar(255, 255, 255), thickness);
        cv::circle(display, cv::Point(10, 501), 5, cv::Scalar(255, 255, 255), thickness);
        cv::circle(display, cv::Point(255, 40), 5, cv::Scalar(0, 0, 0), thickness);
        cv::circle(display, cv::Point(10, 230), 5, cv::Scalar(0, 0, 0), thickness);
        cv::circle(display, cv::Point(100, 260), 5, cv::Scalar(0, 0, 0), thickness);

        // Showing support vectors
        thickness = 2;
        cv::Mat supportVectors = svm->getUncompressedSupportVectors();

        for(int i = 0; i < supportVectors.rows; i++) {
            float* v = supportVectors.ptr<float>(i);
            cv::circle(display, cv::Point((int) v[0], (int) v[1]), 6, cv::Scalar(0, 200, 0), thickness);
        }

        // Display iteration count
        std::stringstream ss;
        ss << "Iteration: " << maxIter;
        cv::putText(display, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);


        // Show the display to the user
        cv::imshow("Classified Data :)", display);
    
    }

    // Save data to an image
    cv::imwrite("images/iteration_" + std::to_string(maxIter) + ".png", display);
    

    cv::waitKey();                                                                             // Waits for a key to be pressed to disappear


    return 0;
}