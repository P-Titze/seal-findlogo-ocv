#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>#
#include <fstream>

using namespace cv;
using namespace std;
Point foundPoint;
char* filename = (char*)"..\\examples\\Testdocs_2.tif";
char* template1_name = (char*)"..\\examples\\ABB_black1.jpg";
char* template2_name = (char*)"..\\examples\\ABB_black2.jpg";
char* template3_name = (char*)"..\\examples\\ABB_black3.jpg";

const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.01f;

int main(int argc, char** argv)
{
    //read cli arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-file") == 0 || strcmp(argv[i], "-f") == 0) {
            i++;
            filename = argv[i];
        }
        if (strcmp(argv[i], "-template") == 0 || strcmp(argv[i], "-t") == 0) {
            i++;
            template1_name = argv[i];
        }
    }

    Mat t = imread(template1_name);
    Mat t2 = imread(template2_name);
    Mat t3 = imread(template3_name);
    Mat f = imread(filename);
    Mat res;



    // Variables to store keypoints and descriptors
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2, output;

    // Detect ORB features and compute descriptors.
    Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
    orb->detectAndCompute(f, Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(t, Mat(), keypoints2, descriptors2);
    //orb->detectAndCompute(t2, Mat(), keypoints2, descriptors2);
    //orb->detectAndCompute(t3, Mat(), keypoints2, descriptors2);


    std::vector<DMatch> matches;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptors1, descriptors2, matches, Mat());

    sort(matches.begin(), matches.end());


    // Remove not so good matches
    const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
    matches.erase(matches.begin() + numGoodMatches, matches.end());

    map<float, map<float, int>> points{};
    // Draw top matches
    Mat imMatches;
    for (int i = 0; i < matches.size(); i++) {
        Point2f p = keypoints1[matches[i].queryIdx].pt - keypoints2[matches[i].trainIdx].pt;
        cout <<"\n" << p;
        auto iter = points.find(p.x);
        if (iter != points.end()) {
            map<float, int> m = iter->second;
            auto iter2 = m.find(p.y);
            if (iter2 != m.end()) {
                m[p.y]++;
            }
            else {
                m[p.y] = 1;
            }
        }
        else {
            points[p.x][p.y] = 1;
        }
    }

    for (auto x : points) {
        // ent1.first is the first key
        for (auto y : x.second) {
            // ent2.first is the second key
            // ent2.second is the data
            rectangle(f, Point(x.first, y.first), Point(x.first + t.cols, y.first + t.rows), Scalar(0, 0, 0), y.second);
        }
    }
    imwrite((string)filename+".out.jpg", f);
    drawMatches(f, keypoints1, t, keypoints2, matches, imMatches);
    imwrite("matches.jpg", imMatches);

    drawKeypoints(t, keypoints1, output, Scalar(0, 0, 255));
    imwrite("debug.jpg", output);

    return 0;
}