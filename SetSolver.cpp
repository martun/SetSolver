#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <time.h>

using namespace std;
using namespace cv;

void FindBlobs(const cv::Mat &binary, std::vector<cv::Rect> &blobs)
{
	blobs.clear();
	
	cv::Mat label_image;
	binary.convertTo(label_image, CV_32SC1);
	//// imwrite("label_image.jpg", label_image);

	// Skip values near the very corners of the image.
	for (int y = 10; y < label_image.rows - 10; y++) {
		int *row = (int*)label_image.ptr(y);
		for (int x = 10; x < label_image.cols - 10; x++) {
			if (row[x] == 0) {
				continue;
			}

			cv::Rect rect;
			cv::floodFill(label_image, cv::Point(x, y), 0, &rect, 0, 0, 4);
			// Skip blobs smaller than 2% of the total image.
			if (rect.width > label_image.cols / 15 && rect.height > label_image.rows / 15) {
				blobs.push_back(rect);
				// imwrite("label_image.jpg", label_image);
			}
			cv::rectangle(label_image, rect, cv::Scalar(0), FILLED, LINE_8);
		}
	}
}

enum Card_type {
	ROUND = 0,
	DIAMOND = 1,
	WAVE = 2,
	NOT_A_CARD_SOME_OTHER_WHITE_OBJECT = 3
};
enum Card_color {
	RED = 0,
	GREEN = 1,
	PURPLE = 2
};
enum Card_count {
	ONE = 0,
	TWO = 2,
	THREE = 3
};
enum Card_filled {
	FULL = 0,
	EMPTY = 1,
	PARTIAL = 2
};

struct Card {
	Card_type type;
	Card_color color;
	Card_count count;
	Card_filled filled;
};

void clear_card_symbol(Mat& card_img) {
	// Sometimes when the card is a bit rotated, 
	// and we cut some part of it which is supposed to contain the shape,
	// I.E. a diamond, some part of the diamond under it gets into the image.
	// We want to clean it up.

	for (int x = 0; x < card_img.rows; x++) {
		for (int y = 0; y < card_img.cols; y++) {
			// Only look for crap near the 3% corners of the image.
			if ((x > 2 && x < card_img.rows - 2) &&
				(y > 2 && y < card_img.cols - 2))
				continue;
			// If completely white or completely black, skip.
			if (card_img.at<Vec3b>(x, y) == Vec3b(255, 255, 255) ||
				card_img.at<Vec3b>(x, y) == Vec3b(0, 0, 0)) {
				continue;
			}

			// Make the region completely white.
			cv::Rect rect;
			cv::floodFill(
				card_img,
				cv::Point(y, x),
				cv::Scalar(255, 255, 255),
				&rect,
				cv::Scalar(30, 30, 30), // can go to much lighter pixels than itself.
				cv::Scalar(70, 70, 70), // but not to darker ones too much.
				8);
		}
	}

	// imwrite("current_card_part_whole_partially_cleared.jpg", card_img);

	// Now we want to make all completely white pixels to black,
	// Also delete the edges.
	for (int i = 0; i < card_img.rows; ++i) {
		for (int j = 0; j < card_img.cols; ++j) {
			if (card_img.at<Vec3b>(i, j) == Vec3b(255, 255, 255)) {
				for (int ich = -1; ich <= 1; ++ich) {
					for (int jch = -1; jch <= 1; ++jch) {
						int ii = i + ich;
						int jj = j + jch;
						if (ii >= 0 && ii <= card_img.rows - 1 &&
							jj >= 0 && jj <= card_img.cols - 1) {
							card_img.at<Vec3b>(ii, jj) = Vec3b(0, 0, 0);
						}
					}
				}
			}
		}
	}

	// imwrite("current_card_part_cleared.jpg", card_img);
}

void clear_card(Mat& card_img) {
	// Sometimes when the card is a bit rotated, there are 
	// grey areas on the corners. I don't want to just crop each image
	// let's find non-white points in the corners and 
	// fill them with white.

	int percent = 3;
	for (int x = 0; x < card_img.rows; x++) {
		for (int y = 0; y < card_img.cols; y++) {
			// Only look for crap near the 3% corners of the image.
			if ((x > card_img.rows * percent / 100 && x < card_img.rows * (100 - percent) / 100) &&
				(y > card_img.cols * percent / 100 && y < card_img.cols * (100 - percent) / 100))
				continue;
			// If completely white or completely black, skip.
			if (card_img.at<Vec3b>(x, y) == Vec3b(255, 255, 255) || 
				card_img.at<Vec3b>(x, y) == Vec3b(0, 0, 0)) {
				continue;
			}

			cv::Rect rect;
			cv::floodFill(
				card_img,
				cv::Point(y, x),
				cv::Scalar(0, 0, 0),
				&rect,
				cv::Scalar(70, 70, 70), // can go to much darker pixels that itself.
				cv::Scalar(30, 30, 30), // but not to lighter ones too much.
				8);
		}
	}

	// imwrite("current_card_whole_partially_cleared.jpg", card_img);

	// Now we want to make all completely black pixels to white,
	// Also delete the edges.
	for (int i = 0; i < card_img.rows; ++i) {
		for (int j = 0; j < card_img.cols; ++j) {
			if (card_img.at<Vec3b>(i, j) == Vec3b(0, 0, 0)) {
				for (int ich = -2; ich <= 2; ++ich) {
					for (int jch = -2; jch <= 2; ++jch) {
						int ii = i + ich;
						int jj = j + jch;
						if (ii >= 0 && ii <= card_img.rows - 1 &&
							jj >= 0 && jj <= card_img.cols - 1) {
							card_img.at<Vec3b>(ii, jj) = Vec3b(255, 255, 255);
						}
					}
				}
			}
		}
	}

	// imwrite("current_card_whole_cleared.jpg", card_img);
	
	/* // This code removed the dots on the cards, small dirts, but also removed the lines in the partially filled cards.
	// Invert the image.
	card_img = cv::Scalar(255,255,255) - card_img;
	cv::morphologyEx(card_img, card_img, CV_MOP_OPEN, Mat(), 
		cv::Point(-1, -1), 2);
	// Invert back.
	card_img = cv::Scalar(255, 255, 255) - card_img;
	// imwrite("current_card_whole_cleared_opened.jpg", card_img);
	*/

	// Now change all white to black.
	for (int x = 0; x < card_img.rows; x++) {
		for (int y = 0; y < card_img.cols; y++) {
			if (card_img.at<Vec3b>(x, y) == Vec3b(255, 255, 255)) {
				card_img.at<Vec3b>(x, y) = Vec3b(0, 0, 0);
			}
		}
	}
}

void get_average_non_black(const Mat& img, cv::Vec3d& average, int& count) {
	average[0] = 0;
	average[1] = 0;
	average[2] = 0;
	count = 0;

	// Compute average color, which is not white.
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
		{
			Vec3b intensity = img.at<Vec3b>(i, j);
			// If not completely white.
			if (intensity[0] != 0 || intensity[1] != 0 || intensity[1] != 0)
			{
				average[0] += intensity[0];
				average[1] += intensity[1];
				average[2] += intensity[2];
				++count;
			}
		}
	average[0] /= count;
	average[1] /= count;
	average[2] /= count;
}

void detect_color(Card& res, const cv::Vec3d& average) {
	// BGR colors.
	//cv::Vec3d red(100, 100, 230);
	//cv::Vec3d green(100, 190, 20);
	//cv::Vec3d purple(175, 90, 110);

	double distance_red = -average[2]; //cv::norm(red - average);
	double distance_green = -average[1]; //cv::norm(green - average);
	double distance_purple = -average[0]; // cv::norm(purple - average);

	if (distance_red < distance_green && distance_red < distance_purple) {
		res.color = Card_color::RED;
	}
	else if (distance_green < distance_purple) {
		res.color = Card_color::GREEN;
	}
	else {
		res.color = Card_color::PURPLE;
	}
}

void detect_fullness(Card& res, Mat& img) {
	// Crop the inner 30%x30%, and check how many pixels are non-white.
	cv::Rect area(img.cols * 35 / 100, img.rows * 35 / 100, 
		img.cols * 3 / 10, img.rows * 3 / 10);
	Mat part = img(area);
	cv::Vec3d average;
	int count;
	get_average_non_black(part, average, count);
	// If less than 10% filled, then empty.
	if (count < part.rows * part.cols / 10) {
		res.filled = Card_filled::EMPTY;
	}
	else if (count < part.rows * part.cols * 2 / 3) {
		// If less than 2/3 full, then lines.
		res.filled = Card_filled::PARTIAL;
	}
	else {
		res.filled = Card_filled::FULL;
	}
}

void detect_shape(Card& res, Mat& img) {
	// Clear corners of the card part, there might be some 
	// pixels from another card part, because we cut a rectangle.
	clear_card_symbol(img);

	Mat img_gray;
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	// Clean the noise.
	// imwrite("Cimg_gray.jpg", img_gray);
	
	// Blur.
	blur(img_gray, img_gray, Size(3, 3));

	// Clear the noise from the first_one.
	// cv::fastNlMeansDenoising(img_gray, img_gray);
	
	// imwrite("Cimg_gray_cleaned.jpg", img_gray);
	
	// Sharpen the image.
	/*Mat filter = (Mat_<int>(3, 3) << -1, -1, -1, 
		-1, 9, -1, 
		-1, -1, -1);
	cv::filter2D(img_gray, img_gray, -1, filter);
	// imwrite("Cimg_gray_cleaned_sharpened.jpg", img_gray);
	*/

	// Clear the small dots in the image, they sometimes cause problems.
	// Commented this, because OPEN-ing resulted to losing some corners
	// of a diamong, making a round figure.
	//cv::morphologyEx(img_gray, img_gray, CV_MOP_OPEN, Mat(),
	//	cv::Point(-1, -1), 2);
	//// imwrite("Cimg_gray_cleaned_2.jpg", img_gray);

	Mat img_bw;
	cv::threshold(img_gray, img_bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	
	// imwrite("Cthreshold_OUTPUT.jpg", img_bw);

	// Find contours
	std::vector<std::vector<cv::Point> > contours;
	vector<Vec4i> hierarchy;
	cv::findContours(
		img_bw.clone(), contours,
		hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	
	std::vector<cv::Point> approx;
	

	// Draw contous so we can see them.
	RNG rng(12345);
	Mat drawing = Mat::zeros(img_bw.size(), CV_8UC3);
	/*for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
	}*/
	
	// In case there is no large objects with a contour
	// set to this type.
	res.type = NOT_A_CARD_SOME_OTHER_WHITE_OBJECT;
	for (int i = 0; i < contours.size(); i++)
	{
		// Approximate contour with accuracy proportional
		// to the contour perimeter
		cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);
		
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		fillConvexPoly(drawing, &approx[0], approx.size(), color, LINE_8, 0);

		// imwrite("Cdrawing.jpg", drawing);

		double area = std::fabs(cv::contourArea(contours[i]));
		// Skip small objects, must be at least 25% of the whole.
		if (area < img_bw.rows * img_bw.cols * 25 / 100)
			continue;

		int points_count = approx.size();
		// Probably a diamond.
		if (points_count <= 6)
		{
			res.type = Card_type::DIAMOND;
		}
		else if (cv::isContourConvex(approx)) {
			// Hope that rounds get convex polygons, but 
			// waves don't. Just hope :)
			res.type = Card_type::ROUND;
		}
		else {
			res.type = Card_type::WAVE;
		}
	}
}

// Determines color, shape and filledness of the shape.
void determine_type(Card& res, Mat& img) {
	// imwrite("next_card.jpg", img);

	// BGR average.
	cv::Vec3d average;
	int count;
	get_average_non_black(img, average, count);

	detect_color(res, average);

	detect_fullness(res, img);

	detect_shape(res, img);
}

// Detects a card in the given region of an image.
Card detect_card(const Mat& img, cv::Rect area) {
	Card res;
	
	Mat card_img = img(area);
	// imwrite("current_card_whole_unclear.jpg", card_img);
	clear_card(card_img);
	// imwrite("current_card_whole.jpg", card_img);

	Mat card_img_dilated;
	cv::dilate(
		card_img, card_img_dilated,
		Mat(), // use default 3x3 kernel.
		Point(-1, -1),
		1, // Just 1 iteration, if 2, some parts start to connect
		1,
		1);
	// imwrite("current_card_whole_dialted.jpg", card_img_dilated);
	
	// Find out how many parts are there.
	std::vector<cv::Rect> blobs;

	Mat img_gray;
	cv::cvtColor(card_img_dilated, img_gray, cv::COLOR_BGR2GRAY);
	Mat img_bw;
	cv::threshold(img_gray, img_bw, 50, 255, CV_THRESH_BINARY);
	// imwrite("current_card_whole_bw.jpg", img_bw);
	
	FindBlobs(img_bw, blobs);
	
	// Probably this is not a card, but some other white object.
	if (blobs.size() > 3 || blobs.size() == 0)
	{
		res.type = Card_type::NOT_A_CARD_SOME_OTHER_WHITE_OBJECT;
		return res;
	}

	switch (blobs.size()) {
	case 1:
		res.count = Card_count::ONE;
		break;
	case 2:
		res.count = Card_count::TWO;
		break;
	default:
		res.count = Card_count::THREE;
	}

	// Add 3 more pixels from each size, probably they exist.
	int pixel_length = 3;
	if (blobs[0].x > pixel_length)
		blobs[0].x -= pixel_length;
	if (blobs[0].y > pixel_length)
		blobs[0].y -= pixel_length;
	if (blobs[0].x + blobs[0].width + 2* pixel_length  < card_img.cols)
		blobs[0].width += 2 * pixel_length;
	if (blobs[0].y + blobs[0].height + 2 * pixel_length < card_img.rows)
		blobs[0].height += 2 * pixel_length;
	
	Mat first_one = card_img(blobs[0]);
	
	// Take the first one, hope it's ok, and try to determine the type.
	determine_type(res, first_one);
	
	return res;
}

bool match(const Card& c1, const Card& c2, const Card& c3) {
	if (c1.type == Card_type::NOT_A_CARD_SOME_OTHER_WHITE_OBJECT ||
		c2.type == Card_type::NOT_A_CARD_SOME_OTHER_WHITE_OBJECT ||
		c3.type == Card_type::NOT_A_CARD_SOME_OTHER_WHITE_OBJECT)
		return false;

	if (!((c1.color == c2.color && c2.color == c3.color) ||
		(c1.color != c2.color && c2.color != c3.color && c1.color != c3.color)))
		return false;
	if (!((c1.type == c2.type && c2.type == c3.type) ||
		(c1.type != c2.type && c2.type != c3.type && c1.type != c3.type)))
		return false;

	if (!((c1.count == c2.count && c2.count == c3.count) ||
		(c1.count != c2.count && c2.count != c3.count && c1.count != c3.count)))
		return false;

	if (!((c1.filled == c2.filled && c2.filled == c3.filled) ||
		(c1.filled != c2.filled && c2.filled != c3.filled && c1.filled != c3.filled)))
		return false;
	return true;
}

std::vector<std::vector<int>> find_matching_cards(std::vector<Card>& cards) {
	std::vector<std::vector<int>> res;
	for (int i1 = 0; i1 < cards.size(); ++i1) {
		for (int i2 = i1+1; i2 < cards.size(); ++i2) {
			for (int i3 = i2 + 1; i3 < cards.size(); ++i3) {
				if (match(cards[i1], cards[i2], cards[i3])) {
					std::vector<int> r;
					r.push_back(i1);
					r.push_back(i2);
					r.push_back(i3);
					res.push_back(r);
				}
			}
		}
	}
	return res;
}

void equalize_histogram(const Mat& img_input, Mat& img_output) {
	//Convert the image from BGR to YCrCb color space
	cvtColor(img_input, img_output, COLOR_BGR2YCrCb);

	//Split the image into 3 channels; Y, Cr and Cb channels respectively and store it in a std::vector
	vector<Mat> vec_channels;
	split(img_output, vec_channels);

	//Equalize the histogram of only the Y channel 
	equalizeHist(vec_channels[0], vec_channels[0]);

	//Merge 3 channels in the vector to form the color image in YCrCB color space.
	merge(vec_channels, img_output);

	//Convert the histogram equalized image from YCrCb to BGR color space again
	cvtColor(img_output, img_output, COLOR_YCrCb2BGR);
}

// Makes all colors very close to white completely white.
void clean_the_white(const Mat& img_initial, Mat& img_bw) {
	Mat img_gray;
	cv::cvtColor(img_initial, img_gray, cv::COLOR_BGR2GRAY);
	
	// Cut the inner 40% of the image, and threshold based on that part only.
	Mat img_gray_50 = img_gray(cv::Rect(
		img_gray.cols * 3 / 10, img_gray.rows * 3 / 10,
		img_gray.cols * 4 / 10, img_gray.rows * 4 / 10));


	Mat img_bw_50;
	double threshold = cv::threshold(
		img_gray_50, img_bw_50, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	// Move the threshold a bit up, we know that our white is pretty white,
	// and anything very close to it must be kept out.
	threshold += (255.0 - threshold) / 2.5;

	// Now use that threshold on the initial image.
	cv::threshold(img_gray, img_bw, threshold, 255, CV_THRESH_BINARY);

	// imwrite("Ximg_gray.jpg", img_gray);
	// imwrite("Ximg_bw.jpg", img_bw);
}

void DoAll(Mat& img_initial, Mat& final_result) {
	// Reduce the size a bit.
	const int perfect_size = 1800;
	if (img_initial.rows > perfect_size && img_initial.cols > perfect_size) {
		int factor = std::min(img_initial.rows / perfect_size, img_initial.cols / perfect_size);
		cv::resize(
			img_initial, img_initial, 
			cv::Size(img_initial.cols / factor, img_initial.rows / factor));
	}
	final_result = img_initial.clone();
	
	Mat img_bw;
	clean_the_white(img_initial, img_bw);
	// imwrite("XXXXimg_bw.jpg", img_bw);
	std::vector<cv::Rect> blobs;
	
	// FindBlobs is very slow, so let's reduce image size by 5x,
	// find the blobs, and then do x5.
	Mat small;
	int small_factor = 5;
	cv::resize(
		img_bw, small,
		cv::Size(img_initial.cols / small_factor, img_initial.rows / small_factor));

	FindBlobs(small, blobs);
	for (int i = 0; i < blobs.size(); ++i) {
		blobs[i].x *= small_factor;
		blobs[i].y *= small_factor;
		blobs[i].width *= small_factor;
		blobs[i].height *= small_factor;
	}

	Mat img_bw_reversed = cv::Scalar(255) - img_bw;

	// Make a white rectangle of needed size.
	Mat cut = img_initial.clone();
	cut = cv::Scalar(255, 255, 255);
	// Make all parts of the image close to white = white.
	img_initial.copyTo(cut, img_bw_reversed);

	std::vector<Card> cards;
	for (int i = 0; i < blobs.size(); ++i) {
		Card next = detect_card(cut, blobs[i]);
		cards.push_back(next);
		// cv::rectangle(cut, blobs[i], cv::Scalar(255, 0, 0), 10);
	}

	std::vector<std::vector<int>> card_ids = find_matching_cards(cards);

	if (card_ids.size() == 0) {
		// no match found.
	}
	else {
		std::vector<cv::Scalar> colors;
		colors.push_back(cv::Scalar(0, 0, 0));
		colors.push_back(cv::Scalar(255, 0, 0));
		colors.push_back(cv::Scalar(0, 0, 255));

		// Don't show more than 3 matches, it's annoying.
		int how_many_matches_to_show = std::min((int)card_ids.size(), 3);

		for (int i = 0; i < how_many_matches_to_show; ++i) {
			if (card_ids[i].size() == 3) {
				// Show the match on the image.
				for (int j = 0; j < 3; ++j) {
					cv::Rect r = blobs[card_ids[i][j]];
					int reduction = i * std::min(r.width, r.height) / 8;
					r.x += reduction;
					r.y += reduction;
					r.width -= 2 * reduction;
					r.height -= 2 * reduction;

					//cv::rectangle(final_result, r, colors[i], 15);
					cv::circle(final_result,
						cv::Point(r.x + r.width / 2, r.y + r.height / 2),
						std::min(r.width, r.height) / 2,
						colors[i],
						7 /* thickness*/
					);
				}
			}
		}
	}
}

int main()
{
	Mat img_initial = imread("test1.jpg");
	Mat final_result;
	DoAll(img_initial, final_result);
	imwrite("FinalResult.jpg", final_result);
	waitKey(0);
	return 0;
}
