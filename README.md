# opencv license plate recognision
The project's goal is to recognize license plate number on a static image.
With the following assumptions:
- Plates on a photo won't be tilted to more than $\pm 45^{\dot}$ degrees relative to horizontal position.
- Plate's longer edge will take at least 0.33 of the photo's wid`th
- Angle between camera's optic axis and the plate's surface won't be bigger than $\pm 45^{\dot}$
- Data contains photos of Polish plates, black characters on white backgroud (7 or 8 characters on each).
- Photos can have different picture resolution.

# How the detection works
The algorithm consists of 3 parts:
- License plate polygon detection (with 4 corners), and transforming its area into a rectangle
- Segmentation into individual characters
- Pattern matching as OCR

## Polygon detection
### Step 1: Initial filtering
I use initial filtering to improve edge detection:
- Convert image to grayscale
- Apply binary thresholding
- Apply Median Blur, morphological close operation
### Step 2: Find and filter contours
For edge detection on filtered binary image, I used simple OpenCV function `cv2.findContours()`.
And filtered the detected contours by area and it's bounding rectangle aspect ratio.
This helped me detect rectangular-ish shapes with correct area.
### Step 3: Approximate contour with a polygon that has 4 vertices.
It's best to use OpenCV function `cv2.approxPolyDP()` with it's epsilon scaled by the contour's length.
However, on some images it is unable to find a polygon that has exactly 4 vertices, so i those cases I use a different approach.

After finding an approximat polygon, take the most top-left and most top-right points of contour, and take them as top-left and top-right corners.
For the bottom corners, simply find contour point, that is the closest to respective corner based on it's x coordinate (width).
This works quite well for the cases, when found polygon has more than 4 vertices (this happens when some edge of license plate is hard to detect as a whole).
### Step 4: Transform roi to a rectangle
Given 4 ordered points of a polygon, transform the image to a rectangle, using simple transformations.

## Segmentation
### Step 1: Initial Filtering
After finding the license plate roi, I applied binary thresholding and medianBlur once again, and expanded the image with a white border, to help with finding character edges.
### Step2 : Find and filter contours
Find contours as previously, calcualte `boundingRect()` for each, but this time I'm using a bigger weapon in terms of filtering:
- Filter by rectangle area based on total image area.
- Filter by aspect ratio, to exclude some very flat contours, near the edges of plate.
- Filter by minimal distance from border, to avoid mistaking license plate border for a character.
- Select 8 biggest characters by area.
After sorting them from left to right, we're done.

Before pattern matching, I expanded each character border by some pixels in each direction.
And applied some median blur and binary + Otsu thresholding.

## Pattern Matching
I used a font to generate pattern images, and then used OpenCV's `cv2.matchTemplate()`, to find the character pattern with highest correlation.
Before that, I scale the target image to template's dimension, which works suprisingly well.

I'm also matching only letters for first two characters, since this is the Polish format of plate numbers.

