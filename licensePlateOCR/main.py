import cv2
import os
import numpy as np
import warnings
from itertools import combinations

def load_license_plates():
    path = "./data/img/"
    files = os.listdir(path)
    image_files = list(filter(lambda fname: fname.endswith(".jpg"), files))   

    plate_numbers = list(map(lambda fname: fname.replace('.jpg', ''), image_files))

    images = list(map(lambda fname: cv2.imread(path + fname), image_files))

    data = [{'img': img, 'num': num} for img, num in zip(images, plate_numbers)]
    return data

class ImageSearch:
    def __init__(self, img, approx_poly_epsilon):

        self.src = img.copy()

        # Filtering
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.thresh = self.thresh(self.gray)
        
        # Find license plate rectangle
        self.license_polygon = self.findLicensePolygon(self.thresh, approx_poly_epsilon=0.01)

        if len(self.license_polygon) != 4:
            self.license_polygon = self.findLicensePolygon(self.thresh, approx_poly_epsilon=0.02)
            if len(self.license_polygon) != 4:
                warnings.warn("Could not locate license plate")

                self.result_img = cv2.drawContours(self.src, self.license_polygon, -1, (0, 0, 255), 10)
                # Find points that are the closest to a rectangle:
                #self.license_polygon = self.bounding_parallelogram(self.license_polygon)  
                return
        
        # Warp license plate to a rectangle
        self.license_plate = self.warpImage(self.license_polygon, self.gray)

        # Find character boundaries
        self.character_segments = self.segmentCharacters(self.license_plate)
        
        # Draw character boundaries
        self.result_img = cv2.cvtColor(self.license_plate, cv2.COLOR_GRAY2BGR)

        for seg in self.character_segments:
            x, y, w, h = seg
            cv2.rectangle(self.result_img, (x, y), (x+w, y+h), (255,0,0), 10)

    def bounding_parallelogram(self, contour):
        # Compute the convex hull of the contour
        hull = cv2.convexHull(contour)

        # Fit a rotated rectangle to the convex hull
        rect = cv2.minAreaRect(hull)

        # Extract the vertices of the rotated rectangle
        box = cv2.boxPoints(rect)

        # Convert the floating-point coordinates to integers
        box = np.intp(box)
        box = box.reshape((-1, 1, 2))
        return box

    def __call__(self):
        return self.result_img

    def thresh(self, gray: np.array):
        "Applies thresholding to gray image, before edge detection, returns: Binary Image"
    
        # Binary thresholding
        _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY) # TODO adjust thresholding if the whole is not visible

        # Gaussian Blur, for noise reduction
        blur = cv2.GaussianBlur(thresh, (3, 3), 0)
        return blur
    
    def findLicensePolygon(self, thresh: np.array, area_range=(280000, 1800000), ratio_range=(1.5, 5), approx_poly_epsilon=0.01):
        "Given binary image, returns contour of a license plate"

        def getRectRatio(rect):
            "Returns the aspect retio of a rectangle"
            x, y, w, h = rect
            return w/h
        
        # Get contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area and ratio, to target only license plates
        min_area, max_area = area_range
        min_ratio, max_ratio = ratio_range
        contours = list(
            filter(lambda c: min_ratio < getRectRatio(cv2.boundingRect(c)) < max_ratio, 
            filter(lambda c: min_area < cv2.contourArea(c) < max_area, contours)
        ))

        # Approximate contours with polygons
        approxContours = []
        for c in contours:
            arc = cv2.arcLength(c, closed=True)
            poly = cv2.approxPolyDP(c, approx_poly_epsilon*arc, closed=True)
            approxContours.append(poly)

        # Exclude complex Convex shapes, by sorting by perimeter/area ratio, and choosing the smallest
        approxContours = sorted(approxContours, key=lambda p: cv2.arcLength(p, closed=True) / cv2.contourArea(poly))
        target_contour = approxContours[0]
        
        # TODO, Need additional filtering if the result is still not a polygon with 4 vertices
        return target_contour

    def segmentCharacters(self, license_plate_img):
        def getRectRatio(rect):
            "Returns the aspect retio of a rectangle"
            x, y, w, h = rect
            return w/h

        def getRectArea(rect):
            "Returns the area of a rectangle"
            x, y, w, h = rect
            return w*h


        def is_inside(rect1, rect2):
            x1, y1, w1, h1 = rect1
            x2, y2, w2, h2 = rect2

            # Check if all vertices of rect1 are inside rect2
            return x2 <= x1 and y2 <= y1 and x2 + w2 >= x1 + w1 and y2 + h2 >= y1 + h1
        
        def getRectDeviation(rect, mean):
            "Returns value [0-1] of how much the area of given rectangle deviates from mean area"
            area = getRectArea(rect)
            return abs(area-mean) / mean

        # Filtering, to get character contours
        blur = cv2.GaussianBlur(license_plate_img, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(binary, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #self.license_plate  = cv2.drawContours(self.license_plate, contours, -1, (255, 0, 255), 10)

        # Filter contours, by ratio, and get 8 bigget area rectangles (at most 8 characters in a registration plates)
        boundingRects = list(map(cv2.boundingRect, contours))
        boundingRects = list(filter(lambda rect: 0.1 < getRectRatio(rect) < 0.99 , boundingRects))
        boundingRects = sorted(boundingRects, key=getRectArea, reverse=True)
        boundingRects = boundingRects[:8]

        #Characters like [O, 0] sometimes result in two overlapping rectangles
        # Remove all bounding rectangles that are enclosed
        boundingRects = list(filter(lambda rectA: not(
                any([is_inside(rectA, rectB) and (getRectArea(rectA) < getRectArea(rectB)) for rectB in boundingRects])), 
                boundingRects
        ))

        # Calculate, for each area how far it deviates from mean area
        # Delete the odd rectanggle if deviation is over a value 
        epsilon = 0.7

        mean = np.mean(list(map(getRectArea, boundingRects)))
        boundingRects = sorted(boundingRects, key=lambda rect: getRectDeviation(rect, mean), reverse=True)
        if getRectDeviation(boundingRects[0], mean) > epsilon:
            pass
            boundingRects = boundingRects[1:]

        boundingRects = sorted(boundingRects, key=lambda rect: rect[0])  # Sort left to right

        if len(boundingRects) < 7 or len(boundingRects) > 8:
            warnings.warn(f"Found incorrect number of characters of license plate ({len(boundingRects)})")

        return boundingRects

    def warpImage(self, license_contour, gray: np.array):
        "Given license plate contour, and gray image, find its corners and warp the image to a rectangle"

        # Create final image containing only license plate bounding rectangle
        mask = np.zeros_like(gray)
        x, y, w, h = cv2.boundingRect(license_contour)
        cv2.fillPoly(mask, [license_contour], color=255)

        mask_inv = cv2.bitwise_not(mask)

        # Clear every pixel except for license plate area
        gray[mask_inv == 255] = 0

        # Create transform
        corners = sorted(np.concatenate(license_contour).tolist())
        pts = self.order_points(corners)
        (tl, tr, br, bl) = pts

        # Finding the maximum width
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Finding the maximum height.
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Final destination co-ordinates.
        destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

        # Getting the homography.
        M = cv2.getPerspectiveTransform(np.float32(pts), np.float32(destination_corners))

        # Perspective transform using homography
        warped  = cv2.warpPerspective(gray, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LINEAR)
        return warped

    def order_points(self, pts):
        "Given pts of a rectangle, rearrange them in order: top-left, topr-giht, bottom-right, bottom-left"

        rect = np.zeros((4, 2), dtype='float32')
        pts = np.array(pts)
        s = pts.sum(axis=1)
        # Top-left point will have the smallest sum.
        rect[0] = pts[np.argmin(s)]
        # Bottom-right point will have the largest sum.
        rect[2] = pts[np.argmax(s)]
    
        diff = np.diff(pts, axis=1)
        # Top-right point will have the smallest difference.
        rect[1] = pts[np.argmin(diff)]
        # Bottom-left will have the largest difference.
        rect[3] = pts[np.argmax(diff)]
        # Return the ordered coordinates.
        return rect.astype('int').tolist()
    
def fit_img(img):
    h, w = img.shape[:2]
    scale = 720/w
    scaled_img = cv2.resize(img, None, fx=scale, fy=scale)
    return scaled_img

def main():
    data = load_license_plates()
    epsilon = 0.01
    key = ord('a')
    i = 0
    flag = True
    while key != ord('q'):
        src = data[i]['img']
        image_search = ImageSearch(src, epsilon)

        plate = image_search()
        src = cv2.resize(src, None, fx=1080/src.shape[1], fy=1080/src.shape[1])
        plate = cv2.resize(plate, None, fx=1080/plate.shape[1], fy=1080/plate.shape[1])
        plate = cv2.rectangle(plate, (5, 0), (1, 30), (255, 255, 255), -1)
        plate = cv2.putText(plate, f'ID: {i:02}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.FILLED)
        cv2.imshow("src", src)
        cv2.imshow("plate", plate)

        key = cv2.waitKey(50)
        if key == ord('h'):
            flag = True
            i = (i - 1) % len(data)
        elif key == ord('l'):
            i = (i + 1) % len(data)
            flag = True
        else:
            flag = False

        if key == ord('z'):
            epsilon = max(0.01, epsilon - 0.01)
            print(epsilon)
        elif key == ord('x'):
            epsilon += 0.01
            print(epsilon)

if __name__ == "__main__":
    main()