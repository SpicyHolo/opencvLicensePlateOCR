import cv2
import os
import numpy as np


def load_license_plates():
    path = "./data/img/"
    files = os.listdir(path)
    image_files = list(
                    filter(lambda fname: fname.endswith(".jpg"), files)
    )   

    plate_numbers = list(
                    map(lambda fname: fname.replace('.jpg', ''), image_files)
    )
    
    images = list(
                    map(lambda fname: cv2.imread(path + fname), image_files)
    )

    data = [{'img': img, 'num': num} for img, num in zip(images, plate_numbers)]
    return data

class ImageSearch:
    def __init__(self, img):
        self.src = img.copy()
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.thresh = self.thresh(self.gray)
        self.result_img = img.copy()
        self.license_polygon = self.findLicensePolygon(self.thresh)
        if len(self.license_polygon) == 4:
            self.result_img = self.transformImage(self.license_polygon, self.gray)

    def __call__(self):
        return self.result_img

    def thresh(self, gray: np.array):
        "Applies thresholding to gray image, before edge detection, returns: Binary Image"
    
        # Binary thresholding
        _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY) # TODO adjust thresholding if the whole is not visible

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        morph = thresh
        #morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel, iterations=1)
        #morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        #morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=1) # TODO possibly remove
        #morph = cv2.dilate(morph, None, iterations=2)

        # Gaussian Blur, for noise reduction
        blur = cv2.GaussianBlur(morph, (3, 3), 0)
        return blur
    
    def findLicensePolygon(self, thresh: np.array, area_range=(280000, 1800000), ratio_range=(1.5, 5), approx_poly_epsilon=0.01):
        "#TODO"
        # Get contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area and ratio, to target only license platesj
        min_area, max_area = area_range
        min_ratio, max_ratio = ratio_range
        contours = list(
            filter(lambda c: min_ratio < self.getContourBoundingRectangleRatio(c) < max_ratio, 
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
        
        # Need additional filtering if the result is still not a polygon with 4 vertices
 
            
            # Some images, result in unexpected polygons, that may cut part of the license plate
            # this deals with that issue
            #target_contour  = cv2.convexHull(target_contour)
        return target_contour

    def transformImage(self, license_contour, gray: np.array):

        # Create final image containing only license plate bounding rectangle
        mask = np.zeros_like(gray)
        x, y, w, h = cv2.boundingRect(license_contour)
        cv2.fillPoly(mask, [license_contour], color=255)

        mask_inv = cv2.bitwise_not(mask)

        # Clear every pixel except for license plate area
        gray[mask_inv == 255] = 0

        # Cut out license plate bounding box
        cut_img = np.zeros((h, w), np.uint8)
        cut_img = gray[y:y+h, x:x+w]
        
        # Create transform
        corners = sorted(np.concatenate(license_contour).tolist())
        pts = self.order_points(corners)
        (tl, tr, br, bl) = pts

        # Finding the maximum width.
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
        final = cv2.warpPerspective(gray, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LINEAR)
        return final

    def getContourBoundingRectangleRatio(self, c: list):
        rect = cv2.boundingRect(c)
        x,y,w,h = rect

        return w/h

    def order_points(self, pts):
        '''Rearrange coordinates to order:
        top-left, top-right, bottom-right, bottom-left'''
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
    
    key = ord('a')
    i = 0
    flag = True
    while key != ord('q'):
        src = data[i]['img']
        image_search = ImageSearch(src)

        plate = fit_img(image_search())

        cv2.imshow("src", fit_img(src))
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
if __name__ == "__main__":
    main()