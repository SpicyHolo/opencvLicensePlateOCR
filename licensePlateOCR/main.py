import cv2
import os
import numpy as np
import warnings
from itertools import combinations

from patternMatch import pattern_match

# loads images data, returns a list of dict pairs with image array, plate number str
def load_license_plates():

    path = "./data/img/"
    files = os.listdir(path)
    image_paths = list(filter(lambda fname: fname.endswith(".jpg"), files))   

    plate_numbers = list(map(lambda fname: fname.replace('.jpg', ''), image_paths))

    images = list(map(lambda fname: cv2.imread(path + fname), image_paths))

    data = [{'img': img, 'num': num} for img, num in zip(images, plate_numbers)]
    return data

class ImageSearch:
    def __init__(self, img, epsilon, flag):
        self.img = img.copy()
        self.number = None
        self.flag = flag

        # All images
        self.gray = None
        self.binary = None
        self.contours_img = None

        # Initial Filtering
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, self.binary = cv2.threshold(self.gray, 110, 255, cv2.THRESH_BINARY)
        self.binary = cv2.medianBlur(self.binary, 7)
        self.binary = cv2.morphologyEx(self.binary, cv2.MORPH_CLOSE, (3, 3), iterations=5) 
        #self.binary = cv2.morphologyEx(self.binary, cv2.MORPH_DILATE, (3, 3), iterations=30) 

        # Get contours, filter
        self.contours, self.contour_img = self.find_contours(self.binary, np.zeros_like(self.img))
        self.filtered_contours, self.filtered_contour_img = self.filter_contours(self.contours, area=(280000, 1800000), ratio=(1.5, 5))

        # Approximate the contour
        self.license_polygon, self.license_polygon_img = self.approx_license_polygon(self.filtered_contours, epsilon=0.009)
    
        if len(self.license_polygon) != 4:
            warnings.warn("Could not find roi (license plate)")
            return

        # Warp license plate to a rectangle
        self.license_plate_img = self.warpImage(self.license_polygon, self.gray)

        ### Character segmentation
        # Initial filter
        _, self.license_binary = cv2.threshold(self.license_plate_img, 110, 255, cv2.THRESH_BINARY)
        self.license_binary = cv2.medianBlur(self.license_binary, 9)

        # Find contours, filter
        offset = 20
        self.license_binary = cv2.copyMakeBorder(self.license_binary, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=[255])


        
        character_contours, self.character_contours_img = self.find_contours(self.license_binary, np.zeros((*self.license_binary.shape, 3)))
        boundingRects = list(map(cv2.boundingRect, character_contours))

        image_area = self.license_binary.shape[0] * self.license_binary.shape[1]

        boundingRects = list(filter(lambda rect: 0.015*image_area < self.get_rect_area(rect) < 0.2*image_area , boundingRects))

        boundingRects = list(filter(lambda rect: 0.1 < self.get_rect_aspect_ratio(rect) < 0.99 , boundingRects))

        # Remove rectangles too left, right border
        boundingRects = list(filter(lambda rect: self.is_rect_away_from_border(rect, self.character_contours_img, min_dist=21), boundingRects))

        # Clean rectangles that are inside another rectangle
        boundingRects = list(filter(lambda rectA: not(
                any([self.is_inside(rectA, rectB) and (self.get_rect_area(rectA) < self.get_rect_area(rectB)) for rectB in boundingRects])), 
                boundingRects
        ))
        
        boundingRects = sorted(boundingRects, key=self.get_rect_area, reverse=True)
        boundingRects = boundingRects[:8]

        boundingRects = sorted(boundingRects, key=lambda rect: rect[0])  # Sort left to right

        # Expand border of each
        offset = 20
        boundingRects = list(map(lambda rect: (rect[0]-offset, rect[1]-offset, rect[2]+2*offset, rect[3]+2*offset), boundingRects))

        for rect in boundingRects:
            x, y, w, h = rect
            cv2.rectangle(self.character_contours_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # self.license_character_rects = self.segmentCharacters(self.license_plate_img)
        
        # Get each character image
        # self.character_imgs = self.getCharacterImages(self.license_plate_img, boundingRects)
        # for i, img in enumerate(self.character_imgs):
        #     cv2.imwrite(f'{i}.jpg', img)

        #OCR
        # match = pattern_match('./licensePlateOCR/assets/arklatrs-webfont.ttf', 100)
        # text = "" 
        # for img in self.character_imgs:
        #     text += match.match2(img)
        # self.number = text 
        
    def warpImage(self, license_pts, gray: np.array):
        
        "Given license plate contour, and gray image, find its corners and warp the image to a rectangle"

        # Create final image containing only license plate bounding rectangle
        mask = np.zeros_like(gray)
        license_contour = np.array(license_pts).reshape((-1, 1, 2))
        x, y, w, h = cv2.boundingRect(license_contour)
        cv2.fillPoly(mask, [license_contour], color=255)

        mask_inv = cv2.bitwise_not(mask)

        # Clear every pixel except for license plate area
        gray[mask_inv == 255] = 0

        tl, tr, br, bl = license_pts
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
        M = cv2.getPerspectiveTransform(np.float32(license_pts), np.float32(destination_corners))

        # Perspective transform using homography
        warped  = cv2.warpPerspective(gray, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LINEAR)
        return warped

    def find_contours(self, binary, mat):
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(mat, contours, -1, (255, 0, 255), 5)
        return contours, mat

    def filter_contours(self, contours, area, ratio):
        area_min, area_max = area
        ratio_min, ratio_max = ratio
        
        # Filter by area, ratio
        contours_filtered = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / h
            c_area = cv2.contourArea(c)
            if (ratio_min < aspect_ratio < ratio_max) and (area_min < c_area < area_max):
                contours_filtered.append(c)
        contour_img = np.zeros_like(self.img)
        cv2.drawContours(contour_img, contours_filtered, -1, (255, 255, 255), 5)

        return contours_filtered, contour_img
         
    def approx_license_polygon(self, contours, epsilon):
        approx_contours = []
        contour_img = np.zeros_like(self.img)

        for c in contours:
            arc_length = cv2.arcLength(c, closed=True)
            poly = cv2.approxPolyDP(c, epsilon*arc_length, closed=True)
            approx_contours.append(poly)

        # Filtering
        # Exclude complex Convex shapes, by sorting by perimeter/area ratio, and choosing the smallest
        approx_contours = sorted(approx_contours, key=lambda p: cv2.arcLength(p, closed=True) / cv2.contourArea(p))
        best_contour = approx_contours[0]

        contour_points = best_contour.reshape((best_contour.shape[0],2))
        contour_points = [tuple(p) for p in contour_points]

        if self.flag: print(contour_points)

        if len(contour_points) != 4:
            # Find top-left, top-right corners 
            top_left, d_tl = None, np.inf
            top_right, d_tr = None, np.inf
            for point in contour_points:
                x, y = point
                d = np.sqrt(x**2 + y**2)
                if d < d_tl:
                    d_tl = d
                    top_left = point
                w = self.img.shape[1]
                d = np.sqrt((w - x)**2 + y**2)
                if d < d_tr:
                    d_tr = d
                    top_right = point
            bottom_left, d_bl = None, np.inf
            bottom_right, d_br = None, np.inf
            for point in contour_points:
                if point != top_left:
                    dx = abs(point[0] - top_left[0])
                    dy = abs(point[1] - top_right[1])
                    if dx < d_bl and dy > 100:
                        d_bl = dx
                        bottom_left = point
                
                if point != top_right:
                    dx = abs(point[0] - top_right[0])
                    dy = abs(point[1] - top_right[1])
                    if dx < d_br and dy > 100:
                        d_br = dx
                        bottom_right = point

            final_pts = np.array([top_left, top_right, bottom_right, bottom_left])
        else:
            final_pts = self.order_points(np.concatenate(best_contour).tolist())
        tl, tr, br, bl = final_pts

        # add offset
        offset = 10
        tl[0] -= offset
        tr[0] -= offset
        bl[1] += offset
        br[1] += offset

        cv2.drawContours(contour_img,[np.array(final_pts).reshape((-1, 1, 2))], -1, (0, 0, 255), 10)
        return final_pts, contour_img

    def order_points(self,pts):
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
    
    def getCharacterImages(self, license_plate_img, bounding_rects):
        src = license_plate_img.copy()
        res_imgs = []
        for rect in bounding_rects:
            x, y, w, h = rect
            img = np.zeros((w, h), dtype=np.uint8)
            img = src[y:y+h, x:x+w]

            blur = cv2.medianBlur(img, 5)
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            res_imgs.append(binary)
        return res_imgs

    def get_rect_area(self,rect):
        x, y, w, h = rect
        return w * h
    def get_rect_aspect_ratio(self, rect):
        x, y, w, h = rect
        return w / h

    def is_inside(self, rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # Check if all vertices of rect1 are inside rect2
        return x2 <= x1 and y2 <= y1 and x2 + w2 >= x1 + w1 and y2 + h2 >= y1 + h1
    
    def is_rect_away_from_border(self, rect, img, min_dist=10):
        h, w = img.shape[:2]
        x, y, rec_w, rec_h = rect
        return not(x < min_dist or x+rec_w > w - min_dist)

def compare(str1, str2):
    return sum(c1 == c2 for c1, c2 in zip(str1, str2))

def resize(img):
    h = img.shape[0]
    w = img.shape[1]
    scale = 1080 /w
    return cv2.resize(img, None, fx=scale, fy=scale)


def main():
    data = load_license_plates()
    epsilon = 0.03
    key = ord('a')
    i, j = 0, 0
    flag = True
    while key != ord('q'):
        src = data[i]['img']
        image_search = ImageSearch(src, epsilon, flag)
        if flag and image_search.number is not None:
            print("In")
            correct = compare(image_search.number, data[i]['num'])
            accuracy = (correct / len(data[i]['num']))*101.
            print(f"Accuracy: {accuracy:.2f}")
            print(image_search.number, " < -- found")
            print(data[i]['num'], " < -- correct")
        flag = False
        images = []
        h, w, _ = src.shape
        images.append(resize(image_search.img))
        #images.append(resize(image_search.binary))
        # images.append(resize(image_search.contour_img))
        # images.append(resize(image_search.filtered_contour_img))
        # images.append(resize(image_search.license_polygon_img))
        # images.append(resize(image_search.license_plate_img))
        images.append(resize(image_search.license_binary))
        images.append(resize(image_search.character_contours_img))

        cv2.imshow("<3", images[j])
        key = cv2.waitKey(50)
        if key == ord('h'):
            i = (i - 1) % len(data)
            print(i)
            flag = True
        elif key == ord('l'):
            i = (i + 1) % len(data)
            print(i)
            flag = True
        elif key == ord('j'):
            j = (j - 1) % len(images)
        elif key == ord('k'):
            j = (j + 1) % len(images)

if __name__ == "__main__":
    main()