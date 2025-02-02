import cv2
import os
import numpy as np
import warnings

from licensePlateOCR.patternMatch import pattern_match

# loads images data, returns a list of dict pairs with image array, plate number str
def loadImages(path):
    files = os.listdir(path)
    image_paths = list(filter(lambda fname: fname.endswith(".jpg"), files))   

    plate_numbers = list(map(lambda fname: fname.replace('.jpg', ''), image_paths))

    images = list(map(lambda fname: cv2.imread(path + fname), image_paths))

    data = [{'img': img, 'num': num, 'filename': os.path.basename(path)} for img, num, path in zip(images, plate_numbers, image_paths)]
    return data

class imageSearch:
    def __init__(self, img, epsilon, emergency_number):
        self.img = img.copy()
        self.number = None

        # Initial Filtering
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, self.binary = cv2.threshold(self.gray, 110, 255, cv2.THRESH_BINARY)
        self.binary = cv2.medianBlur(self.binary, 7)
        self.binary = cv2.morphologyEx(self.binary, cv2.MORPH_CLOSE, (3, 3), iterations=5) 

        # Get contours, filter
        self.contours= self.find_contours(self.binary)
        if not self.contours:
           warnings.warn("No plate found on an image")
           self.number = emergency_number
           return

        self.filtered_contours = self.filter_contours(self.contours, area=(280000, 1800000), ratio=(1.5, 5))
        if not self.filtered_contours:
           warnings.warn("No plate found on an image")
           self.number = emergency_number
           return

        # Approximate the contour
        self.license_polygon = self.approx_license_polygon(self.filtered_contours, epsilon=0.009)

        if self.license_polygon is None or len(self.license_polygon) == 0:
           warnings.warn("No plate found on an image")
           self.number = emergency_number
           return
        
        if len(self.license_polygon) != 4:
            warnings.warn("No plate found on an image (polygon vertices != 4)")
            self.number = emergency_number
            return

        # Warp license plate to a rectangle
        self.license_plate_img = self.warpImage(self.license_polygon, self.gray)

        ### Character segmentation
        # Initial filter
        _, self.license_binary = cv2.threshold(self.license_plate_img, 110, 255, cv2.THRESH_BINARY)
        self.license_binary = cv2.medianBlur(self.license_binary, 9)

        # Find contours, filter, expands roi by offset in all dims
        offset = 20
        self.license_binary = cv2.copyMakeBorder(self.license_binary, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=[255])

        character_contours = self.find_contours(self.license_binary)
        if not character_contours:
           warnings.warn("No plate found on an image")
           self.number = emergency_number
           return

        boundingRects = list(map(cv2.boundingRect, character_contours))

        image_area = self.license_binary.shape[0] * self.license_binary.shape[1]
        
        boundingRects = list(filter(lambda rect: 0.015*image_area < self.get_rect_area(rect) < 0.2*image_area , boundingRects))
        boundingRects = list(filter(lambda rect: 0.1 < self.get_rect_aspect_ratio(rect) < 0.99 , boundingRects))

        # Remove rectangles too left, right border
        boundingRects = list(filter(lambda rect: self.is_rect_away_from_border(rect, self.license_binary, min_dist=21), boundingRects))

        # Clean rectangles that are inside another rectangle
        boundingRects = list(filter(lambda rectA: not(
                any([self.is_inside(rectA, rectB) and (self.get_rect_area(rectA) < self.get_rect_area(rectB)) for rectB in boundingRects])), 
                boundingRects
        ))
        
        boundingRects = sorted(boundingRects, key=self.get_rect_area, reverse=True)
        boundingRects = boundingRects[:8]

        boundingRects = sorted(boundingRects, key=lambda rect: rect[0])  # Sort left to right

        # Expand border of each
        boundingRects = list(map(lambda rect: self.expandRect(rect, self.license_binary.shape, (20, 20, 20, 20)), boundingRects))

        if not boundingRects:
           warnings.warn("No characters found on the plate")
           self.number = emergency_number
           return

        if not(len(boundingRects) == 7 or len(boundingRects) == 8):
            warnings.warn("Found incorrect number of characters")
            self.number = emergency_number
            return 
        
        # Get each character image
        self.character_imgs = self.getCharacterImages(self.license_binary, boundingRects)

        #OCR
        match = pattern_match('./licensePlateOCR/assets/arklatrs-webfont.ttf', 500)
        text = "" 
        for i, img in enumerate(self.character_imgs):
            if i < 2:
                text += match.match(img, only_letters=True)
            else:
                text += match.match(img)

        self.number = text 

    # given rectangle, expands it by offset in all directions, keeping the rectangle in image bounds 
    def expandRect(self, rect, img_shape, offsets):
        img_h, img_w = img_shape[:2]
        x, y, w, h = rect
        x_off, y_off, w_off, h_off = offsets
        
        # Calculate the new top-left corner after applying offsets
        new_x = max(0, x - x_off)
        new_y = max(0, y - y_off)
        
        # Calculate the new width and height after applying offsets
        new_w = w + x_off + w_off
        new_h = h + y_off + h_off
        
        # Ensure the expanded rectangle fits within the image dimensions
        new_w = min(new_w, img_w - new_x)
        new_h = min(new_h, img_h - new_y)
        
        return (new_x, new_y, new_w, new_h)
 
    def find_contours(self, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        return contours

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
                
        return contours_filtered
         
    def approx_license_polygon(self, contours, epsilon):
        approx_contours = []

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

        return final_pts

    def warpImage(self, license_pts, gray):
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

    def getLicenseNumber(self):
        return self.number
    
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
        
def resize(img):
    h = img.shape[0]
    w = img.shape[1]
    scale = 1080 /w
    return cv2.resize(img, None, fx=scale, fy=scale)

def main():
    data = loadImages('./data/img')
    epsilon = 0.03
    key = ord('a')
    i, j = 0, 0
    flag = True
    while key != ord('q') and i < len(data):
        src = data[i]['img']
        image_search = imageSearch(src, epsilon)

        images = []
        h, w, _ = src.shape
        images.append(resize(image_search.img))
        
        key = cv2.waitKey(10)
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