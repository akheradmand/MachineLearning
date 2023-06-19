import cv2
import os

my_image = cv2.imread("numbers.jpg")
# print(my_image.shape)

number=0
count_j=1
for i in range(0,1000,20):
    for j in range(0,2000,20):
        
        crop_image=my_image[i:i+20,j:j+20]
        # cv2.imshow("",crop_image)
        # cv2.waitKey()
        os.makedirs(f"crop_images/{number}",exist_ok=True)
        cv2.imwrite(f"crop_images/{number}/number{number}_{count_j}.jpg",crop_image)
        count_j+=1

        if count_j>500:
            number+=1
            count_j=1