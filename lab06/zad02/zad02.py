import cv2
import numpy as np

for i in range(0, 14):      
    image = cv2.imread(f'c:/Users/maria/Desktop/ug/sem4/IntelegencjaObliczeniowa/lab06/zad02/obrazy/obraz{i}.jpg')
    naive_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    better_gray = np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)



    combined_image = np.concatenate((image, cv2.cvtColor(naive_gray, cv2.COLOR_GRAY2BGR), cv2.cvtColor(better_gray, cv2.COLOR_GRAY2BGR)), axis=1)

    cv2.putText(combined_image, 'Original', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,  2)
    cv2.putText(combined_image, 'Naive Grayscale', (image.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,  2)
    cv2.putText(combined_image, 'Better Grayscale', (image.shape[1] * 2 + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    cv2.imwrite(f'porownanie/{i}.jpg', combined_image)

    cv2.waitKey(0)
cv2.destroyAllWindows()