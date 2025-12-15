import cv2
import numpy as np

# Putanja do slike sa parazitom
img = cv2.imread("cell_images/Parasitized/C39P4thinF_original_IMG_20150622_105253_cell_101.png")
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Klikom misa prikazujemo LAB vrednosti piksel-a
def show_lab(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        l_val, a_val, b_val = lab[y, x]
        print(f"Kliknuto na: ({x},{y}) → L={l_val}, A={a_val}, B={b_val}")

cv2.namedWindow("Klikni na parazita")
cv2.setMouseCallback("Klikni na parazita", show_lab)

while True:
    cv2.imshow("Klikni na parazita", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc za izlaz
        break

cv2.destroyAllWindows()
