import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#  LAB pragovi (fino podeseni)
L_MIN, L_MAX = 20, 145  # osvetljenost: ignorisemo previse tamno ili svetlo
A_MIN, A_MAX = 140, 190  # crvenkasta komponenta
B_MIN, B_MAX = 70, 160  # plavkasta komponenta
AREA_THRESH = 15  # broj piksela ljubicaste da bi bilo "zarazeno"


def is_infected(img_bgr):
    # Konverzija u LAB prostor
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Segmentacija po sva 3 kanala (L, A, B)
    mask = (
            cv2.inRange(l, L_MIN, L_MAX) &
            cv2.inRange(a, A_MIN, A_MAX) &
            cv2.inRange(b, B_MIN, B_MAX)
    )

    # Morfolosko ciscenje suma
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Brojanje "ljubicastih" piksela
    purple_area = cv2.countNonZero(mask)
    return purple_area > AREA_THRESH, mask

def show_processing_steps(img_path):
    #  1. Ucitavanje slike
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    #  2. Konverzija u LAB prostor
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Normalizacija radi lepseg prikaza (LAB je 0–255, ali ne linearno za prikaz)
    l_norm = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
    a_norm = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
    b_norm = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)

    # 3. Segmentacija po pragovima
    mask = (
        cv2.inRange(l, L_MIN, L_MAX) &
        cv2.inRange(a, A_MIN, A_MAX) &
        cv2.inRange(b, B_MIN, B_MAX)
    )

    # Morfolosko ciscenje
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)

    # 4. Obelazavanje detekcije (konture na originalu)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_detected = img_rgb.copy()
    cv2.drawContours(img_detected, contours, -1, (255, 0, 0), 1)  # plave konture

    # 5. Prikaz svih faza obrade
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    axs[0, 0].imshow(img_rgb)
    axs[0, 0].set_title("Originalna slika")

    axs[0, 1].imshow(l_norm, cmap='gray')
    axs[0, 1].set_title("L kanal (osvetljenost)")

    axs[0, 2].imshow(a_norm, cmap='gray')
    axs[0, 2].set_title("A kanal (zeleno–crveno)")

    axs[1, 0].imshow(b_norm, cmap='gray')
    axs[1, 0].set_title("B kanal (plavo–žuto)")

    axs[1, 1].imshow(mask_clean, cmap="gray")
    axs[1, 1].set_title("Binarna maska")

    axs[1, 2].imshow(img_detected)
    axs[1, 2].set_title("Obeležena detekcija")

    for ax in axs.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def evaluate(dataset_root="cell_images"):
    parasitized_dir = Path(dataset_root) / "Parasitized"
    uninfected_dir = Path(dataset_root) / "Uninfected"

    tp = fp = tn = fn = 0
    false_positive_example = None

    # Zarazene slike
    for img_path in parasitized_dir.glob("*.png"):
        img = cv2.imread(str(img_path))
        pred, _ = is_infected(img)
        if pred:
            tp += 1
        else:
            fn += 1

    # Nezarazene slike
    for img_path in uninfected_dir.glob("*.png"):
        img = cv2.imread(str(img_path))
        pred, mask = is_infected(img)
        if pred:
            fp += 1
            if false_positive_example is None:
                false_positive_example = (img, mask, img_path.name)
        else:
            tn += 1

    # Metricne vrednosti
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)

    print(f"Tačnost:       {acc:.3f}")
    print(f"Sensitivnost:  {sens:.3f}")
    print(f"Specifičnost:  {spec:.3f}")
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # Matrica konfuzije
    cm = np.array([[tn, fp],
                   [fn, tp]])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negativno (nezaraženo)", "Pozitivno (zaraženo)"]
    )
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.xlabel("Predikcija modela")
    plt.ylabel("Stvarna klasa")
    plt.title("Matrica konfuzije (detekcija malarije)")
    plt.tight_layout()
    plt.show()

    # Prikaz jedne lazno pozitivne slike
    if false_positive_example:
        img_fp, mask_fp, name_fp = false_positive_example
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(cv2.cvtColor(img_fp, cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"Lažno pozitivna slika\n{name_fp}")
        print({name_fp})
        axs[0].axis("off")
        axs[1].imshow(mask_fp, cmap="gray")
        axs[1].set_title("Maska ljubičaste regije")
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    evaluate("cell_images")  # Postavi putanju ako ti je drugacije
    example_img = Path("cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_162.png")
    show_processing_steps(example_img)