# main.py

import cv2
import numpy as np
from digit_recognition import load_trained_model, recognize_digit

def defineBiggestContour(contours):
    """
    Findet die größte Kontur, die wahrscheinlich das Sudoku-Raster ist.
    """
    biggest = np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000:  # Mindestfläche anpassen
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

def orderPoints(pts):
    """
    Ordnet die Punkte im Uhrzeigersinn: Top-left, Top-right, Bottom-right, Bottom-left.
    """
    pts = pts.reshape((4, 2))
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect

def main():
    img_path = "Resources/sudoku2.jpg"
    img_width, img_height = 810, 810  # Quadratische Größe

    # Bild vorverarbeiten
    img = cv2.resize(cv2.imread(img_path), (img_width, img_height))
    if img is None:
        print(f"Fehler: Bild unter {img_path} konnte nicht geladen werden.")
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_processed = cv2.adaptiveThreshold(
        img_blur,
        255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Konturen finden
    contours, _ = cv2.findContours(
        img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Größte Kontur finden (Sudoku-Raster)
    biggest_contour, biggest_area = defineBiggestContour(contours)
    if biggest_contour.size > 0:
        biggest_contour = orderPoints(biggest_contour)

        # Perspektive korrigieren (Warp Perspective)
        pts1 = np.float32(biggest_contour)
        pts2 = np.float32([
            [0, 0],                             # Top-left
            [img_width - 1, 0],                 # Top-right
            [img_width - 1, img_height - 1],    # Bottom-right
            [0, img_height - 1]                 # Bottom-left
        ])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_sudoku = cv2.warpPerspective(img, matrix, (img_width, img_height))
        img_sudoku_black = cv2.cvtColor(img_sudoku, cv2.COLOR_BGR2GRAY)

        # Sudoku in einzelne Zellen aufteilen
        all_fields = []
        rows = np.vsplit(img_sudoku_black, 9)
        for row in rows:
            columns = np.hsplit(row, 9)
            for cell in columns:
                all_fields.append(cell)

        # Laden des vortrainierten Modells
        model = load_trained_model(force_train=True)  # Erzwinge das Neutraining

        # Erkennen der Ziffern und Aufbau des Sudoku-Gitters
        sudoku_grid = []
        for idx, cell in enumerate(all_fields):
            digit = recognize_digit(cell, model)
            sudoku_grid.append(digit)
            # Debugging-Ausgabe, um zu überprüfen, ob die Ziffern erkannt werden
            print(f"Zelle {idx+1}: Erkanntes Zeichen: {digit}")

        # Konvertiere die flache Liste in eine 2D-Liste
        sudoku_grid_2d = [sudoku_grid[i*9:(i+1)*9] for i in range(9)]

        # Sudoku drucken und Bestätigung abfragen
        print("\nErkanntes Sudoku:")
        for row in sudoku_grid_2d:
            print(" ".join(str(num) if num != 0 else "." for num in row))

        confirmation = input("\nIst das erkannte Sudoku korrekt? (j/n): ")
        if confirmation.lower() != 'j':
            print("Sudoku wurde nicht bestätigt. Programm beendet.")
            return
        else:
            print("Sudoku wird gelöst...\n")

        # Sudoku-Löser-Funktionen
        def is_valid(board, row, col, num):
            for i in range(9):
                if board[row][i] == num or board[i][col] == num:
                    return False
            start_row, start_col = 3 * (row // 3), 3 * (col // 3)
            for i in range(3):
                for j in range(3):
                    if board[start_row + i][start_col + j] == num:
                        return False
            return True

        def solve_sudoku(board):
            for row in range(9):
                for col in range(9):
                    if board[row][col] == 0:
                        for num in range(1, 10):
                            if is_valid(board, row, col, num):
                                board[row][col] = num
                                if solve_sudoku(board):
                                    return True
                                board[row][col] = 0
                        return False
            return True

        # Kopiere das erkannte Sudoku-Gitter
        sudoku_solution = [row.copy() for row in sudoku_grid_2d]

        if solve_sudoku(sudoku_solution):
            # Ausgabe des gelösten Sudokus
            print("Gelöstes Sudoku:")
            for row in sudoku_solution:
                print(" ".join(str(num) for num in row))
        else:
            print("Keine Lösung gefunden.")
    else:
        print("Kein gültiges Sudoku-Raster gefunden.")

if __name__ == "__main__":
    main()
