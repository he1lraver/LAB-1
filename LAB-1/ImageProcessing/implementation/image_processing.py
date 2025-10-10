"""
Модуль image_processing.py

Улучшенная реализация интерфейса IImageProcessing.
"""

import time
import math
import numpy as np
from scipy.ndimage import convolve
from typing import Tuple


class ImageProcessing:
    """
    Реализация обработки изображений.
    """

    def __init__(self):
        self.optimization_enabled = True

    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Оптимизированная свёртка с использованием scipy.convolve.
        """
        start_time = time.time()
        
        result = convolve(image.astype(np.float32), kernel, mode='constant', cval=0.0)
        
        end_time = time.time()
        print(f"Свёртка выполнена за {end_time - start_time:.4f} секунд")
        
        return result

    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Преобразование в grayscale.
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Изображение должно быть RGB")
        
        return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.
        """
        start_time = time.time()
        
        if gamma <= 0:
            raise ValueError("Гамма должна быть положительным числом")
        
        image_normalized = image.astype(np.float32) / 255.0
        corrected_image = np.power(image_normalized, gamma)
        result = (corrected_image * 255).astype(np.uint8)
        
        end_time = time.time()
        print(f"Гамма-коррекция выполнена за {end_time - start_time:.4f} секунд")
        
        return result

    def _sobel_operators(self) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает операторы Собеля."""
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        return sobel_x, sobel_y

    def _non_maximum_suppression(self, magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """
        Подавление немаксимумов для тонких границ.
        """
        height, width = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        
        direction = direction * (180.0 / np.pi)
        direction[direction < 0] += 180
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                angle = direction[i, j]
                
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                elif 22.5 <= angle < 67.5:
                    neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
                elif 67.5 <= angle < 112.5:
                    neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                else:
                    neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                
                if magnitude[i, j] >= max(neighbors):
                    suppressed[i, j] = magnitude[i, j]
        
        return suppressed

    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Обнаружение границ с улучшенным алгоритмом Кэнни.
        """
        start_time = time.time()
        
        gray = self._rgb_to_grayscale(image)
        
        # Вычисляем градиенты Собеля
        sobel_x, sobel_y = self._sobel_operators()
        grad_x = self._convolution(gray.astype(np.float32), sobel_x)
        grad_y = self._convolution(gray.astype(np.float32), sobel_y)
        
        # Величина и направление градиента
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Подавление немаксимумов
        suppressed = self._non_maximum_suppression(magnitude, direction)
        
        # Простой порог
        threshold = 0.1 * np.max(suppressed)
        edges = (suppressed > threshold).astype(np.uint8) * 255
        
        end_time = time.time()
        print(f"Обнаружение границ выполнено за {end_time - start_time:.4f} секунд")
        
        return edges

    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Детектор углов на основе алгоритма Харриса.
        Углы отображаются красным цветом.
        """
        start_time = time.time()
        
        if len(image.shape) == 3:
            gray = self._rgb_to_grayscale(image)
        else:
            gray = image
        
        # Вычисляем градиенты
        sobel_x, sobel_y = self._sobel_operators()
        Ix = self._convolution(gray.astype(np.float32), sobel_x)
        Iy = self._convolution(gray.astype(np.float32), sobel_y)
        
        # Вычисляем элементы матрицы структуры
        Ix2 = Ix * Ix
        Iy2 = Iy * Iy
        Ixy = Ix * Iy
        
        # Усредняем с помощью гауссова окна
        window = self._gaussian_kernel(5, 1.5)
        Sx2 = self._convolution(Ix2, window)
        Sy2 = self._convolution(Iy2, window)
        Sxy = self._convolution(Ixy, window)
        
        # Вычисляем меру угла
        k = 0.1
        det_M = Sx2 * Sy2 - Sxy * Sxy
        trace_M = Sx2 + Sy2
        R = det_M - k * (trace_M ** 2)
        
        # Нормализация и порог
        R_positive = np.maximum(R, 0)
        if R_positive.max() > 0:
            R_normalized = R_positive / R_positive.max()
        else:
            R_normalized = R_positive
        
        mean_response = np.mean(R_normalized)
        std_response = np.std(R_normalized)
        corner_threshold = mean_response + 1.5 * std_response
        
        # Подавление немаксимумов
        corners_binary = self._non_maximum_suppression_corners(R_normalized, threshold=corner_threshold)
        
        # Рисуем углы красным цветом
        result_image = image.copy()
        corner_coords = np.argwhere(corners_binary)
        
        for y, x in corner_coords:
            result_image = self._draw_red_corner(result_image, x, y)
        
        end_time = time.time()
        print(f"Обнаружение углов выполнено за {end_time - start_time:.4f} секунд")
        print(f"Найдено углов: {len(corner_coords)}")
        
        return result_image

    def _non_maximum_suppression_corners(self, corner_response: np.ndarray, 
                                       threshold: float = 0.1) -> np.ndarray:
        """
        Подавление немаксимумов для углов.
        """
        height, width = corner_response.shape
        is_corner = np.zeros_like(corner_response, dtype=bool)
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                current_val = corner_response[i, j]
                
                if current_val < threshold:
                    continue
                
                # Проверяем локальный максимум 3x3
                is_local_max = True
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        if corner_response[i + di, j + dj] > current_val:
                            is_local_max = False
                            break
                    if not is_local_max:
                        break
                
                if is_local_max:
                    is_corner[i, j] = True
        
        return is_corner

    def _draw_red_corner(self, image: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        Рисует красный маркер угла на изображении.
        """
        result = image.copy()
        height, width = image.shape[:2]
        
        red_color = (0, 0, 255)  # Красный цвет в BGR
        size = 3
        
        # Рисуем крестик
        for i in range(-size, size + 1):
            # Горизонтальная линия
            px = x + i
            py = y
            if 0 <= px < width and 0 <= py < height:
                result[py, px] = red_color
            
            # Вертикальная линия
            px = x
            py = y + i
            if 0 <= px < width and 0 <= py < height:
                result[py, px] = red_color
        
        return result

    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Обнаружение окружностей с помощью преобразования Хафа.
        """
        start_time = time.time()
        
        gray = self._rgb_to_grayscale(image)
        
        # Обнаружение границ
        edges = self.edge_detection(image)
        edges_binary = (edges > 128)
        
        height, width = gray.shape
        
        # Параметры окружностей
        min_radius = max(10, min(height, width) // 30)
        max_radius = min(120, min(height, width) // 4)
        
        # Собираем точки границ
        edge_points = np.argwhere(edges_binary)
        
        # Ограничиваем количество точек для производительности
        if len(edge_points) > 5000:
            step = len(edge_points) // 5000
            edge_points = edge_points[::step]
        
        # Создаем аккумулятор
        accumulator = np.zeros((height, width, max_radius - min_radius + 1), dtype=np.uint16)
        
        # Голосование в пространстве Хафа
        for y, x in edge_points:
            for r in range(min_radius, max_radius + 1):
                for angle in range(0, 360, 3):
                    rad = math.radians(angle)
                    a = int(x + r * math.cos(rad))
                    b = int(y + r * math.sin(rad))
                    
                    if 0 <= a < width and 0 <= b < height:
                        accumulator[b, a, r - min_radius] += 1
        
        # Поиск кандидатов
        circles = []
        vote_threshold = 0.17 * accumulator.max()
        
        for r_idx, r in enumerate(range(min_radius, max_radius + 1)):
            for y in range(height):
                for x in range(width):
                    if accumulator[y, x, r_idx] > vote_threshold:
                        circles.append((x, y, r, accumulator[y, x, r_idx]))
        
        # Сортировка и фильтрация
        circles.sort(key=lambda x: x[3], reverse=True)
        final_circles = []
        
        for x, y, r, score in circles:
            duplicate = False
            for existing in final_circles:
                x2, y2, r2 = existing
                distance = math.sqrt((x - x2)**2 + (y - y2)**2)
                if distance < 30 and abs(r - r2) < max(5, r * 0.3):
                    duplicate = True
                    break
            
            if not duplicate:
                final_circles.append((x, y, r))
            
            if len(final_circles) >= 30:
                break
        
        # Рисуем результат
        result_image = image.copy()
        for x, y, r in final_circles:
            result_image = self._draw_circle(result_image, x, y, r)
        
        end_time = time.time()
        print(f"Обнаружение окружностей выполнено за {end_time - start_time:.4f} секунд")
        print(f"Найдено окружностей: {len(final_circles)}")
        
        return result_image

    def _draw_circle(self, image: np.ndarray, center_x: int, center_y: int, 
                    radius: int) -> np.ndarray:
        """Рисует окружность на изображении."""
        result = image.copy()
        height, width = image.shape[:2]
        
        green_color = (0, 255, 0)  # Зеленый цвет
        
        # Рисуем контур
        for angle in np.linspace(0, 2 * np.pi, 100):
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            
            if 0 <= x < width and 0 <= y < height:
                result[y, x] = green_color
        
        return result

    def _gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Создает гауссово ядро."""
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
        return kernel / np.sum(kernel)

    def convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Публичный метод свёртки."""
        return self._convolution(image, kernel)

    def rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Публичный метод преобразования в grayscale."""
        return self._rgb_to_grayscale(image)

    def gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Публичный метод гамма-коррекции."""
        return self._gamma_correction(image, gamma)