"""
Модуль image_processing.py

Улучшенная реализация интерфейса IImageProcessing.
"""

import time
import math, cv2
import numpy as np
from interfaces.i_image_processing import IImageProcessing


class ImageProcessing(IImageProcessing):
    """
    Улучшенная реализация интерфейса IImageProcessing.
    """

    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Оптимизированная свёртка изображения с заданным ядром.
        """
        start_time = time.time()
        
        image_height, image_width = image.shape[:2]
        kernel_height, kernel_width = kernel.shape
        
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        
        # Добавляем padding
        if len(image.shape) == 3:
            padded_image = np.pad(image, 
                                ((pad_height, pad_height), 
                                 (pad_width, pad_width), 
                                 (0, 0)), 
                                mode='edge')  # Используем edge padding вместо constant
        else:
            padded_image = np.pad(image, 
                                ((pad_height, pad_height), 
                                 (pad_width, pad_width)), 
                                mode='edge')
        
        output = np.zeros_like(image, dtype=np.float32)
        
        # Векторизованная версия для улучшения производительности
        for i in range(image_height):
            for j in range(image_width):
                region = padded_image[i:i + kernel_height, j:j + kernel_width]
                if len(image.shape) == 3:
                    output[i, j] = np.sum(region * kernel[:, :, np.newaxis], axis=(0, 1))
                else:
                    output[i, j] = np.sum(region * kernel)
        
        # Нормализация с сохранением типа данных
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        end_time = time.time()
        print(f"Свёртка выполнена за {end_time - start_time:.4f} секунд")
        
        return output

    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого.
        """
        start_time = time.time()
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Изображение должно быть RGB")
        
        # Используем формулу luminance
        grayscale = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        
        end_time = time.time()
        print(f"Преобразование в grayscale выполнено за {end_time - start_time:.4f} секунд")
        
        return grayscale

    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.
        """
        start_time = time.time()
        
        if gamma <= 0:
            raise ValueError("Гамма должна быть положительным числом")
        
        # Нормализуем и применяем гамма-коррекцию
        image_normalized = image.astype(np.float32) / 255.0
        corrected_image = np.power(image_normalized, gamma)
        result = (corrected_image * 255).astype(np.uint8)
        
        end_time = time.time()
        print(f"Гамма-коррекция выполнена за {end_time - start_time:.4f} секунд")
        
        return result

    def _gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Создает гауссово ядро."""
        kernel = np.fromfunction(
            lambda x, y: (1/(2*np.pi*sigma**2)) * 
                        np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2) / (2*sigma**2)),
            (size, size)
        )
        return kernel / np.sum(kernel)

    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Улучшенное обнаружение границ с помощью оператора Собеля.
        """
        start_time = time.time()
        
        # Преобразуем в grayscale
        gray = self._rgb_to_grayscale(image)
        
        # Ядра Собеля
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Применяем свёртки
        grad_x = self._convolution(gray.astype(np.float32), sobel_x)
        grad_y = self._convolution(gray.astype(np.float32), sobel_y)
        
        # Вычисляем величину градиента и направление
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        # Нормализуем результат
        if gradient_magnitude.max() > 0:
            edges = (gradient_magnitude * 255 / gradient_magnitude.max()).astype(np.uint8)
        else:
            edges = gradient_magnitude.astype(np.uint8)
        
        # Применяем пороговую обработку для лучшего выделения границ
        threshold = 0.3 * edges.max()
        edges[edges < threshold] = 0
        edges[edges >= threshold] = 255
        
        end_time = time.time()
        print(f"Обнаружение границ выполнено за {end_time - start_time:.4f} секунд")
        
        return edges

    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Улучшенный детектор углов Харриса.
        """
        start_time = time.time()
        
        gray = self._rgb_to_grayscale(image)
        gray_float = gray.astype(np.float32)
        
        # Вычисляем производные
        Ix = self._convolution(gray_float, np.array([[-1, 0, 1]]))
        Iy = self._convolution(gray_float, np.array([[-1], [0], [1]]))
        
        # Вычисляем элементы матрицы структуры
        Ix2 = Ix * Ix
        Iy2 = Iy * Iy
        Ixy = Ix * Iy
        
        # Применяем гауссово размытие
        gaussian_kernel = self._gaussian_kernel(3, 1.0)
        Sx2 = self._convolution(Ix2, gaussian_kernel)
        Sy2 = self._convolution(Iy2, gaussian_kernel)
        Sxy = self._convolution(Ixy, gaussian_kernel)
        
        # Параметры детектора
        k = 0.001
        height, width = gray.shape
        
        # Вычисляем угол ответа
        corner_response = np.zeros((height, width))
        
        for i in range(height):
            for j in range(width):
                M = np.array([[Sx2[i, j], Sxy[i, j]],
                             [Sxy[i, j], Sy2[i, j]]])
                
                det = np.linalg.det(M)
                trace = np.trace(M)
                R = det - k * (trace ** 2)
                corner_response[i, j] = R
        
        # Нормализуем и применяем non-maximum suppression
        corner_response = (corner_response - corner_response.min()) / (corner_response.max() - corner_response.min())
        
        # Находим локальные максимумы
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(corner_response, size=3) == corner_response
        corner_response[~local_max] = 0
        
        # Пороговая обработка
        threshold = 0.01
        corners = corner_response > threshold
        
        # Отмечаем углы на изображении
        result_image = image.copy()
        y_coords, x_coords = np.where(corners)
        
        for y, x in zip(y_coords, x_coords):
            # Рисуем маленькие кружки вместо отдельных пикселей
            cv2.circle(result_image, (x, y), 3, (0, 0, 255), -1)
        
        end_time = time.time()
        print(f"Обнаружение углов выполнено за {end_time - start_time:.4f} секунд")
        print(f"Найдено углов: {len(y_coords)}")
        
        return result_image

    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Оптимизированное обнаружение окружностей с помощью преобразования Хафа.
        """
        start_time = time.time()
        
        gray = self._rgb_to_grayscale(image)
        
        # Обнаруживаем границы с помощью оператора Собеля
        edges = self.edge_detection(image)
        
        # Бинаризуем границы
        edges_binary = (edges > 128).astype(np.uint8)
        
        # Параметры окружностей
        min_radius = 10
        max_radius = min(image.shape[:2]) // 4
        result_image = image.copy()
        
        # Используем градиентную информацию для оптимизации
        sobel_x = self._convolution(gray.astype(np.float32), np.array([[-1, 0, 1]]))
        sobel_y = self._convolution(gray.astype(np.float32), np.array([[-1], [0], [1]]))
        
        height, width = gray.shape
        circles = []
        
        # Упрощенный вариант преобразования Хафа с градиентной информацией
        edge_points = np.argwhere(edges_binary > 0)
        
        # Ограничиваем количество точек для производительности
        if len(edge_points) > 1000:
            indices = np.random.choice(len(edge_points), 1000, replace=False)
            edge_points = edge_points[indices]
        
        for point in edge_points:
            y, x = point
            
            # Используем направление градиента
            if sobel_x[y, x] != 0 or sobel_y[y, x] != 0:
                gradient_dir = math.atan2(sobel_y[y, x], sobel_x[y, x])
                
                # Проверяем радиусы в направлении, перпендикулярном градиенту
                for r in range(min_radius, max_radius + 1, 2):  # Шаг 2 для производительности
                    # Центр вдоль направления, перпендикулярного градиенту
                    center_x1 = int(x + r * math.cos(gradient_dir + math.pi/2))
                    center_y1 = int(y + r * math.sin(gradient_dir + math.pi/2))
                    center_x2 = int(x + r * math.cos(gradient_dir - math.pi/2))
                    center_y2 = int(y + r * math.sin(gradient_dir - math.pi/2))
                    
                    for cx, cy in [(center_x1, center_y1), (center_x2, center_y2)]:
                        if 0 <= cx < width and 0 <= cy < height:
                            # Проверяем, достаточно ли edge points на предполагаемой окружности
                            circle_points = 0
                            for angle in range(0, 360, 30):  # Проверяем каждые 30 градусов
                                check_x = int(cx + r * math.cos(math.radians(angle)))
                                check_y = int(cy + r * math.sin(math.radians(angle)))
                                if (0 <= check_x < width and 0 <= check_y < height and 
                                    edges_binary[check_y, check_x] > 0):
                                    circle_points += 1
                            
                            if circle_points >= 6:  # Минимум 6 точек на окружности
                                circles.append((cx, cy, r))
        
        # Убираем дубликаты
        unique_circles = []
        for circle in circles:
            is_duplicate = False
            for uc in unique_circles:
                if (abs(circle[0] - uc[0]) < 10 and 
                    abs(circle[1] - uc[1]) < 10 and 
                    abs(circle[2] - uc[2]) < 10):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_circles.append(circle)
        
        # Рисуем окружности
        for x, y, r in unique_circles:
            cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result_image, (x, y), 2, (255, 0, 0), 3)
        
        end_time = time.time()
        print(f"Обнаружение окружностей выполнено за {end_time - start_time:.4f} секунд")
        print(f"Найдено окружностей: {len(unique_circles)}")
        
        return result_image