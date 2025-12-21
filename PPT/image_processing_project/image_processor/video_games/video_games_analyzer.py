import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Generator, Tuple
import time
import os



class DataPipeline:
    """Система обработки данных видеоигр с использованием генераторов."""
    
    def __init__(self, csv_path: str, parquet_path: str = None, chunksize: int = 100):
        """
        Инициализация пайплайна.
        
        Args:
            csv_path (str): Путь к CSV файлу
            parquet_path (str): Путь к Parquet файлу
            chunksize (int): Размер чанка для чтения
        """
        self.csv_path = csv_path
        self.parquet_path = parquet_path or csv_path.replace('.csv', '.parquet')
        self.chunksize = chunksize
    
    def read_csv_generator(self) -> Generator[pd.DataFrame, None, None]:
        """Генератор для чтения CSV файла по частям."""
        print(f" Читаем CSV по частям (размер чанка: {self.chunksize} строк)")
        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunksize):
            yield chunk
    
    def filter_valid_data(self, data_generator: Generator) -> Generator[pd.DataFrame, None, None]:
        """Генератор фильтрации невалидных данных."""
        print(" Фильтруем невалидные данные (NaN значения)")
        for chunk in data_generator:
            filtered_chunk = chunk.dropna(subset=['Metrics.Review Score', 'Metrics.Sales'])
            if len(filtered_chunk) > 0:
                yield filtered_chunk
    
    # ========== ЗАДАНИЕ 1: Лучший/худший годы по продажам ==========
    def analyze_sales_by_year(self, data_generator: Generator) -> pd.DataFrame:
        """Агрегирует общие продажи по годам."""
        print("\n ЗАДАНИЕ 1: Агрегация продаж по годам")
        
        yearly_sales = {}
        
        for chunk in data_generator:
            grouped = chunk.groupby('Release.Year')['Metrics.Sales'].sum()
            
            for year, sales in grouped.items():
                if year in yearly_sales:
                    yearly_sales[year] += sales
                else:
                    yearly_sales[year] = sales
        
        result_df = pd.DataFrame(list(yearly_sales.items()), 
                                  columns=['Year', 'Total Sales'])
        result_df = result_df.sort_values('Year')
        
        best_year = result_df.loc[result_df['Total Sales'].idxmax()]
        worst_year = result_df.loc[result_df['Total Sales'].idxmin()]
        
        print(f" Лучший год: {int(best_year['Year'])} (${best_year['Total Sales']:.2f}M)")
        print(f" Худший год: {int(worst_year['Year'])} (${worst_year['Total Sales']:.2f}M)")
        
        return result_df
    
    # ========== ЗАДАНИЕ 2: Издатели с разбросом оценок ==========
    def analyze_publisher_variance(self, data_generator: Generator) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Вычисляет дисперсию и доверительные интервалы оценок для издателей."""
        print("\n ЗАДАНИЕ 2: Дисперсия оценок издателей (95% CI)")
        
        publisher_scores = {}
        
        for chunk in data_generator:
            for _, row in chunk.iterrows():
                publisher = row['Metadata.Publishers']
                score = row['Metrics.Review Score']
                
                if publisher not in publisher_scores:
                    publisher_scores[publisher] = []
                publisher_scores[publisher].append(score)
        
        stats_data = []
        for publisher, scores in publisher_scores.items():
            if len(scores) >= 2:  # Минимум 2 игры для расчета дисперсии
                scores_array = np.array(scores)
                mean = np.mean(scores_array)
                variance = np.var(scores_array, ddof=1)
                std = np.std(scores_array, ddof=1)
                n = len(scores_array)
                
                se = std / np.sqrt(n)
                ci = 1.96 * se
                
                stats_data.append({
                    'Publisher': publisher,
                    'Mean Score': mean,
                    'Variance': variance,
                    'Std Dev': std,
                    'CI Lower': mean - ci,
                    'CI Upper': mean + ci,
                    'Game Count': n
                })
        
        stats_df = pd.DataFrame(stats_data)
        stats_df = stats_df[stats_df['Game Count'] >= 5]  # Минимум 5 игр
        
        top3_variance = stats_df.nlargest(3, 'Variance')
        bottom3_variance = stats_df.nsmallest(3, 'Variance')
        
        print(f"\n ТОП-3 издателя с наибольшим разбросом:")
        for idx, row in top3_variance.iterrows():
            print(f"   {row['Publisher']}: σ²={row['Variance']:.2f}, σ={row['Std Dev']:.2f}")
        
        print(f"\n ТОП-3 издателя с наименьшим разбросом:")
        for idx, row in bottom3_variance.iterrows():
            print(f"   {row['Publisher']}: σ²={row['Variance']:.2f}, σ={row['Std Dev']:.2f}")
        
        combined_df = pd.concat([top3_variance, bottom3_variance])
        ci_df = combined_df[['Publisher', 'Mean Score', 'CI Lower', 'CI Upper', 'Variance']].copy()
        
        return stats_df, ci_df
    
    # ========== ЗАДАНИЕ 3: Игры по рейтингу E/T/M за каждый год ==========
    def count_games_by_rating(self, data_generator: Generator) -> pd.DataFrame:
        """Подсчитывает количество игр каждого рейтинга (E, T, M) по годам."""
        print("\n ЗАДАНИЕ 3: Количество игр по рейтингу (E/T/M) за каждый год")
        
        df_list = []

        for chunk in data_generator:
            filtered_chunk = chunk[chunk['Release.Rating'].isin(['E', 'T', 'M'])]
            grouped_df = filtered_chunk.groupby(['Release.Year', 'Release.Rating']).size().reset_index(name='Game Count')
            df_list.append(grouped_df)

        if df_list:
            result_df = pd.concat(df_list, ignore_index=True)
        else:
            result_df = pd.DataFrame(columns=['Release.Year', 'Release.Rating', 'Game Count'])

        # Приводим к единым именам столбцов
        result_df = result_df.rename(columns={'Release.Year': 'Year', 'Release.Rating': 'Rating'})

        # Агрегируем по годам и рейтингам
        result_df = result_df.groupby(['Year', 'Rating']).sum().reset_index()
        result_df = result_df.sort_values(['Year', 'Rating'])

        # Визуализация/информация
        pivot_df = result_df.pivot(index='Year', columns='Rating', values='Game Count').fillna(0)
        print(f"\n Распределение игр по рейтингам:\n{pivot_df}")

        return result_df



class ParquetManager:
    """Управление конвертацией и работой с Parquet файлами."""
    
    def __init__(self, csv_path: str, parquet_path: str = None):
        """Инициализация менеджера Parquet."""
        self.csv_path = csv_path
        self.parquet_path = parquet_path or csv_path.replace('.csv', '.parquet')
    
    def csv_to_parquet(self) -> None:
        """Конвертирует CSV в Parquet (если не существует)."""
        if os.path.exists(self.parquet_path):
            print(f" Parquet файл уже существует: {self.parquet_path}")
            return
        
        print(f" Конвертируем CSV в Parquet...")
        df = pd.read_csv(self.csv_path)
        df.to_parquet(self.parquet_path, engine='pyarrow', compression='snappy')
        print(f" Parquet файл создан: {self.parquet_path}")
    
    def compare_read_speed(self) -> dict:
        """Сравнивает скорость чтения CSV vs Parquet."""
        print("\n Сравнение скорости чтения...")
        
        start_csv = time.time()
        df_csv = pd.read_csv(self.csv_path)
        time_csv = time.time() - start_csv
        
        start_parquet = time.time()
        df_parquet = pd.read_parquet(self.parquet_path, engine='pyarrow')
        time_parquet = time.time() - start_parquet
        
        results = {
            'CSV Time': time_csv,
            'Parquet Time': time_parquet,
            'Speedup': time_csv / time_parquet
        }
        
        print(f"   CSV:      {time_csv:.4f} сек")
        print(f"   Parquet:  {time_parquet:.4f} сек")
        print(f"   Ускорение: {results['Speedup']:.2f}x")
        
        return results
    
    def get_parquet_subset(self, columns: list) -> pd.DataFrame:
        """Читает только нужные столбцы из Parquet (оптимизация)."""
        print(f" Читаем выбранные столбцы из Parquet: {columns}")
        return pd.read_parquet(self.parquet_path, columns=columns, engine='pyarrow')
    
    def calculate_correlation(self) -> pd.DataFrame:
        """ДОП. ЗАДАНИЕ: Корреляция между Review Score и Sales."""
        print("\n ДОП. ЗАДАНИЕ: Корреляция Score vs Sales")
        
        df = pd.read_parquet(
            self.parquet_path, 
            columns=['Metrics.Review Score', 'Metrics.Sales', 'Release.Year'],
            engine='pyarrow'
        )
        
        df = df.dropna(subset=['Metrics.Review Score', 'Metrics.Sales'])
        
        overall_corr = df['Metrics.Review Score'].corr(df['Metrics.Sales'])
        print(f" Общая корреляция: {overall_corr:.4f}")
        
        yearly_corr = df.groupby('Release.Year').apply(
            lambda x: x['Metrics.Review Score'].corr(x['Metrics.Sales'])
        )
        
        result_df = pd.DataFrame({
            'Year': yearly_corr.index,
            'Correlation': yearly_corr.values
        })
        
        print(f"\n Корреляция по годам:\n{result_df}")
        
        return result_df


class DataVisualizer:
    """Визуализация данных."""
    
    @staticmethod
    def plot_sales_by_year(df: pd.DataFrame, output_path: str = None):
        """ГРАФИК 1: Bar plot для продаж по годам."""
        plt.figure(figsize=(12, 6))
        
        plt.bar(df['Year'], df['Total Sales'], color='steelblue', alpha=0.8, edgecolor='black')
        
        best_idx = df['Total Sales'].idxmax()
        worst_idx = df['Total Sales'].idxmin()
        
        plt.bar(df.loc[best_idx, 'Year'], df.loc[best_idx, 'Total Sales'], 
                color='green', alpha=0.9, label='Лучший год')
        plt.bar(df.loc[worst_idx, 'Year'], df.loc[worst_idx, 'Total Sales'], 
                color='red', alpha=0.9, label='Худший год')
        
        plt.xlabel('Год', fontsize=12, fontweight='bold')
        plt.ylabel('Суммарные продажи ($ млн)', fontsize=12, fontweight='bold')
        plt.title('Продажи видеоигр по годам', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f" График сохранен: {output_path}")
        plt.show()
    
    @staticmethod
    def plot_publisher_variance(ci_df: pd.DataFrame, output_path: str = None):
        """ГРАФИК 2: Bar plot с доверительными интервалами для издателей."""
        plt.figure(figsize=(14, 7))
        
        x = range(len(ci_df))
        y = ci_df['Mean Score'].values
        yerr = [
            (ci_df['Mean Score'] - ci_df['CI Lower']).values,
            (ci_df['CI Upper'] - ci_df['Mean Score']).values
        ]
        
        colors = ['#FF6B6B'] * 3 + ['#4ECDC4'] * 3
        
        plt.bar(x, y, yerr=yerr, capsize=8, alpha=0.75, color=colors, 
                error_kw={'linewidth': 2.5}, edgecolor='black', linewidth=1.5)
        
        plt.axhline(y=ci_df['Mean Score'].mean(), color='gray', linestyle='--', 
                    linewidth=2, alpha=0.7, label='Среднее')
        
        plt.xlabel('Издатель', fontsize=12, fontweight='bold')
        plt.ylabel('Средний Review Score', fontsize=12, fontweight='bold')
        plt.title('Топ-3 издателя с наибольшим и наименьшим разбросом оценок\n(с 95% доверительными интервалами)', 
                  fontsize=14, fontweight='bold')
        plt.xticks(x, ci_df['Publisher'].values, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f" График сохранен: {output_path}")
        plt.show()
    
    @staticmethod
    def plot_rating_trends(df: pd.DataFrame, output_path: str = None):
        """ГРАФИК 3: Line plot для количества игр по рейтингам E/T/M."""
        plt.figure(figsize=(12, 6))
        
        for rating in ['E', 'T', 'M']:
            rating_data = df[df['Rating'] == rating]
            plt.plot(rating_data['Year'], rating_data['Game Count'], 
                     'o-', label=f'Rating {rating}', linewidth=2.5, markersize=8)
        
        plt.xlabel('Год', fontsize=12, fontweight='bold')
        plt.ylabel('Количество игр', fontsize=12, fontweight='bold')
        plt.title('Количество выпущенных игр по возрастному рейтингу (E/T/M)', 
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f" График сохранен: {output_path}")
        plt.show()
    
    @staticmethod
    def plot_correlation_scatter(corr_df: pd.DataFrame, output_path: str = None):
        """ГРАФИК 4 (ДОП.): Scatter plot для корреляции Score vs Sales."""
        plt.figure(figsize=(12, 6))
        
        plt.scatter(corr_df['Year'], corr_df['Correlation'], 
                    s=200, alpha=0.7, c=corr_df['Correlation'], 
                    cmap='coolwarm', edgecolors='black', linewidth=2)
        
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
        
        plt.colorbar(label='Корреляция')
        plt.xlabel('Год', fontsize=12, fontweight='bold')
        plt.ylabel('Корреляция (Score vs Sales)', fontsize=12, fontweight='bold')
        plt.title('Корреляция между оценкой игры и продажами по годам', 
                  fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f" График сохранен: {output_path}")
        plt.show()
    
    @staticmethod
    def plot_speed_comparison(speeds: dict, output_path: str = None):
        """ГРАФИК 5: Сравнение скорости CSV vs Parquet."""
        plt.figure(figsize=(10, 6))
        
        formats = ['CSV', 'Parquet']
        times = [speeds['CSV Time'], speeds['Parquet Time']]
        colors = ['#FF6B6B', '#4ECDC4']
        
        plt.scatter(formats, times, s=1200, c=colors, alpha=0.7, 
                    edgecolors='black', linewidth=3)
        
        for i, (fmt, time_val) in enumerate(zip(formats, times)):
            plt.text(i, time_val + max(times)*0.03, f'{time_val:.4f}s', 
                     ha='center', fontsize=13, fontweight='bold')
        
        plt.ylabel('Время чтения (секунды)', fontsize=12, fontweight='bold')
        plt.title(f'Сравнение скорости чтения (Ускорение: {speeds["Speedup"]:.2f}x)', 
                  fontsize=14, fontweight='bold')
        plt.ylim(0, max(times) * 1.25)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f" График сохранен: {output_path}")
        plt.show()



# Создание папки results если ее нет
from pathlib import Path
Path("results/lab3").mkdir(parents=True, exist_ok=True)


