import polars as pl
import os
import shutil
from pathlib import Path
import yaml

# Список всех моделей
models = [
    "lin_cafa5_esm2S1280_cafa6-sparse-labels13500_cond_v2",
    "lin_cafa6_t5_cafa6-sparse-labels13500_raw_v2",
    "pb_cafa5_t5esm2S1280_cafa6-sparse-labels4500_cond_v2",
    "lin_cafa5_esm2S1280_cafa6-sparse-labels13500_raw_v2",
    "lin_cafa6_t5esm2S1280_cafa6-sparse-labels13500_cond_v2",
    "pb_cafa5_t5esm2S1280_cafa6-sparse-labels4500_raw_v2",
    "lin_cafa5_t5_cafa6-sparse-labels13500_cond_v2",
    "lin_cafa6_t5esm2S1280_cafa6-sparse-labels13500_raw_v2",
    "pb_cafa6_esm2S1280_cafa6-sparse-labels4500_cond_v2",
    "lin_cafa5_t5_cafa6-sparse-labels13500_raw_v2",
    "pb_cafa6_esm2S1280_cafa6-sparse-labels4500_raw_v2",
    "lin_cafa5_t5esm2S1280_cafa6-sparse-labels13500_cond_v2",
    "pb_cafa5_esm2S1280_cafa6-sparse-labels4500_cond_v2",
    "pb_cafa6_t5_cafa6-sparse-labels4500_raw_v2",
    "lin_cafa5_t5esm2S1280_cafa6-sparse-labels13500_raw_v2",
    "pb_cafa5_esm2S1280_cafa6-sparse-labels4500_raw_v2",
    "pb_cafa6_t5esm2S1280_cafa6-sparse-labels4500_cond_v2",
    "lin_cafa6_esm2S1280_cafa6-sparse-labels13500_cond_v2",
    "pb_cafa5_t5_cafa6-sparse-labels4500_cond_v2",
    "pb_cafa6_t5esm2S1280_cafa6-sparse-labels4500_raw_v2",
    "lin_cafa6_esm2S1280_cafa6-sparse-labels13500_raw_v2",
    "pb_cafa5_t5_cafa6-sparse-labels4500_raw_v2",
    "lin_cafa6_t5_cafa6-sparse-labels13500_cond_v2",
    "pb_cafa6_t5_cafa6-sparse-labels4500_cond_v2"
]

# Определение префиксов и суффиксов
prefixes = ['lin', 'pb']
suffixes = ['13500_cond_v2', '13500_raw_v2', '4500_cond_v2', '4500_raw_v2']

def get_model_info(model_name):
    """Извлекает префикс и суффикс из названия модели"""
    for prefix in prefixes:
        if model_name.startswith(prefix):
            for suffix in suffixes:
                if model_name.endswith(suffix):
                    return prefix, suffix
    return None, None

def aggregate_file(models_list, file_name, output_path):
    """Агрегирует один файл из всех моделей по EntryID, суммируя все колонки и деля на 6"""
    dfs = []
    for model in models_list:
        model_path = Path(model)
        file_path = model_path / "predictions" / file_name
        if file_path.exists():
            df = pl.read_parquet(file_path)
            dfs.append(df)
    
    if dfs:
        # Объединяем все данные
        combined = pl.concat(dfs)
        
        # Группируем по EntryID и суммируем все остальные колонки, затем делим на 6 и приводим к int
        if len(combined.columns) > 1:
            # Все колонки кроме EntryID
            value_cols = [col for col in combined.columns if col != 'EntryID']
            
            # Агрегируем: группируем по EntryID, суммируем, делим на 6 и приводим к int
            aggregated = combined.group_by('EntryID').agg([
                (pl.col(col).sum() / 6).cast(pl.Int32).alias(col) for col in value_cols
            ])
            
            # Переупорядочиваем колонки: EntryID первая, остальные в алфавитном порядке
            all_columns = ['EntryID'] + sorted(value_cols)
            aggregated = aggregated.select(all_columns)
            
            aggregated.write_parquet(output_path)
            return True
    return False

def aggregate_test_folders(models_list, test_output_dir):
    """Агрегирует все файлы из test папок"""
    # Собираем все файлы из test папок всех моделей
    test_files_data = {}
    
    for model in models_list:
        model_path = Path(model)
        test_dir = model_path / "predictions" / "test"
        if test_dir.exists():
            for file_path in test_dir.glob("*.parquet"):
                if file_path.name not in test_files_data:
                    test_files_data[file_path.name] = []
                test_files_data[file_path.name].append(file_path)
    
    # Для каждого файла агрегируем данные из всех моделей
    for file_name, file_paths in test_files_data.items():
        dfs = []
        for file_path in file_paths:
            df = pl.read_parquet(file_path)
            dfs.append(df)
        
        if dfs:
            # Объединяем все данные
            combined = pl.concat(dfs)
            
            # Группируем по EntryID и суммируем все остальные колонки, затем делим на 6 и приводим к int
            if len(combined.columns) > 1 and 'EntryID' in combined.columns:
                # Все колонки кроме EntryID
                value_cols = [col for col in combined.columns if col != 'EntryID']
                
                # Агрегируем: группируем по EntryID, суммируем, делим на 6 и приводим к int
                aggregated = combined.group_by('EntryID').agg([
                    (pl.col(col).sum() / 6).cast(pl.Int32).alias(col) for col in value_cols
                ])
                
                # Переупорядочиваем колонки: EntryID первая, остальные в алфавитном порядке
                all_columns = ['EntryID'] + sorted(value_cols)
                aggregated = aggregated.select(all_columns)
                
                aggregated.write_parquet(test_output_dir / file_name)

def aggregate_predictions(model_groups):
    """Агрегирует предсказания моделей"""
    for (prefix, suffix), models_list in model_groups.items():
        print(f"Агрегация для {prefix}_{suffix}...")
        
        # Создаем папку для агрегированных результатов
        output_folder = Path(f"{prefix}_{suffix}")
        output_folder.mkdir(exist_ok=True)
        
        # Создаем подпапки
        dumps_folder = output_folder / "dumps"
        predictions_folder = output_folder / "predictions"
        dumps_folder.mkdir(exist_ok=True)
        predictions_folder.mkdir(exist_ok=True)
        
        # Копируем config.yaml из первой модели
        if models_list:
            first_model_path = Path(models_list[0])
            config_path = first_model_path / "dumps" / "config.yaml"
            if config_path.exists():
                shutil.copy(config_path, dumps_folder / "config.yaml")
        
        # Агрегация OOF предсказаний
        success1 = aggregate_file(models_list, "oof_pred.parquet", predictions_folder / "oof_pred.parquet")
        success2 = aggregate_file(models_list, "oof_old_pred.parquet", predictions_folder / "oof_old_pred.parquet")
        
        if success1:
            print(f"  Создан oof_pred.parquet для {prefix}_{suffix}")
        if success2:
            print(f"  Создан oof_old_pred.parquet для {prefix}_{suffix}")
        
        # Агрегация тестовых предсказаний
        test_output_dir = predictions_folder / "test"
        test_output_dir.mkdir(exist_ok=True)
        aggregate_test_folders(models_list, test_output_dir)
        print(f"  Агрегированы тестовые файлы для {prefix}_{suffix}")

def main():
    # Группируем модели по префиксам и суффиксам
    model_groups = {}
    
    for model in models:
        prefix, suffix = get_model_info(model)
        if prefix and suffix:
            key = (prefix, suffix)
            if key not in model_groups:
                model_groups[key] = []
            model_groups[key].append(model)
    
    # Выводим группы для проверки
    print("Группы моделей:")
    for (prefix, suffix), models_list in model_groups.items():
        print(f"{prefix}_{suffix}: {len(models_list)} моделей")
    
    # Агрегируем предсказания
    aggregate_predictions(model_groups)
    print("\nАгрегация завершена!")

if __name__ == "__main__":
    main()

