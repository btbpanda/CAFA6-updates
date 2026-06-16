import argparse
import os
import sys
import pandas as pd
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.utils import murmurhash3_32
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', type=str)
parser.add_argument('-t', '--fasta', type=str)
parser.add_argument('-c', '--config', type=str)
parser.add_argument('-e', '--embed-path', type=str)
parser.add_argument('-g', '--graph-path', type=str)
parser.add_argument('-fl', '--freq-labels', type=str)
parser.add_argument('-sl', '--sparse-labels', type=str)
parser.add_argument('-f', '--fold-id', type=int)
parser.add_argument('-d', '--device', type=str)


# ====================== Dataset ======================
class ProteinDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


# ====================== Модель с Skip Connections ======================
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3, use_dropout=True):
        super(ResidualBlock, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if use_dropout else nn.Identity()
        
        # Skip connection (проекция если размерности не совпадают)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Skip connection
        out = out + identity
        
        return out


class ProteinMLP(nn.Module):
    def __init__(self, input_dim, output_dim, 
                 hidden_dims=[2048, 1024, 512, 256], 
                 dropout=0.3, 
                 use_dropout=True):
        super(ProteinMLP, self).__init__()
        
        # Residual блоки
        self.blocks = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.blocks.append(ResidualBlock(prev_dim, hidden_dim, dropout, use_dropout))
            prev_dim = hidden_dim
        
        # Выходной слой
        self.output_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x):
        # Проходим через residual блоки
        for block in self.blocks:
            x = block(x)
        
        # Выходной слой
        x = self.output_layer(x)
        
        return x


# ====================== Loss с маскированием NaN ======================
class BCEWithLogitsLossNaN(nn.Module):
    """BCE Loss с маскированием NaN значений для conditional режима"""
    def __init__(self):
        super(BCEWithLogitsLossNaN, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, outputs, targets):
        # Создаем маску для не-NaN значений
        mask = ~torch.isnan(targets)
        
        # Заменяем NaN на 0 для вычисления loss (значения будут замаскированы)
        targets_masked = torch.where(mask, targets, torch.zeros_like(targets))
        
        # Вычисляем loss
        loss = self.bce(outputs, targets_masked)
        
        # Применяем маску
        loss = loss * mask.float()
        
        # Возвращаем среднее по не-NaN элементам
        return loss.sum() / mask.float().sum().clamp(min=1.0)


# ====================== Обучение ======================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_data in loader:
        if len(batch_data) == 2:
            batch_features, batch_labels = batch_data
        else:
            batch_features = batch_data[0]
            batch_labels = batch_data[0]  # не должно произойти
            
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 2:
                batch_features, batch_labels = batch_data
            else:
                batch_features = batch_data[0]
                batch_labels = batch_data[0]  # не должно произойти
                
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def predict(model, loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_data in loader:
            if isinstance(batch_data, (list, tuple)):
                batch_features = batch_data[0]
            else:
                batch_features = batch_data
            
            batch_features = batch_features.to(device)
            
            outputs = model(batch_features)
            preds = torch.sigmoid(outputs)
            predictions.append(preds.cpu().numpy())
    
    return np.vstack(predictions)


if __name__ == '__main__':
    
    args = parser.parse_args()
    
    # Optional: set the device to run
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    try:
        from protlib.metric import obo_parser, Graph, get_topk_targets
        from protlib.models.prepocess import get_features, get_folds, get_targets_from_parquet
    except ImportError:
        print('Alarm')
        pass
    
    ################################
    # PREPARE
    ################################
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    ontologies = []
    for ns, terms_dict in obo_parser(args.graph_path).items():
        ontologies.append(Graph(ns, terms_dict, None, True))
    
    # select required targets
    split = [config['bp'], config['mf'], config['cc']]
    cols = []
    
    for n, i in enumerate(split):
        cols.extend(get_topk_targets(
            ontologies[n],
            i,
            train_path=args.freq_labels
        ))
    
    ################################
    # READ TARGETS
    ################################
    
    Y_train = get_targets_from_parquet(
        os.path.join(args.sparse_labels, config['train_labels'], ),
        ontologies,
        split=split,
        ids=cols,
        fillna=not config['conditional']
    )
    
    Y_old = get_targets_from_parquet(
        os.path.join(args.sparse_labels, config['old_train_labels'], ),
        ontologies,
        split=split,
        ids=cols,
        fillna=not config['conditional']
    )
    
    Y_train = pd.concat([Y_train, Y_old, ], axis=0, ignore_index=True)
    
    Y_train, prot_names = Y_train.values, Y_train.columns.tolist()
    
    ################################
    # GET FEATURES
    ################################
    # main train file
    train = pd.read_feather(
        os.path.join(args.fasta, 'train_seq.feather')
    )
    train['is_cafa6'] = True
    cafa6_size = train.shape[0]
    
    X_train = get_features(
        train,
        embed_path=args.embed_path,
        prefix='train',
        embed_list=config['embeds'],
        tax_list=config['tax_list']
    )
    
    # old train file
    old_train = pd.read_feather(
        os.path.join(args.fasta, 'old_train_seq.feather')
    )
    old_train['is_cafa6'] = False
    cafa_old_size = old_train.shape[0]
    
    X_old = get_features(
        old_train,
        embed_path=args.embed_path,
        prefix='old_train',
        embed_list=config['embeds'],
        tax_list=config['tax_list']
    )
    
    # joint dataset
    train = pd.concat([train, old_train, ], ignore_index=True)
    train['fold'] = get_folds(train['seq'].apply(murmurhash3_32, seed=42))  # hash to speed up
    
    X_train = np.concatenate([X_train, X_old, ], axis=0)
    
    # test file
    test = pd.read_feather(
        os.path.join(args.fasta, 'test_seq.feather')
    )
    X_test = get_features(
        test,
        embed_path=args.embed_path,
        prefix='test',
        embed_list=config['embeds'],
        tax_list=config['tax_list']
    )
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    
    ################################
    # DUMP METADATA
    ################################
    
    os.makedirs(
        os.path.join(args.output, config['name'], 'dumps'),
        exist_ok=True
    )
    
    # oof pred
    os.makedirs(
        os.path.join(args.output, config['name'], 'oof_pred'),
        exist_ok=True
    )
    
    # old oof pred
    os.makedirs(
        os.path.join(args.output, config['name'], 'oof_old_pred'),
        exist_ok=True
    )
    
    # test pred
    os.makedirs(
        os.path.join(args.output, config['name'], 'test_pred'),
        exist_ok=True
    )
    
    # Сохраняем метаданные
    np.save(
        os.path.join(args.output, config['name'], 'dumps', 'prot_names.npy'),
        np.array(prot_names)
    )
    
    np.save(
        os.path.join(args.output, config['name'], 'dumps', 'prot_ids.npy'),
        np.array(cols)
    )
    
    with open(os.path.join(args.output, config['name'], 'dumps', 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f)
    
    ################################
    # TRAINING PARAMETERS
    ################################
    
    EPOCHS = config['train_params'].get('epochs', 30)
    BATCH_SIZE = config['train_params'].get('batch_size', 128)
    LEARNING_RATE = config['train_params'].get('learning_rate', 1e-3)
    WEIGHT_DECAY = config['train_params'].get('weight_decay', 0.0003)
    SWA_LR = config['train_params'].get('swa_lr', 3e-4)
    SWA_START_EPOCH = config['train_params'].get('swa_start_epoch', 3)
    EARLY_STOPPING_PATIENCE = config['train_params'].get('early_stopping_patience', 5)
    HIDDEN_DIMS = config['train_params'].get('hidden_dims', [2048, 1024, 512, 256])
    DROPOUT = config['train_params'].get('dropout', 0.3)
    USE_DROPOUT = config['train_params'].get('use_dropout', True)
    NUM_WORKERS = config['train_params'].get('num_workers', 4)
    
    ################################
    # TRAINING LOOP
    ################################
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nИспользуется устройство: {device}")
    
    input_dim = X_train.shape[1]
    output_dim = len(cols)
    
    print(f"\nПараметры модели:")
    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  Hidden dims: {HIDDEN_DIMS}")
    print(f"  Dropout: {DROPOUT if USE_DROPOUT else 'Disabled'}")
    print(f"  Embeds: {config['embeds']}")
    
    ################################
    # FIT PREDICT
    ################################
    
    train_sl = train['fold'] != args.fold_id
    valid_sl = train['fold'] == args.fold_id
    pred_sl = valid_sl
    
    if config['train_data'] == 'cafa6':
        train_sl = train_sl & train['is_cafa6']
    
    if config['valid_data'] == 'cafa6':
        valid_sl = valid_sl & train['is_cafa6']
    
    train_sl, valid_sl, pred_sl = np.nonzero(train_sl)[0], np.nonzero(valid_sl)[0], np.nonzero(pred_sl)[0]
    
    print(f"\n{'='*60}")
    print(f"Training Fold {args.fold_id}")
    print(f"{'='*60}")
    print(f"Train size: {train_sl.shape[0]}, Val size: {valid_sl.shape[0]}")
    
    # Проверяем, существует ли уже модель
    model_path = os.path.join(args.output, config['name'], 'dumps', f'model_{args.fold_id}.pth')
    model_exists = os.path.exists(model_path)
    
    if model_exists:
        print(f"\n{'='*60}")
        print(f"Модель уже обучена: {model_path}")
        print(f"Загружаем модель для предсказаний...")
        print(f"{'='*60}")
        
        # Загружаем модель
        checkpoint = torch.load(model_path)
        model = ProteinMLP(input_dim, output_dim, 
                          hidden_dims=HIDDEN_DIMS, 
                          dropout=DROPOUT, 
                          use_dropout=USE_DROPOUT).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        
    else:
        print(f"\nНачинаем обучение модели...")
        
        # Разделение данных
        X_train_fold = X_train[train_sl]
        y_train_fold = Y_train[train_sl]
        
        X_val_fold = X_train[valid_sl]
        y_val_fold = Y_train[valid_sl]
        
        # Создание датасетов и загрузчиков
        train_dataset = ProteinDataset(X_train_fold, y_train_fold)
        val_dataset = ProteinDataset(X_val_fold, y_val_fold)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        # Инициализация модели
        model = ProteinMLP(input_dim, output_dim, 
                          hidden_dims=HIDDEN_DIMS, 
                          dropout=DROPOUT, 
                          use_dropout=USE_DROPOUT).to(device)
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        
        # Используем loss с маскированием NaN (работает корректно и без NaN)
        criterion = BCEWithLogitsLossNaN()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, anneal_strategy="cos", swa_lr=SWA_LR)
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        # Обучение
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate(model, val_loader, criterion, device)
            
            # SWA
            if epoch >= SWA_START_EPOCH:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                
                torch.save({
                    'epoch': epoch,
                    'fold': args.fold_id,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': {
                        'input_dim': input_dim,
                        'output_dim': output_dim,
                        'hidden_dims': HIDDEN_DIMS,
                        'dropout': DROPOUT,
                        'use_dropout': USE_DROPOUT
                    }
                }, model_path)
                print(f"  → Checkpoint saved (val_loss: {val_loss:.4f})")
            else:
                epochs_without_improvement += 1
            
            swa_status = "SWA" if epoch >= SWA_START_EPOCH else "Regular"
            print(f"Epoch {epoch+1:2d}/{EPOCHS} [{swa_status}] - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} "
                  f"(best: {best_val_loss:.4f}, patience: {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE})")
            
            # Early stopping
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Обновление BN статистики для SWA модели
        print("\nОбновление BatchNorm статистики для SWA модели...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        
        # Финальная валидация с SWA моделью
        final_val_loss = validate(swa_model, val_loader, criterion, device)
        print(f"Final validation loss (SWA): {final_val_loss:.4f}")
    
    ################################
    # DUMP PREDS
    ################################
    
    # OOF предсказания
    print("\nСоздание OOF предсказаний...")
    X_pred_fold = X_train[pred_sl]
    pred_dataset = ProteinDataset(X_pred_fold)
    pred_loader = DataLoader(pred_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    oof_pred = predict(swa_model, pred_loader, device)
    
    # Сохранение OOF предсказаний в формате как в train_pb_c6.py
    oof_df = pd.DataFrame(oof_pred, columns=prot_names)
    oof_df['EntryID'] = train['EntryID'].values[pred_sl]
    oof_df['is_cafa6'] = train['is_cafa6'].values[pred_sl]
    
    oof_df.query('is_cafa6').drop('is_cafa6', axis=1).to_parquet(
        os.path.join(args.output, config['name'], 'oof_pred', f'fold_{args.fold_id}.parquet'),
        index=False,
    )
    print(f"OOF предсказания (CAFA6) сохранены: {os.path.join(args.output, config['name'], 'oof_pred', f'fold_{args.fold_id}.parquet')}")
    
    oof_df.query('~is_cafa6').drop('is_cafa6', axis=1).to_parquet(
        os.path.join(args.output, config['name'], 'oof_old_pred', f'fold_{args.fold_id}.parquet'),
        index=False,
    )
    print(f"OOF предсказания (old_train) сохранены: {os.path.join(args.output, config['name'], 'oof_old_pred', f'fold_{args.fold_id}.parquet')}")
    
    # Предсказания для теста
    print("Создание предсказаний для теста...")
    test_dataset = ProteinDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_pred = predict(swa_model, test_loader, device)
    
    # Сохранение предсказаний для теста в формате как в train_pb_c6.py
    test_pred_df = pd.DataFrame(test_pred, columns=prot_names)
    test_pred_df['EntryID'] = test['EntryID'].values
    
    test_pred_df.to_parquet(
        os.path.join(args.output, config['name'], 'test_pred', f'fold_{args.fold_id}.parquet'),
        index=False,
    )
    print(f"Test предсказания сохранены: {os.path.join(args.output, config['name'], 'test_pred', f'fold_{args.fold_id}.parquet')}")
    
    print(f"\n{'='*60}")
    print("Завершено!")
    print(f"{'='*60}")
    print(f"Checkpoint: {model_path}")
    print(f"OOF предсказания (CAFA6): {os.path.join(args.output, config['name'], 'oof_pred', f'fold_{args.fold_id}.parquet')}")
    print(f"OOF предсказания (old_train): {os.path.join(args.output, config['name'], 'oof_old_pred', f'fold_{args.fold_id}.parquet')}")
    print(f"Test предсказания: {os.path.join(args.output, config['name'], 'test_pred', f'fold_{args.fold_id}.parquet')}")

