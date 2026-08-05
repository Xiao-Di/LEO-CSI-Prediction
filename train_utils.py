"""
可复用的训练与评估工具函数。
包括: 离线训练、MAML 内循环适配、模型评估。
支持两种数据格式:
  - 单星 (x, aux, y): VanillaLSTM, VanillaTransformer
  - 多星 (csi_hist, node_attrs, y_csi, mask): HybridV2
"""

import os
import torch
import pandas as pd
import numpy as np

from config import DEVICE, LR_OFFLINE, LR_MAML, EPOCHS, LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE, LR_MIN, EARLY_STOP_PATIENCE
from metrics import calculate_nmse_db


def _unpack_batch(batch):
    """
    统一解包批次:
      - (x, aux, y) → return x, aux, y, None
      - (csi_hist, node_attrs, y_csi, mask) → return csi_hist, node_attrs, y_csi, mask
    """
    if len(batch) == 4:
        csi_hist, node_attrs, y_csi, mask = batch
        return csi_hist, node_attrs, y_csi, mask
    else:
        x, aux, y = batch
        return x, aux, y, None


def _model_forward(model, x, aux, mask=None):
    """
    统一模型前向传播:
      - 单星模型: model(x, aux)
      - 多星模型: model(csi_hist, node_attrs, mask)
    """
    if mask is not None:
        return model(x, aux, mask)
    return model(x, aux)


def train_model(
    model,
    train_loader,
    criterion,
    use_physics_loss,
    val_loader=None,
    optimizer=None,
    epochs=None,
    early_stop=None,
    verbose=True,
):
    """
    离线预训练一个模型（支持早停）。

    参数:
        model:              待训练模型
        train_loader:       DataLoader (返回 x, aux, y 或 csi_hist, node_attrs, y_csi, mask)
        criterion:          损失函数 (MSELoss 或 SAGINPhysicsLoss)
        use_physics_loss:   criterion 是否返回 (total, nmse, phys) 三元组
        val_loader:         验证集 DataLoader，用于早停监控（可选）
        optimizer:          优化器，默认 AdamW(lr=LR_OFFLINE)
        epochs:             训练轮数，默认 EPOCHS
        early_stop:         早停 patience（val loss 无改善的轮数），默认 20
        verbose:            是否打印训练日志

    返回:
        (loss_curve, stopped_epoch): loss 曲线列表和实际停止的 epoch
        若未触发早停，stopped_epoch = epochs
    """
    if early_stop is None:
        early_stop = EARLY_STOP_PATIENCE
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_OFFLINE)
    if epochs is None:
        epochs = EPOCHS

    model.train()
    loss_curve = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE, min_lr=LR_MIN,
    )

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # 动态物理损失权重: 仅前 30% epoch 启用，之后关闭
        # 防止在模型已收敛时物理梯度覆盖回归梯度
        phys_ratio = max(0.0, 1.0 - epoch / (epochs * 0.3))

        epoch_loss = 0
        for batch in train_loader:
            x, aux, y, mask = _unpack_batch(batch)
            x, aux, y = x.to(DEVICE), aux.to(DEVICE), y.to(DEVICE)
            if mask is not None:
                mask = mask.to(DEVICE)
            optimizer.zero_grad()

            pred = _model_forward(model, x, aux, mask)
            if use_physics_loss:
                # 物理损失需要主星辅助特征和历史 CSI 最后一步
                if mask is not None:
                    s_aux = aux[:, 0, :]  # 主星
                    x_last = x[:, 0, -1, :]  # 主星最后一步 CSI
                else:
                    s_aux = aux
                    x_last = x[:, -1, :]
                _, l_nmse, l_physics = criterion(pred, y, s_aux, x_last)
                # 动态加权
                loss = criterion.lambda_nmse * l_nmse + phys_ratio * criterion.lambda_phys * l_physics
            else:
                loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_curve.append(avg_loss)

        if val_loader is not None:
            val_loss = _compute_avg_loss(model, val_loader, criterion, use_physics_loss)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            scheduler.step(avg_loss)

        if verbose:
            current_lr = optimizer.param_groups[0]["lr"]
            if val_loader is not None:
                print(
                    f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} "
                    f"Val: {val_loss:.6f} (LR: {current_lr:.2e})"
                )
            else:
                print(
                    f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} (LR: {current_lr:.2e})"
                )

        if val_loader is not None and patience_counter >= early_stop:
            if verbose:
                print(f"  早停触发 @ epoch {epoch+1} (patience={early_stop}, "
                      f"best_val={best_val_loss:.6f})")
            model.load_state_dict(best_model_state)
            return loss_curve, epoch + 1

    if val_loader is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
    return loss_curve, epochs


def _compute_avg_loss(model, loader, criterion, use_physics_loss):
    """计算验证集平均 loss"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            x, aux, y, mask = _unpack_batch(batch)
            x, aux, y = x.to(DEVICE), aux.to(DEVICE), y.to(DEVICE)
            if mask is not None:
                mask = mask.to(DEVICE)
            pred = _model_forward(model, x, aux, mask)
            if use_physics_loss:
                if mask is not None:
                    s_aux = aux[:, 0, :]
                    x_last = x[:, 0, -1, :]
                else:
                    s_aux = aux
                    x_last = x[:, -1, :]
                loss, _, _ = criterion(pred, y, s_aux, x_last)
            else:
                loss = criterion(pred, y)
            total_loss += loss.item()
    model.train()
    return total_loss / len(loader)


def maml_inner_loop(
    model,
    support_loader,
    criterion,
    use_physics_loss,
    adapt_steps=1,
    lr=None,
    verbose=True,
):
    """MAML 内循环适配 (Inner Loop)。"""
    if lr is None:
        lr = LR_MAML

    model.train()
    maml_opt = torch.optim.SGD(model.parameters(), lr=lr)
    batch = next(iter(support_loader))
    sx, s_aux, sy, s_mask = _unpack_batch(batch)
    sx, s_aux, sy = sx.to(DEVICE), s_aux.to(DEVICE), sy.to(DEVICE)
    if s_mask is not None:
        s_mask = s_mask.to(DEVICE)

    adapt_curve = []
    for step in range(1, adapt_steps + 1):
        maml_opt.zero_grad()
        pred = _model_forward(model, sx, s_aux, s_mask)
        if use_physics_loss:
            if s_mask is not None:
                s_aux_phys = s_aux[:, 0, :]
                x_last = sx[:, 0, -1, :]
            else:
                s_aux_phys = s_aux
                x_last = sx[:, -1, :]
            loss, l_nmse, l_physics = criterion(pred, sy, s_aux_phys, x_last)
        else:
            loss = criterion(pred, sy)

        loss.backward()
        maml_opt.step()

        if verbose:
            print(f"  MAML Step {step}/{adapt_steps} - Loss: {loss.item():.6f}")
        adapt_curve.append(
            {
                "Step": step,
                "Loss": loss.item(),
                "L_NMSE": l_nmse.item() if use_physics_loss else loss.item(),
                "L_Physics": l_physics.item() if use_physics_loss else 0.0,
            }
        )

    return pd.DataFrame(adapt_curve)


def evaluate_model(
    model,
    test_loader,
    use_physics_loss=False,  # kept for API compatibility
    return_details=False,
    max_details=3,
    config_label=None,
    config_value=None,
):
    """
    在测试集上评估模型，返回平均 NMSE (dB)。
    """
    model.eval()
    batch_nmse = []
    details = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x, aux, y, mask = _unpack_batch(batch)
            x, aux, y = x.to(DEVICE), aux.to(DEVICE), y.to(DEVICE)
            if mask is not None:
                mask = mask.to(DEVICE)
            pred = _model_forward(model, x, aux, mask)
            batch_nmse.append(calculate_nmse_db(pred, y))

            if return_details and i < max_details:
                detail = {"Sample_ID": i}
                if config_label:
                    detail[config_label] = config_value
                detail["NMSE_dB"] = batch_nmse[-1]
                details.append(detail)

    avg_nmse = np.mean(batch_nmse)

    if return_details:
        return avg_nmse, pd.DataFrame(details)
    return avg_nmse


def parse_config_value(f_path):
    """从文件名解析配置值 (Speed 或 SNR)"""
    basename = os.path.basename(f_path)
    parts = basename.split("_")

    if "Speed_" in basename:
        # Exp1 / Exp3: Exp1_Speed_10_SNR20_test.csv
        # parts: ['Exp1', 'Speed', '10', 'SNR20', 'test.csv']
        speed_idx = parts.index("Speed") + 1
        return parts[speed_idx]
    elif "SNR_" in basename:
        # Exp2: Exp2_Speed100_SNR_5_test.csv
        # parts: ['Exp2', 'Speed100', 'SNR', '5', 'test.csv']
        snr_idx = parts.index("SNR") + 1
        return parts[snr_idx]
    else:
        # Fallback
        return parts[-1].replace(".csv", "")
