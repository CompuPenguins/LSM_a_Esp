"""
pointcloud_models.py
====================
Dos modelos de nube de puntos para reconocimiento de glosas LSM.

Ambos reciben la misma entrada y exponen la misma interfaz:
    model.fit(X, y)          -- entrenar
    model.predict(X)         -- predecir clase
    model.predict_proba(X)   -- distribución de probabilidad

INPUT esperado:
    X : np.ndarray (N, T, 133, 2)  -- N muestras, T frames, 133 kpts, x/y
    y : np.ndarray (N,)            -- etiquetas enteras

Internamente ambos modelos colapsan la dimensión temporal mediante
un descriptor de nube de puntos que captura tanto la FORMA (estáticas)
como la TRAYECTORIA (dinámicas).

Descriptor de nube de puntos usado:
    Para cada muestra (T, 133, 2) se computa un vector fijo que incluye:
      - Centroide de la nube por frame → trayectoria del movimiento
      - Covarianza de keypoints → forma/dispersión espacial
      - Percentiles de velocidad entre frames → dinamismo
      - Estadísticos de manos izq/der por separado

Modelos:
    1. DistanceModel  : kNN con distancia Euclidea sobre descriptores
    2. PointNetModel  : MLP ligero que aprende el espacio de probabilidades
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# DESCRIPTOR DE NUBE DE PUNTOS
# Convierte (T, 133, 2) → vector 1D de tamaño fijo
# ─────────────────────────────────────────────────────────────────────────────

# Índices COCO-WholeBody relevantes
_BODY_IDX   = list(range(0, 17))         # pose principal
_LHAND_IDX  = list(range(91, 112))       # mano izquierda (21 kpts)
_RHAND_IDX  = list(range(112, 133))      # mano derecha   (21 kpts)
_FACE_IDX   = list(range(23, 91))        # cara (ignorada en el descriptor)

def _cloud_descriptor(seq: np.ndarray) -> np.ndarray:
    """
    seq : (T, 133, 2) float32
    Returns: vector 1D de tamaño fijo (~120 dims)
    """
    T = seq.shape[0]
    feats = []

    for indices, name in [(_BODY_IDX, "body"), (_LHAND_IDX, "lhand"), (_RHAND_IDX, "rhand")]:
        cloud = seq[:, indices, :]           # (T, K, 2)
        K = len(indices)

        # --- Forma global: centroide y dispersión espacial ---
        centroid = cloud.mean(axis=1)        # (T, 2)  — trayectoria del centroide
        centered = cloud - centroid[:, None, :]   # (T, K, 2)

        # Estadísticos de posición media (inicio, medio, fin)
        feats.append(centroid[0])            # (2,)  posición inicial
        feats.append(centroid[T//2])         # (2,)  posición media
        feats.append(centroid[-1])           # (2,)  posición final
        feats.append(centroid.mean(axis=0))  # (2,)  centroide promedio

        # Rango de movimiento del centroide (bbox de la trayectoria)
        feats.append(centroid.max(axis=0) - centroid.min(axis=0))  # (2,)

        # --- Forma de la nube: covarianza aplanada ---
        flat = centered.reshape(T, K * 2)    # (T, K*2)
        cov  = np.cov(flat.T)                # (K*2, K*2) — grande, reducir
        # Tomar solo la diagonal (varianza por coordenada) y primeros eigenvalues
        diag = np.diag(cov)                  # (K*2,)
        # Resumir en percentiles para tamaño fijo
        feats.append(np.percentile(diag, [25, 50, 75, 100]))  # (4,)

        # --- Dinámica: velocidad entre frames ---
        vel = np.diff(centroid, axis=0)      # (T-1, 2)
        speed = np.linalg.norm(vel, axis=-1) # (T-1,)
        if len(speed) > 0:
            feats.append(np.array([
                speed.mean(),
                speed.std(),
                speed.max(),
                np.percentile(speed, 75),
            ]))                              # (4,)
        else:
            feats.append(np.zeros(4))

        # --- Velocidad de los keypoints de manos (más discriminativo) ---
        kvel = np.diff(cloud, axis=0)        # (T-1, K, 2)
        kspeed = np.linalg.norm(kvel, axis=-1)  # (T-1, K)
        if len(kspeed) > 0:
            feats.append(np.percentile(kspeed, [25, 50, 75], axis=0).mean(axis=1))  # (3,)
        else:
            feats.append(np.zeros(3))

    descriptor = np.concatenate([f.ravel() for f in feats])
    return descriptor.astype(np.float32)


def build_descriptors(X: np.ndarray) -> np.ndarray:
    """
    X : (N, T, 133, 2)
    Returns: (N, D) descriptores
    """
    return np.stack([_cloud_descriptor(x) for x in X])


# ─────────────────────────────────────────────────────────────────────────────
# MODELO 1: DISTANCIA — kNN sobre descriptores normalizados
# ─────────────────────────────────────────────────────────────────────────────

class DistanceModel:
    """
    Clasificador basado en distancia Euclidea en el espacio de descriptores.
    Usa kNN con normalización StandardScaler.

    Simple, interpretable, sin backprop.
    Funciona bien para glosas con forma espacial distintiva.
    Puede fallar en glosas cuya diferencia es solo temporal (velocidad/ritmo).
    """

    def __init__(self, k: int = 5, metric: str = "euclidean"):
        """
        k      : vecinos más cercanos
        metric : 'euclidean' | 'cosine' | 'manhattan'
        """
        self.k = k
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("knn",    KNeighborsClassifier(
                n_neighbors=k,
                metric=metric,
                weights="distance",   # vecinos cercanos pesan más
                algorithm="ball_tree",
                n_jobs=-1,
            )),
        ])
        self.classes_: Optional[np.ndarray] = None
        self._desc_cache: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DistanceModel":
        """
        X : (N, T, 133, 2)
        y : (N,) int
        """
        print(f"[DistanceModel] Extrayendo descriptores de {len(X)} muestras...")
        D = build_descriptors(X)
        self._desc_cache = D
        print(f"[DistanceModel] Descriptor dim: {D.shape[1]}")
        self.pipeline.fit(D, y)
        self.classes_ = self.pipeline.named_steps["knn"].classes_
        print(f"[DistanceModel] Entrenado. k={self.k}, clases={len(self.classes_)}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N, T, 133, 2) → (N,) int"""
        D = build_descriptors(X)
        return self.pipeline.predict(D)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X: (N, T, 133, 2) → (N, num_classes) float"""
        D = build_descriptors(X)
        return self.pipeline.predict_proba(D)

    def predict_single(
        self,
        seq: np.ndarray,
        top_k: int = 3,
    ) -> dict:
        """
        seq : (T, 133, 2) — una sola glosa
        Returns dict con clase predicha, confianza y top-k
        """
        X = seq[None]                          # (1, T, 133, 2)
        proba = self.predict_proba(X)[0]       # (num_classes,)
        top_idx = np.argsort(proba)[::-1][:top_k]
        return {
            "predicted_class": int(self.classes_[top_idx[0]]),
            "confidence":      float(proba[top_idx[0]]),
            "top_k": [
                {"class": int(self.classes_[i]), "prob": float(proba[i])}
                for i in top_idx
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# MODELO 2: PROBABILÍSTICO — PointNet ligero
# ─────────────────────────────────────────────────────────────────────────────

class _PointNetBackbone(nn.Module):
    """
    PointNet simplificado sobre la nube de puntos temporal.

    Trata cada frame como un "punto" en R^(133*2):
        (B, T, 266) → max-pooling global → (B, latent_dim)

    Luego fusiona con el descriptor estadístico para capturar
    tanto la estructura global como la dinámica.
    """

    def __init__(self, input_dim: int = 266, latent_dim: int = 256):
        super().__init__()

        # Shared MLP por punto (equiv. Conv1D con kernel=1)
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),   # aplicado sobre el canal, no el tiempo
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )

        # BatchNorm sobre la dimensión de canal en secuencias
        # Se aplica transponiendo: (B, T, C) → (B, C, T) → BN → (B, T, C)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(latent_dim)

        self.latent_dim = latent_dim

    def forward(self, x: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
        """
        x          : (B, T, 266)
        valid_mask : (B, T) bool — True en frames reales
        Returns    : (B, latent_dim)
        """
        B, T, C = x.shape

        # Shared MLP por punto — procesar como (B*T, C) para BatchNorm
        x_flat = x.reshape(B * T, C)

        h = F.relu(self.bn1(nn.Linear(C, 128, device=x.device, dtype=x.dtype)(x_flat)))
        # Nota: usar módulos pre-definidos es más eficiente; aquí usamos
        # la versión con parámetros fijos del __init__

        # Re-implementar correctamente con los módulos del __init__:
        h = x_flat
        for i, layer in enumerate(self.point_mlp):
            if isinstance(layer, nn.BatchNorm1d):
                h = layer(h)
            elif isinstance(layer, nn.Linear):
                h = layer(h)
            elif isinstance(layer, nn.ReLU):
                h = F.relu(h)

        h = h.reshape(B, T, self.latent_dim)  # (B, T, latent_dim)

        # Max-pooling global sobre frames válidos
        if valid_mask is not None:
            # Enmascarar frames de padding con -inf antes del max
            mask = valid_mask.unsqueeze(-1).float()           # (B, T, 1)
            h = h * mask + (1 - mask) * (-1e9)

        global_feat = h.max(dim=1).values                     # (B, latent_dim)
        return global_feat


class PointNetModel(nn.Module):
    """
    Modelo probabilístico basado en PointNet para clasificación de glosas.

    Arquitectura:
        1. PointNet backbone → max-pooling global sobre frames
        2. Descriptor estadístico (mismo que DistanceModel) → MLP
        3. Fusión de ambas representaciones → clasificador
        4. Softmax → distribución de probabilidad sobre glosas

    Ventajas sobre kNN:
        - Aprende qué aspectos de la nube son discriminativos
        - Puede combinar forma y dinámica de forma no lineal
        - Produce probabilidades calibradas con temperatura

    Desventajas:
        - Requiere entrenamiento (necesita GPU / tiempo)
        - Menos interpretable que kNN
    """

    def __init__(
        self,
        num_classes:   int   = 249,
        input_dim:     int   = 266,    # 133 * 2
        latent_dim:    int   = 256,
        desc_dim:      int   = None,   # se infiere en el primer forward
        dropout:       float = 0.3,
        temperature:   float = 1.0,    # para calibración de probabilidades
    ):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim  = latent_dim
        self.temperature = temperature
        self._desc_dim   = desc_dim

        # --- PointNet sobre frames ---
        self.frame_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )

        # --- MLP sobre descriptor estadístico (se construye en primer forward) ---
        # Se inicializa en _build_desc_mlp() tras conocer desc_dim
        self.desc_mlp   = None
        self._built     = False

        # --- Clasificador final (se construye tras conocer desc_dim) ---
        self.classifier = None
        self.dropout    = nn.Dropout(dropout)

    def _build_desc_mlp(self, desc_dim: int, device):
        """Construye los módulos que dependen del tamaño del descriptor."""
        self.desc_mlp = nn.Sequential(
            nn.Linear(desc_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout.p),
            nn.Linear(128, 128),
            nn.ReLU(),
        ).to(device)

        fusion_dim = self.latent_dim + 128
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout.p),
            nn.Linear(256, self.num_classes),
        ).to(device)

        self._built  = True
        self._desc_dim = desc_dim

    def forward(
        self,
        frames:    Tensor,                     # (B, T, 266)
        desc:      Tensor,                     # (B, desc_dim)
        valid_mask: Optional[Tensor] = None,   # (B, T) bool
    ) -> Tensor:
        """Returns logits (B, num_classes)"""

        # Construir módulos si es el primer forward
        if not self._built:
            self._build_desc_mlp(desc.shape[1], frames.device)

        B, T, C = frames.shape

        # 1. PointNet: shared MLP por frame → max-pooling global
        h = frames.reshape(B * T, C)
        h = self.frame_mlp(h)                  # (B*T, latent_dim)
        h = h.reshape(B, T, self.latent_dim)   # (B, T, latent_dim)

        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1).float()
            h = h * mask + (1 - mask) * (-1e9)
        global_feat = h.max(dim=1).values      # (B, latent_dim)

        # 2. Descriptor estadístico
        desc_feat = self.desc_mlp(desc)        # (B, 128)

        # 3. Fusión + clasificador
        fused  = torch.cat([global_feat, desc_feat], dim=-1)   # (B, latent_dim+128)
        fused  = self.dropout(fused)
        logits = self.classifier(fused)        # (B, num_classes)

        return logits / self.temperature

    def predict_proba_tensor(
        self,
        frames:    Tensor,
        desc:      Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns probabilidades (B, num_classes)"""
        with torch.no_grad():
            logits = self(frames, desc, valid_mask)
            return F.softmax(logits, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# WRAPPER DE ENTRENAMIENTO para PointNetModel
# ─────────────────────────────────────────────────────────────────────────────

class PointNetTrainer:
    """
    Wrapper sklearn-like para entrenar y evaluar PointNetModel.

    Uso:
        trainer = PointNetTrainer(num_classes=249)
        trainer.fit(X_train, y_train)
        preds   = trainer.predict(X_test)
        probas  = trainer.predict_proba(X_test)
    """

    def __init__(
        self,
        num_classes:  int   = 249,
        latent_dim:   int   = 256,
        dropout:      float = 0.3,
        lr:           float = 1e-3,
        epochs:       int   = 50,
        batch_size:   int   = 32,
        patience:     int   = 10,
        temperature:  float = 1.0,
        device:       str   = "auto",
        tmax:         int   = 200,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_classes = num_classes
        self.tmax        = tmax
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.patience    = patience

        self.model = PointNetModel(
            num_classes=num_classes,
            latent_dim=latent_dim,
            dropout=dropout,
            temperature=temperature,
        ).to(self.device)

        self.optimizer = None   # se crea en fit() tras conocer desc_dim
        self.lr        = lr
        self.classes_: Optional[np.ndarray] = None
        self.scaler    = StandardScaler()

    # ------------------------------------------------------------------
    def _prepare(
        self, X: np.ndarray, fit_scaler: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        X : (N, T, 133, 2)
        Returns: frames (N, T, 266), desc (N, D), valid_mask (N, T)
        """
        N, T_orig, K, C = X.shape

        # Padding / truncamiento a tmax
        T = min(T_orig, self.tmax)
        X_trunc = X[:, :T, :, :]                             # (N, T, 133, 2)
        frames  = X_trunc.reshape(N, T, K * C).astype(np.float32)  # (N, T, 266)

        # Máscara de validez (todos reales si no hay padding)
        valid = np.ones((N, T), dtype=bool)

        # Padding a tmax si es necesario
        if T < self.tmax:
            pad_frames = np.zeros((N, self.tmax - T, K * C), dtype=np.float32)
            frames = np.concatenate([frames, pad_frames], axis=1)
            pad_valid = np.zeros((N, self.tmax - T), dtype=bool)
            valid  = np.concatenate([valid, pad_valid], axis=1)

        # Descriptor estadístico
        desc = build_descriptors(X)                           # (N, D)
        if fit_scaler:
            desc = self.scaler.fit_transform(desc)
        else:
            desc = self.scaler.transform(desc)

        frames_t = torch.from_numpy(frames).to(self.device)
        desc_t   = torch.from_numpy(desc.astype(np.float32)).to(self.device)
        valid_t  = torch.from_numpy(valid).to(self.device)

        return frames_t, desc_t, valid_t

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "PointNetTrainer":
        """
        X : (N, T, 133, 2)
        y : (N,) int
        """
        print(f"[PointNetTrainer] Preparando datos ({len(X)} muestras)...")
        frames_t, desc_t, valid_t = self._prepare(X, fit_scaler=True)
        labels_t = torch.from_numpy(y.astype(np.int64)).to(self.device)
        self.classes_ = np.unique(y)

        dataset = TensorDataset(frames_t, desc_t, valid_t, labels_t)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        # Trigger de construcción de módulos internos con un mini-batch
        dummy_f, dummy_d, dummy_v, _ = next(iter(loader))
        _ = self.model(dummy_f[:1], dummy_d[:1], dummy_v[:1])   # build interno
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        print(f"[PointNetTrainer] Entrenando en {self.device}...")
        print(f"{'Epoch':>6} | {'Loss':>8} | {'Acc':>6}")
        print("-" * 30)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total   = 0

            for frames_b, desc_b, valid_b, labels_b in loader:
                self.optimizer.zero_grad()
                logits = self.model(frames_b, desc_b, valid_b)
                loss   = criterion(logits, labels_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                preds   = logits.argmax(dim=-1)
                correct += (preds == labels_b).sum().item()
                total   += len(labels_b)

            scheduler.step()
            avg_loss = epoch_loss / len(loader)
            acc      = correct / total

            if epoch % 5 == 0 or epoch == 1:
                print(f"{epoch:>6d} | {avg_loss:>8.4f} | {acc:>6.3f}")

            # Early stopping
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping en época {epoch}.")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        print(f"[PointNetTrainer] Entrenamiento completo. Mejor loss: {best_loss:.4f}")
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N, T, 133, 2) → (N,) int"""
        proba = self.predict_proba(X)
        idx   = proba.argmax(axis=-1)
        return self.classes_[idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X: (N, T, 133, 2) → (N, num_classes) float"""
        self.model.eval()
        frames_t, desc_t, valid_t = self._prepare(X, fit_scaler=False)

        with torch.no_grad():
            logits = self.model(frames_t, desc_t, valid_t)
            proba  = F.softmax(logits, dim=-1).cpu().numpy()
        return proba

    def predict_single(self, seq: np.ndarray, top_k: int = 3) -> dict:
        """
        seq : (T, 133, 2) — una sola glosa grabada
        """
        proba = self.predict_proba(seq[None])[0]   # (num_classes,)
        top_idx = np.argsort(proba)[::-1][:top_k]
        return {
            "predicted_class": int(self.classes_[top_idx[0]]),
            "confidence":      float(proba[top_idx[0]]),
            "top_k": [
                {"class": int(self.classes_[i]), "prob": float(proba[i])}
                for i in top_idx
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# COMPARADOR: misma interfaz, ambos modelos
# ─────────────────────────────────────────────────────────────────────────────

class ModelComparison:
    """
    Entrena y evalúa ambos modelos con los mismos datos.

    Uso:
        cmp = ModelComparison(num_classes=249)
        results = cmp.fit_evaluate(X_train, y_train, X_test, y_test)
        cmp.print_report(results)
    """

    def __init__(self, num_classes: int = 249, k: int = 5, **pointnet_kwargs):
        self.distance_model = DistanceModel(k=k)
        self.pointnet_model = PointNetTrainer(num_classes=num_classes, **pointnet_kwargs)

    def fit_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
    ) -> dict:
        from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

        results = {}

        for name, model in [
            ("DistanceModel (kNN)", self.distance_model),
            ("PointNetModel",       self.pointnet_model),
        ]:
            print(f"\n{'='*50}")
            print(f"  {name}")
            print(f"{'='*50}")

            model.fit(X_train, y_train)
            preds  = model.predict(X_test)
            probas = model.predict_proba(X_test)
            labels = np.unique(y_train)

            top1 = accuracy_score(y_test, preds)
            f1   = f1_score(y_test, preds, average="macro",
                            labels=list(range(len(labels))), zero_division=0)
            top3 = top_k_accuracy_score(y_test, probas, k=3,
                                        labels=list(range(len(labels))))

            results[name] = {
                "top1_accuracy": top1,
                "top3_accuracy": top3,
                "macro_f1":      f1,
            }

        return results

    @staticmethod
    def print_report(results: dict):
        print(f"\n{'─'*55}")
        print(f"{'Modelo':<30} {'Top-1':>7} {'Top-3':>7} {'F1':>7}")
        print(f"{'─'*55}")
        for name, m in results.items():
            print(
                f"{name:<30} "
                f"{m['top1_accuracy']:>7.3f} "
                f"{m['top3_accuracy']:>7.3f} "
                f"{m['macro_f1']:>7.3f}"
            )
        print(f"{'─'*55}")

    def predict_single(self, seq: np.ndarray, top_k: int = 3) -> dict:
        """
        Predicción en tiempo real con ambos modelos.
        seq : (T, 133, 2)
        """
        return {
            "distance_model": self.distance_model.predict_single(seq, top_k),
            "pointnet_model": self.pointnet_model.predict_single(seq, top_k),
        }