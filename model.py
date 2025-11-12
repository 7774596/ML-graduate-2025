import numpy as np


def _rbf_kernel(X, Y, gamma):
    """RBF kernel K(x, y) = exp(-gamma * ||x - y||^2).

    Vectorized computation using (x - y)^2 = ||x||^2 + ||y||^2 - 2 x·y.
    Uses float32 for memory efficiency.
    """
    X = X.astype(np.float32, copy=False)
    Y = Y.astype(np.float32, copy=False)
    X_norm = np.sum(X * X, axis=1, keepdims=True)
    Y_norm = np.sum(Y * Y, axis=1, keepdims=True).T
    K = X_norm + Y_norm - 2.0 * (X @ Y.T)
    K = np.exp(-gamma * np.clip(K, 0.0, None)).astype(np.float32, copy=False)
    return K


def _poly_kernel(X, Y, gamma, degree, coef0):
    """Polynomial kernel K(x, y) = (gamma * x·y + coef0)^degree."""
    X = X.astype(np.float32, copy=False)
    Y = Y.astype(np.float32, copy=False)
    K = (gamma * (X @ Y.T) + coef0)
    # For numerical stability, cast to float32 after power
    K = np.power(K, degree).astype(np.float32, copy=False)
    return K


class Model:
    """
    Kernel SVM (one-vs-rest) implemented in pure NumPy with a simplified SMO.

    Notes:
    - Supports 'rbf' and 'poly' kernels.
    - Trains 10 binary classifiers for digits 0..9.
    - Only NumPy is used; external ML libraries are NOT used (per assignment rules).
    """

    def __init__(
        self,
        n_features=784,
        n_classes=10,
        kernel="rbf",
        C=2.0,
        gamma=None,
        degree=3,
        coef0=1.0,
        tol=1e-3,
        max_passes=3,
        max_iter=1000,
        verbose=True,
        seed=42,
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.kernel = kernel
        self.C = float(C)
        self.gamma = gamma  # if None, will be set in fit() via heuristic
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.max_passes = int(max_passes)
        self.max_iter = int(max_iter)
        self.verbose = bool(verbose)
        self.seed = int(seed)

        # Will be set during fit
        self._X_train = None  # (n_samples, n_features), used only during training
        self._K = None  # precomputed kernel matrix among training points
        # list per class: {sv_idx, sv_X, alpha_y, b}
        self._cls_params = []
        # Standardization params
        self.X_mean = None
        self.X_std = None

    # ------------------------ Kernel utilities ------------------------
    def _get_gamma(self, X):
        # By the time we call kernel ops in training/inference, self.gamma must be numeric.
        if self.gamma is None:
            # fallback to safe default
            return 1.0 / float(X.shape[1])
        return float(self.gamma)

    def _kernel_matrix(self, X, Y=None):
        if Y is None:
            Y = X
        if self.kernel == "rbf":
            gamma = self._get_gamma(X if Y is X else self._X_train if self._X_train is not None else X)
            return _rbf_kernel(X, Y, gamma)
        elif self.kernel == "poly":
            gamma = self._get_gamma(X if Y is X else self._X_train if self._X_train is not None else X)
            return _poly_kernel(X, Y, gamma, self.degree, self.coef0)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    # ------------------------ SMO optimizer ------------------------
    def _smo_train_binary(self, K, y, rng):
        """Train binary SVM with SMO on a precomputed kernel matrix.

        Args:
            K: (n, n) kernel matrix (float32)
            y: (n,) labels in {-1, +1} (float32)
            rng: np.random.RandomState
        Returns:
            alphas: (n,) Lagrange multipliers
            b: bias
        """
        n = K.shape[0]
        alphas = np.zeros(n, dtype=np.float32)
        b = 0.0
        # f = K @ (alphas * y) + b, maintained incrementally
        f = np.zeros(n, dtype=np.float32)

        passes = 0
        iters = 0
        eps = 1e-5

        while passes < self.max_passes and iters < self.max_iter:
            num_changed = 0
            for i in range(n):
                # Decision error with current bias b
                E_i = (f[i] + b) - y[i]
                cond1 = (y[i] * E_i < -self.tol and alphas[i] < self.C)
                cond2 = (y[i] * E_i > self.tol and alphas[i] > 0.0)
                if not (cond1 or cond2):
                    continue

                # Select j != i randomly
                j = i
                while j == i:
                    j = rng.randint(0, n)

                E_j = (f[j] + b) - y[j]

                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                if y[i] != y[j]:
                    L = max(0.0, alpha_j_old - alpha_i_old)
                    H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                else:
                    L = max(0.0, alpha_i_old + alpha_j_old - self.C)
                    H = min(self.C, alpha_i_old + alpha_j_old)
                if L == H:
                    continue

                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                # Update alpha_j
                alpha_j_new = alpha_j_old - y[j] * (E_i - E_j) / eta
                # Clip
                if alpha_j_new > H:
                    alpha_j_new = H
                elif alpha_j_new < L:
                    alpha_j_new = L

                if abs(alpha_j_new - alpha_j_old) < eps:
                    continue

                # Update alpha_i accordingly
                alpha_i_new = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha_j_new)

                # Compute b1, b2 and update b
                b1 = (
                    b
                    - E_i
                    - y[i] * (alpha_i_new - alpha_i_old) * K[i, i]
                    - y[j] * (alpha_j_new - alpha_j_old) * K[i, j]
                )
                b2 = (
                    b
                    - E_j
                    - y[i] * (alpha_i_new - alpha_i_old) * K[i, j]
                    - y[j] * (alpha_j_new - alpha_j_old) * K[j, j]
                )

                if 0.0 < alpha_i_new < self.C:
                    b = b1
                elif 0.0 < alpha_j_new < self.C:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)

                # Update alphas
                alphas[i] = alpha_i_new
                alphas[j] = alpha_j_new

                # Incrementally update f = K @ (alphas * y) + b
                delta_i = (alpha_i_new - alpha_i_old) * y[i]
                delta_j = (alpha_j_new - alpha_j_old) * y[j]
                if delta_i != 0.0:
                    f += delta_i * K[:, i]
                if delta_j != 0.0:
                    f += delta_j * K[:, j]
                # f stores only the sum term; b is applied when computing E or decision.

                num_changed += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0
            iters += 1

            if self.verbose and (iters % 5 == 0 or passes == 0):
                # quick training objective (hinge dual not shown); report SV count as proxy
                sv_count = int(np.sum(alphas > eps))
                print(f"SMO iter={iters} passes={passes} changed={num_changed} SVs~{sv_count}")

        return alphas, float(b)

    # ------------------------ Public API ------------------------
    def fit(self, X, y):
        """Train one-vs-rest kernel SVM.

        Args:
            X: np.ndarray, shape (n_samples, H, W) or (n_samples, n_features), values in [0,1]
            y: np.ndarray, shape (n_samples,), int labels [0..9]
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        elif X.ndim != 2:
            raise ValueError("X must be (n,H,W) or (n,n_features)")
        y = np.asarray(y, dtype=np.int32).ravel()
        n = X.shape[0]
        if y.shape[0] != n:
            raise ValueError("X and y must have the same number of samples")

        # Standardize features for more stable distances
        self.X_mean = X.mean(axis=0, dtype=np.float32)
        self.X_std = X.std(axis=0, dtype=np.float32)
        self.X_std[self.X_std < 1e-6] = 1.0
        X = (X - self.X_mean) / self.X_std

        self._X_train = X  # keep standardized train set during training

        # Auto-select gamma if None using median heuristic on a subset
        if self.kernel == "rbf" and (self.gamma is None):
            rng = np.random.RandomState(self.seed)
            m = min(n, 800)
            idx = rng.choice(n, m, replace=False)
            Xs = X[idx]
            Xn = np.sum(Xs * Xs, axis=1, keepdims=True)
            D = Xn + Xn.T - 2.0 * (Xs @ Xs.T)
            # Use median of upper triangle excluding zeros
            iu = np.triu_indices_from(D, k=1)
            dist = D[iu]
            med = np.median(dist)
            if med <= 0:
                med = np.mean(dist[dist > 0]) if np.any(dist > 0) else 1.0
            self.gamma = 1.0 / (2.0 * float(med))
            if self.verbose:
                print(f"Auto gamma set to {self.gamma:.6f} via median heuristic")

        # Precompute full kernel matrix once (shared by all one-vs-rest classifiers)
        if self.verbose:
            print(f"Precomputing {self.kernel} kernel matrix for {n} samples...")
        self._K = self._kernel_matrix(X)

        rng = np.random.RandomState(self.seed)
        self._cls_params = []

        for cls in range(self.n_classes):
            if self.verbose:
                print(f"Training OvR classifier for class {cls} ...")
            y_bin = np.where(y == cls, 1.0, -1.0).astype(np.float32)
            alphas, b = self._smo_train_binary(self._K, y_bin, rng)
            # Keep only support vectors to save memory and speed inference
            sv_mask = alphas > 1e-6
            sv_idx = np.nonzero(sv_mask)[0].astype(np.int32)
            alpha_y = (alphas[sv_mask] * y_bin[sv_mask]).astype(np.float32)
            sv_X = self._X_train[sv_idx].astype(np.float32, copy=False)
            # Precompute norms for faster RBF predict
            sv_norm = np.sum(sv_X * sv_X, axis=1).astype(np.float32, copy=False)
            self._cls_params.append({
                "sv_idx": sv_idx,
                "sv_X": sv_X,
                "sv_norm": sv_norm,
                "alpha_y": alpha_y,
                "b": float(b),
            })

        if self.verbose:
            total_svs = sum(p["sv_idx"].size for p in self._cls_params)
            print(f"Training finished. Total SVs across classes: {total_svs}")

    def predict(self, X):
        """Predict class labels for X.

        Args:
            X: (n_samples, n_features) or (n_samples, H, W)
        Returns:
            labels: (n_samples,)
        """
        if not self._cls_params:
            raise RuntimeError("Model is not trained. Call fit() first.")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        elif X.ndim != 2:
            raise ValueError("X must be (n,H,W) or (n,n_features)")

        # Apply standardization if available
        if self.X_mean is not None and self.X_std is not None:
            X = (X - self.X_mean) / self.X_std

        scores = np.empty((X.shape[0], self.n_classes), dtype=np.float32)
        if self.kernel == "rbf":
            # Precompute X norm once
            X_norm = np.sum(X * X, axis=1, keepdims=True)
            gamma = self._get_gamma(X)
            for cls, params in enumerate(self._cls_params):
                sv_X = params["sv_X"]
                alpha_y = params["alpha_y"]
                b = params["b"]
                if sv_X.size == 0:
                    scores[:, cls] = b
                    continue
                sv_norm = params["sv_norm"]  # (n_sv,)
                # dist^2 = ||x||^2 + ||sv||^2 - 2 x·sv
                G = X @ sv_X.T
                D = X_norm + sv_norm[None, :] - 2.0 * G
                K_new = np.exp(-gamma * np.clip(D, 0.0, None), dtype=np.float32)
                scores[:, cls] = K_new @ alpha_y + b
        else:
            for cls, params in enumerate(self._cls_params):
                sv_X = params["sv_X"]  # (n_sv, n_features)
                alpha_y = params["alpha_y"]  # shape (n_sv,)
                b = params["b"]
                if sv_X.size == 0:
                    scores[:, cls] = b
                else:
                    K_new = self._kernel_matrix(X, sv_X)
                    scores[:, cls] = K_new @ alpha_y + b

        return np.argmax(scores, axis=1)

    # ------------------------ Serialization ------------------------
    def save(self, path="svm_model.npz"):
        """Save trained model to an .npz file.

        The file contains hyperparameters and per-class support vectors.
        """
        if not self._cls_params:
            raise RuntimeError("Nothing to save: model not trained.")

        arrays = {}
        # meta info
        arrays['meta_kernel'] = np.array([self.kernel])
        arrays['meta_C'] = np.array([self.C], dtype=np.float32)
        arrays['meta_gamma'] = np.array([self.gamma if self.gamma is not None else -1.0], dtype=np.float32)
        arrays['meta_degree'] = np.array([self.degree], dtype=np.int32)
        arrays['meta_coef0'] = np.array([self.coef0], dtype=np.float32)
        arrays['meta_n_features'] = np.array([self.n_features], dtype=np.int32)
        arrays['meta_n_classes'] = np.array([self.n_classes], dtype=np.int32)
        # Save standardization params
        if self.X_mean is not None:
            arrays['X_mean'] = self.X_mean.astype(np.float32, copy=False)
        if self.X_std is not None:
            arrays['X_std'] = self.X_std.astype(np.float32, copy=False)

        for cls, params in enumerate(self._cls_params):
            arrays[f'sv_X_cls{cls}'] = params['sv_X']
            arrays[f'alpha_y_cls{cls}'] = params['alpha_y']
            arrays[f'b_cls{cls}'] = np.array([params['b']], dtype=np.float32)
            if 'sv_norm' in params:
                arrays[f'sv_norm_cls{cls}'] = params['sv_norm']

        np.savez(path, **arrays)
        if self.verbose:
            print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        """Load model from an .npz file produced by save()."""
        data = np.load(path, allow_pickle=False)
        kernel = str(data['meta_kernel'][0])
        C = float(data['meta_C'][0])
        gamma_raw = float(data['meta_gamma'][0])
        gamma = None if gamma_raw < 0 else gamma_raw
        degree = int(data['meta_degree'][0])
        coef0 = float(data['meta_coef0'][0])
        n_features = int(data['meta_n_features'][0])
        n_classes = int(data['meta_n_classes'][0])

        model = cls(
            n_features=n_features,
            n_classes=n_classes,
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            verbose=False,
        )
        model._cls_params = []
        # Load standardization params if present
        if 'X_mean' in data and 'X_std' in data:
            model.X_mean = data['X_mean'].astype(np.float32, copy=False)
            model.X_std = data['X_std'].astype(np.float32, copy=False)

        for cls_idx in range(n_classes):
            sv_X_key = f'sv_X_cls{cls_idx}'
            alpha_y_key = f'alpha_y_cls{cls_idx}'
            b_key = f'b_cls{cls_idx}'
            sv_X = data[sv_X_key]
            alpha_y = data[alpha_y_key]
            b = float(data[b_key][0])
            model._cls_params.append({
                'sv_idx': np.arange(sv_X.shape[0], dtype=np.int32),
                'sv_X': sv_X.astype(np.float32, copy=False),
                'alpha_y': alpha_y.astype(np.float32, copy=False),
                'b': b,
            })
            # Optional sv_norm for faster RBF prediction
            sv_norm_key = f'sv_norm_cls{cls_idx}'
            if sv_norm_key in data:
                model._cls_params[-1]['sv_norm'] = data[sv_norm_key].astype(np.float32, copy=False)
            else:
                model._cls_params[-1]['sv_norm'] = np.sum(sv_X.astype(np.float32, copy=False)**2, axis=1)
        return model
