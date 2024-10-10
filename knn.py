from pathlib import Path
import dill as pickle
import numpy as np
from usearch.index import Index
from sklearn.preprocessing import StandardScaler

class UsearchKNeighborsRegressor:
    def __init__(self, k: int = 5):
        self.index = None  # 初期状態では Index は作成しない
        self.k = k
        self.y = None
        self.scaler = StandardScaler()
        self.ndim = None  # ndim は初期化時には設定しない

    def fit(self, X: np.ndarray, y: np.ndarray):
        """データをインデックスに追加し、ターゲット値を保存"""
        self.ndim = X.shape[1]  # X から次元数を計算
        self.index = Index(ndim=self.ndim)  # n_dim に基づいて Index を初期化

        # スケーリング
        X_scaled = self.scaler.fit_transform(X)
        self.index.add(np.arange(len(X_scaled)), X_scaled)
        self.y = np.array(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """与えられたデータに対して予測を行う"""
        X_scaled = self.scaler.transform(X)
        results = self.index.search(X_scaled, self.k)
        neighbors_y = self.y[results.keys]

        return np.mean(neighbors_y, axis=1)
    
    def save(self, model_path: str):
        """index、scaler、モデルパラメータ（y, k, ndim）をpickleで保存"""
        model_path = Path(model_path)
        index_path = model_path.with_suffix(".index")
        
        # インデックスをバイナリ形式で保存
        self.index.save(str(index_path))  # usearch は文字列パスを期待するため str に変換
        
        # y, k, scaler, ndim をpickleで保存し、インデックスのパスを一緒に保存
        with model_path.open('wb') as f:
            pickle.dump({
                'index_path': str(index_path),
                'y': self.y,
                'k': self.k,
                'scaler': self.scaler,
                'ndim': self.ndim  # n_dim を保存
            }, f)

    def load(self, model_path: str):
        """index、scaler、モデルパラメータ（y, k, ndim）をpickleで読み込む"""
        model_path = Path(model_path)
        
        # pickleファイルからy, k, scaler, index_pathを読み込む
        with model_path.open('rb') as f:
            model_data = pickle.load(f)
            self.y = model_data['y']
            self.k = model_data['k']
            scaler = model_data['scaler']
            self.ndim = model_data['ndim']  # n_dim を復元

            # Indexを復元するために、ndimを使って再初期化
            self.index = Index(ndim=self.ndim)
            self.index.load(model_data['index_path'])
        
        self.scaler = scaler  # scalerを復元
        return self.scaler  # scaler を返す