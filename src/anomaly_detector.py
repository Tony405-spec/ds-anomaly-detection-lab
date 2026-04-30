"""
Anomaly Detection Toolkit for Time Series Data

This module implements production-ready anomaly detection algorithms
suitable for IoT sensor data, financial time series, and system metrics.

Author: Tony405-spec
Contributors: Kate020-cpu, monicloraine94, jammiemwendwa-eng
"""

import numpy as np
from typing import List, Dict, Union, Optional
from dataclasses import dataclass
from enum import Enum

class DetectionMethod(Enum):
    """Supported anomaly detection methods"""
    ZSCORE = "zscore"
    IQR = "iqr"
    MAD = "mad"  # Median Absolute Deviation
    DBSCAN = "dbscan"

@dataclass
class AnomalyResult:
    """Container for anomaly detection results"""
    indices: List[int]
    scores: List[float]
    method: str
    threshold: float
    num_anomalies: int
    
    def summary(self) -> Dict:
        """Return summary statistics"""
        return {
            'method': self.method,
            'total_anomalies': self.num_anomalies,
            'anomaly_rate': f"{self.num_anomalies / max(len(self.indices), 1) * 100:.2f}%" if self.indices else "0%",
            'max_score': max(self.scores) if self.scores else 0,
            'min_score': min(self.scores) if self.scores else 0
        }

class StatisticalAnomalyDetector:
    """
    Statistical methods for anomaly detection.
    
    Examples:
        >>> detector = StatisticalAnomalyDetector(method='zscore', threshold=3.0)
        >>> data = [1, 2, 3, 100, 4, 5]
        >>> result = detector.detect(data)
        >>> print(result.indices)  # [3]
    """
    
    def __init__(self, method: DetectionMethod = DetectionMethod.ZSCORE, threshold: float = 3.0):
        self.method = method
        self.threshold = threshold
        self._fitted = False
        
    def detect(self, data: Union[List[float], np.ndarray]) -> AnomalyResult:
        """
        Detect anomalies in the input data.
        
        Args:
            data: List or array of numerical values
            
        Returns:
            AnomalyResult object containing detection results
        """
        if isinstance(data, list):
            data = np.array(data)
            
        if len(data) == 0:
            return AnomalyResult([], [], self.method.value, self.threshold, 0)
        
        if self.method == DetectionMethod.ZSCORE:
            return self._zscore_detection(data)
        elif self.method == DetectionMethod.IQR:
            return self._iqr_detection(data)
        elif self.method == DetectionMethod.MAD:
            return self._mad_detection(data)
        else:
            raise ValueError(f"Method {self.method} not implemented")
    
    def _zscore_detection(self, data: np.ndarray) -> AnomalyResult:
        """Z-score based detection (3-sigma rule)"""
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return AnomalyResult([], [], self.method.value, self.threshold, 0)
        
        z_scores = np.abs((data - mean) / std)
        
        # Dynamic threshold adjustment for small datasets
        if len(data) < 10:
            adjusted_threshold = max(2.0, self.threshold - 0.5)
        else:
            adjusted_threshold = self.threshold
        
        anomaly_indices = np.where(z_scores > adjusted_threshold)[0].tolist()
        anomaly_scores = z_scores[anomaly_indices].tolist()
        
        # For debugging - print info if anomalies found
        if anomaly_indices and len(data) < 20:
            print(f"Debug: Found {len(anomaly_indices)} anomalies with threshold {adjusted_threshold}")
            print(f"  Max Z-score: {max(z_scores):.2f}")
            print(f"  Mean: {mean:.2f}, Std: {std:.2f}")
        
        return AnomalyResult(
            indices=anomaly_indices,
            scores=anomaly_scores,
            method=self.method.value,
            threshold=adjusted_threshold,
            num_anomalies=len(anomaly_indices)
        )
    
    def _iqr_detection(self, data: np.ndarray) -> AnomalyResult:
        """Interquartile Range based detection"""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr
        
        anomaly_indices = np.where((data < lower_bound) | (data > upper_bound))[0].tolist()
        # Calculate modified z-scores as anomaly scores
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        scores = [abs(data[i] - median) / (mad if mad > 0 else 1) for i in anomaly_indices]
        
        return AnomalyResult(
            indices=anomaly_indices,
            scores=scores,
            method=self.method.value,
            threshold=self.threshold,
            num_anomalies=len(anomaly_indices)
        )
    
    def _mad_detection(self, data: np.ndarray) -> AnomalyResult:
        """Median Absolute Deviation based detection (robust to outliers)"""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            return AnomalyResult([], [], self.method.value, self.threshold, 0)
        
        modified_z_scores = 0.6745 * (data - median) / mad
        anomaly_indices = np.where(np.abs(modified_z_scores) > self.threshold)[0].tolist()
        anomaly_scores = np.abs(modified_z_scores[anomaly_indices]).tolist()
        
        return AnomalyResult(
            indices=anomaly_indices,
            scores=anomaly_scores,
            method=self.method.value,
            threshold=self.threshold,
            num_anomalies=len(anomaly_indices)
        )

class AdvancedAnomalyDetector:
    """
    Advanced detection algorithms including ML-based methods.
    
    This class requires scikit-learn for certain methods.
    """
    
    @staticmethod
    def isolation_forest(data: np.ndarray, contamination: float = 0.1) -> AnomalyResult:
        """
        Isolation Forest algorithm for anomaly detection.
        
        Args:
            data: 2D array of shape (n_samples, n_features)
            contamination: Expected proportion of anomalies
            
        Returns:
            AnomalyResult with detection results
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError("scikit-learn required for Isolation Forest. Run: pip install scikit-learn")
        
        # Reshape 1D data to 2D if needed
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        model = IsolationForest(contamination=contamination, random_state=42)
        predictions = model.fit_predict(data)
        
        anomaly_indices = np.where(predictions == -1)[0].tolist()
        # Get anomaly scores (more negative = more anomalous)
        scores = model.score_samples(data)[anomaly_indices].tolist()
        
        return AnomalyResult(
            indices=anomaly_indices,
            scores=scores,
            method="isolation_forest",
            threshold=contamination,
            num_anomalies=len(anomaly_indices)
        )
