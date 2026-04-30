"""
Unit tests for anomaly detection module.
"""

import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.anomaly_detector import StatisticalAnomalyDetector, DetectionMethod, AnomalyResult

class TestStatisticalAnomalyDetector(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.normal_data = [1, 2, 3, 2, 1, 2, 3, 2, 1]
        # Create a clear outlier that's far from the mean
        # Data: mostly around 50, one value at 200
        self.data_with_outlier = [48, 52, 49, 51, 200, 50, 48, 52, 49]
        self.empty_data = []
        # Data with subtle anomaly
        self.subtle_anomaly = [10, 10, 10, 10, 100, 10, 10, 10, 10]
        
    def test_zscore_detection(self):
        """Test Z-score method identifies outliers correctly"""
        # Lower threshold to catch the outlier
        detector = StatisticalAnomalyDetector(method=DetectionMethod.ZSCORE, threshold=2.5)
        result = detector.detect(self.data_with_outlier)
        
        self.assertIsInstance(result, AnomalyResult)
        self.assertEqual(result.method, "zscore")
        self.assertTrue(result.num_anomalies >= 1)  # Should find at least 1 outlier
        # The outlier should be at index 4 (the value 200)
        self.assertIn(4, result.indices)
        
    def test_iqr_detection(self):
        """Test IQR method identifies outliers"""
        detector = StatisticalAnomalyDetector(method=DetectionMethod.IQR, threshold=1.5)
        result = detector.detect(self.data_with_outlier)
        
        self.assertEqual(result.method, "iqr")
        self.assertGreater(result.num_anomalies, 0)
        self.assertIn(4, result.indices)  # Should catch the 200 value
        
    def test_mad_detection(self):
        """Test MAD method (robust to outliers)"""
        detector = StatisticalAnomalyDetector(method=DetectionMethod.MAD, threshold=3.0)
        result = detector.detect(self.data_with_outlier)
        
        self.assertEqual(result.method, "mad")
        self.assertGreater(result.num_anomalies, 0)
        
    def test_empty_data(self):
        """Test handling of empty input"""
        detector = StatisticalAnomalyDetector()
        result = detector.detect(self.empty_data)
        
        self.assertEqual(result.num_anomalies, 0)
        self.assertEqual(result.indices, [])
        
    def test_summary_method(self):
        """Test result summary generation"""
        detector = StatisticalAnomalyDetector()
        result = detector.detect(self.data_with_outlier)
        summary = result.summary()
        
        self.assertIn('method', summary)
        self.assertIn('total_anomalies', summary)
        self.assertIn('anomaly_rate', summary)
        
    def test_constant_data(self):
        """Test detection on constant data (no variance)"""
        constant_data = [5, 5, 5, 5, 5]
        detector = StatisticalAnomalyDetector()
        result = detector.detect(constant_data)
        
        # Should detect no anomalies when all values are equal
        self.assertEqual(result.num_anomalies, 0)
        
    def test_extreme_outlier(self):
        """Test detection of extreme outlier with default threshold"""
        extreme_data = [1, 2, 1, 2, 1, 2, 1000, 1, 2]
        detector = StatisticalAnomalyDetector(method=DetectionMethod.ZSCORE, threshold=3.0)
        result = detector.detect(extreme_data)
        
        # Default threshold should catch this extreme outlier
        self.assertGreater(result.num_anomalies, 0)
        
    def test_different_thresholds(self):
        """Test sensitivity to threshold changes"""
        data = [1, 2, 1, 2, 20, 1, 2, 1, 2]
        
        # Strict threshold (high)
        detector_strict = StatisticalAnomalyDetector(method=DetectionMethod.ZSCORE, threshold=3.0)
        result_strict = detector_strict.detect(data)
        
        # Lenient threshold (low)
        detector_lenient = StatisticalAnomalyDetector(method=DetectionMethod.ZSCORE, threshold=1.5)
        result_lenient = detector_lenient.detect(data)
        
        # Lenient should find more anomalies
        self.assertGreaterEqual(result_lenient.num_anomalies, result_strict.num_anomalies)
        
    def test_numpy_array_input(self):
        """Test that numpy arrays work as input"""
        import numpy as np
        data = np.array([1, 2, 3, 100, 4, 5])
        detector = StatisticalAnomalyDetector(method=DetectionMethod.ZSCORE, threshold=2.5)
        result = detector.detect(data)
        
        self.assertGreater(result.num_anomalies, 0)
        
    def test_edge_case_single_value(self):
        """Test single value input"""
        detector = StatisticalAnomalyDetector()
        result = detector.detect([42])
        
        self.assertEqual(result.num_anomalies, 0)
        
    def test_all_identical_large_values(self):
        """Test many identical values"""
        large_identical = [100] * 1000
        detector = StatisticalAnomalyDetector()
        result = detector.detect(large_identical)
        
        self.assertEqual(result.num_anomalies, 0)

if __name__ == '__main__':
    unittest.main()
