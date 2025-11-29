"""
Sustainability Tracking Module
Monitors environmental impact of data processing operations
"""

import time
import json
from datetime import datetime
from typing import Dict


class SustainabilityTracker:
    """Track environmental impact of data processing"""
    
    def __init__(self, region: str = 'global_average'):
        """
        Initialize sustainability tracker
        
        Args:
            region: Energy grid region for carbon intensity calculations
        """
        # Energy consumption metrics (kWh)
        self.energy_metrics = {
            'text_processing_1mb': 0.00005,  # kWh per MB
            'inference_1k_tokens': 0.000004,  # kWh per 1000 tokens
            'model_training_1epoch': 0.5,     # kWh per epoch
        }
        
        # Carbon intensity by region (kg CO2 per kWh)
        self.carbon_intensity = {
            'global_average': 0.475,
            'us_average': 0.386,
            'eu_average': 0.295,
            'renewable': 0.05,
        }
        
        # Water usage (liters per kWh)
        self.water_per_kwh = 1.8
        
        self.current_region = region
        
        # Session tracking
        self.session_metrics = {
            'total_energy_kwh': 0.0,
            'total_carbon_kg': 0.0,
            'total_water_liters': 0.0,
            'operations_count': 0,
            'data_processed_mb': 0.0,
            'data_saved_mb': 0.0,
            'start_time': time.time(),
        }
    
    def calculate_data_size_mb(self, data) -> float:
        """
        Calculate data size in MB
        
        Args:
            data: Data to measure (str, dict, or other)
            
        Returns:
            Size in MB
        """
        if isinstance(data, str):
            return len(data.encode('utf-8')) / (1024 * 1024)
        elif isinstance(data, dict):
            return len(json.dumps(data).encode('utf-8')) / (1024 * 1024)
        elif isinstance(data, (list, tuple)):
            return sum(self.calculate_data_size_mb(item) for item in data)
        else:
            return len(str(data).encode('utf-8')) / (1024 * 1024)
    
    def track_operation(self, 
                       operation_type: str,
                       data_size_mb: float = 0.0,
                       tokens: int = 0,
                       epochs: int = 0) -> Dict:
        """
        Track a data processing operation
        
        Args:
            operation_type: Type of operation (text_processing, inference, training)
            data_size_mb: Data size in MB
            tokens: Number of tokens processed
            epochs: Number of training epochs
            
        Returns:
            Dict with operation metrics
        """
        energy_kwh = 0.0
        
        # Calculate energy based on operation type
        if operation_type == 'text_processing' and data_size_mb > 0:
            energy_kwh = data_size_mb * self.energy_metrics['text_processing_1mb']
        elif operation_type == 'inference' and tokens > 0:
            energy_kwh = (tokens / 1000) * self.energy_metrics['inference_1k_tokens']
        elif operation_type == 'training' and epochs > 0:
            energy_kwh = epochs * self.energy_metrics['model_training_1epoch']
        
        # Calculate environmental impact
        carbon_kg = energy_kwh * self.carbon_intensity[self.current_region]
        water_liters = energy_kwh * self.water_per_kwh
        
        # Update session metrics
        self.session_metrics['total_energy_kwh'] += energy_kwh
        self.session_metrics['total_carbon_kg'] += carbon_kg
        self.session_metrics['total_water_liters'] += water_liters
        self.session_metrics['operations_count'] += 1
        self.session_metrics['data_processed_mb'] += data_size_mb
        
        return {
            'timestamp': datetime.now().isoformat(),
            'operation_type': operation_type,
            'energy_kwh': energy_kwh,
            'carbon_kg': carbon_kg,
            'water_liters': water_liters,
            'data_size_mb': data_size_mb,
        }
    
    def calculate_savings(self, 
                         original_data_mb: float,
                         optimized_data_mb: float) -> Dict:
        """
        Calculate savings from data optimization
        
        Args:
            original_data_mb: Original data size
            optimized_data_mb: Optimized data size
            
        Returns:
            Dict with savings metrics
        """
        data_saved_mb = original_data_mb - optimized_data_mb
        
        # Calculate immediate savings
        energy_saved = data_saved_mb * self.energy_metrics['text_processing_1mb']
        carbon_saved = energy_saved * self.carbon_intensity[self.current_region]
        water_saved = energy_saved * self.water_per_kwh
        
        # Annual projections (assuming daily processing)
        annual_energy = energy_saved * 365
        annual_carbon = carbon_saved * 365
        annual_water = water_saved * 365
        
        # Equivalencies
        trees_equivalent = annual_carbon / 21  # 1 tree absorbs ~21kg CO2/year
        car_miles = annual_carbon / 0.000404   # Average car emits 404g CO2/mile
        
        # Update session metrics
        self.session_metrics['data_saved_mb'] += data_saved_mb
        
        return {
            'immediate_savings': {
                'data_mb': data_saved_mb,
                'reduction_percentage': (data_saved_mb / original_data_mb * 100) if original_data_mb > 0 else 0,
                'energy_kwh': energy_saved,
                'carbon_kg': carbon_saved,
                'water_liters': water_saved,
            },
            'projected_annual_savings': {
                'energy_kwh': annual_energy,
                'carbon_kg': annual_carbon,
                'water_liters': annual_water,
                'carbon_trees_equivalent': trees_equivalent,
                'car_miles_equivalent': car_miles,
            }
        }
    
    def get_session_summary(self) -> Dict:
        """
        Get summary of current session
        
        Returns:
            Dict with session statistics
        """
        carbon_kg = self.session_metrics['total_carbon_kg']
        energy_kwh = self.session_metrics['total_energy_kwh']
        
        return {
            'session_duration_minutes': (time.time() - self.session_metrics['start_time']) / 60,
            'total_operations': self.session_metrics['operations_count'],
            'data_processed_mb': self.session_metrics['data_processed_mb'],
            'data_saved_mb': self.session_metrics['data_saved_mb'],
            'environmental_impact': {
                'energy_kwh': energy_kwh,
                'carbon_kg': carbon_kg,
                'water_liters': self.session_metrics['total_water_liters'],
            },
            'equivalencies': {
                'trees_planted': carbon_kg / 21,
                'car_miles': carbon_kg / 0.000404,
                'smartphone_charges': energy_kwh / 0.012,
                'led_bulbs_1hour': energy_kwh / 0.01,
            },
            'region': self.current_region,
        }
    
    def reset_session(self):
        """Reset session metrics"""
        self.session_metrics = {
            'total_energy_kwh': 0.0,
            'total_carbon_kg': 0.0,
            'total_water_liters': 0.0,
            'operations_count': 0,
            'data_processed_mb': 0.0,
            'data_saved_mb': 0.0,
            'start_time': time.time(),
        }
    
    def change_region(self, region: str):
        """
        Change energy grid region
        
        Args:
            region: New region name
        """
        if region in self.carbon_intensity:
            self.current_region = region
        else:
            raise ValueError(f"Unknown region: {region}. Available: {list(self.carbon_intensity.keys())}")


if __name__ == "__main__":
    # Example usage
    tracker = SustainabilityTracker()
    
    print("=" * 70)
    print("SUSTAINABILITY TRACKER TEST")
    print("=" * 70)
    
    # Simulate data processing
    print("\n1. Processing 100 MB of data...")
    op1 = tracker.track_operation('text_processing', data_size_mb=100)
    print(f"   Energy: {op1['energy_kwh']:.6f} kWh")
    print(f"   Carbon: {op1['carbon_kg']:.6f} kg CO2")
    
    # Simulate data optimization
    print("\n2. Data optimization: 100 MB → 65 MB")
    savings = tracker.calculate_savings(100, 65)
    print(f"   Data saved: {savings['immediate_savings']['data_mb']:.2f} MB")
    print(f"   Reduction: {savings['immediate_savings']['reduction_percentage']:.1f}%")
    print(f"   Annual carbon saved: {savings['projected_annual_savings']['carbon_kg']:.3f} kg")
    print(f"   Trees equivalent: {savings['projected_annual_savings']['carbon_trees_equivalent']:.2f}")
    
    # Session summary
    print("\n3. Session Summary:")
    summary = tracker.get_session_summary()
    print(f"   Operations: {summary['total_operations']}")
    print(f"   Total energy: {summary['environmental_impact']['energy_kwh']:.6f} kWh")
    print(f"   Total carbon: {summary['environmental_impact']['carbon_kg']:.6f} kg CO2")
    print(f"   Smartphone charges: {summary['equivalencies']['smartphone_charges']:.1f}")
