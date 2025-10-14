import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

class AtmosphericCorrection:
    """Atmospheric correction for thermal imagery."""
    
    def __init__(self, 
                 sensor_params: Dict,
                 atmospheric_params: Optional[Dict] = None):
        """
        Args:
            sensor_params: Dictionary with sensor-specific parameters
                - wavelength: Central wavelength (μm)
                - bandwidth: Spectral bandwidth (μm)
                - calibration_constants: [K1, K2] for Planck function
            atmospheric_params: Atmospheric parameters
                - transmittance: Atmospheric transmittance [0-1]
                - upwelling_radiance: Upwelling atmospheric radiance
                - downwelling_radiance: Downwelling atmospheric radiance
        """
        self.sensor_params = sensor_params
        self.atmospheric_params = atmospheric_params or {
            'transmittance': 0.85,
            'upwelling_radiance': 2.5,
            'downwelling_radiance': 3.2
        }
        
        # Physical constants
        self.h = 6.626e-34  # Planck constant
        self.c = 2.998e8    # Speed of light
        self.k = 1.381e-23  # Boltzmann constant
        
    def radiance_to_temperature(self, 
                               radiance: torch.Tensor,
                               emissivity: torch.Tensor = None) -> torch.Tensor:
        """Convert radiance to brightness temperature using Planck's law."""
        if emissivity is None:
            emissivity = torch.ones_like(radiance)
        
        # Sensor calibration constants
        K1 = self.sensor_params['calibration_constants'][0]
        K2 = self.sensor_params['calibration_constants'][1]
        
        # Planck function inversion
        temperature = K2 / torch.log(K1 / radiance + 1)
        
        return temperature
    
    def temperature_to_radiance(self, 
                               temperature: torch.Tensor,
                               emissivity: torch.Tensor = None) -> torch.Tensor:
        """Convert temperature to radiance using Planck's law."""
        if emissivity is None:
            emissivity = torch.ones_like(temperature)
        
        # Sensor calibration constants
        K1 = self.sensor_params['calibration_constants'][0]
        K2 = self.sensor_params['calibration_constants'][1]
        
        # Planck function
        radiance = K1 / (torch.exp(K2 / temperature) - 1)
        
        return radiance * emissivity
    
    def atmospheric_correction_forward(self, 
                                     surface_radiance: torch.Tensor,
                                     emissivity: torch.Tensor) -> torch.Tensor:
        """Apply atmospheric effects to surface radiance."""
        tau = self.atmospheric_params['transmittance']
        Lu = self.atmospheric_params['upwelling_radiance']
        Ld = self.atmospheric_params['downwelling_radiance']
        
        # At-sensor radiance equation
        sensor_radiance = (tau * emissivity * surface_radiance + 
                          tau * (1 - emissivity) * Ld + Lu)
        
        return sensor_radiance
    
    def atmospheric_correction_inverse(self, 
                                     sensor_radiance: torch.Tensor,
                                     emissivity: torch.Tensor) -> torch.Tensor:
        """Remove atmospheric effects from sensor radiance."""
        tau = self.atmospheric_params['transmittance']
        Lu = self.atmospheric_params['upwelling_radiance']
        Ld = self.atmospheric_params['downwelling_radiance']
        
        # Solve for surface radiance
        surface_radiance = ((sensor_radiance - Lu - tau * (1 - emissivity) * Ld) / 
                           (tau * emissivity))
        
        return surface_radiance
    
    def surface_temperature_from_sensor(self, 
                                       sensor_radiance: torch.Tensor,
                                       emissivity: torch.Tensor) -> torch.Tensor:
        """Convert sensor radiance to surface temperature."""
        # Remove atmospheric effects
        surface_radiance = self.atmospheric_correction_inverse(sensor_radiance, emissivity)
        
        # Convert to temperature
        surface_temperature = self.radiance_to_temperature(surface_radiance, emissivity)
        
        return surface_temperature

class EmissivityHandler:
    """Handle emissivity maps and class-wise emissivity tables."""
    
    def __init__(self, emissivity_table: Dict[int, float]):
        """
        Args:
            emissivity_table: Mapping from land cover class to emissivity value
        """
        self.emissivity_table = emissivity_table
        
    def get_emissivity_map(self, 
                          land_cover_mask: torch.Tensor) -> torch.Tensor:
        """Generate emissivity map from land cover classification."""
        emissivity_map = torch.zeros_like(land_cover_mask, dtype=torch.float32)
        
        for class_id, emissivity_value in self.emissivity_table.items():
            mask = (land_cover_mask == class_id)
            emissivity_map[mask] = emissivity_value
            
        return emissivity_map
    
    def estimate_emissivity_from_ndvi(self, 
                                     ndvi: torch.Tensor,
                                     soil_emissivity: float = 0.92,
                                     vegetation_emissivity: float = 0.98) -> torch.Tensor:
        """Estimate emissivity from NDVI using vegetation fraction method."""
        # Vegetation fraction from NDVI
        ndvi_soil = 0.2
        ndvi_veg = 0.8
        
        vegetation_fraction = torch.clamp(
            (ndvi - ndvi_soil) / (ndvi_veg - ndvi_soil), 0, 1
        )
        
        # Linear mixing of soil and vegetation emissivity
        emissivity = (soil_emissivity * (1 - vegetation_fraction) + 
                     vegetation_emissivity * vegetation_fraction)
        
        return emissivity

class PhysicsConsistencyLoss(nn.Module):
    """Physics-based consistency losses for thermal SR."""
    
    def __init__(self, 
                 atmospheric_corrector: AtmosphericCorrection,
                 emissivity_handler: EmissivityHandler):
        super().__init__()
        self.atmospheric_corrector = atmospheric_corrector
        self.emissivity_handler = emissivity_handler
        
    def temperature_bias_loss(self, 
                             sr_temperature: torch.Tensor,
                             reference_temperature: torch.Tensor,
                             roi_mask: torch.Tensor = None) -> torch.Tensor:
        """Compute temperature bias penalty over regions of interest."""
        if roi_mask is not None:
            sr_masked = sr_temperature * roi_mask
            ref_masked = reference_temperature * roi_mask
            bias = torch.mean(torch.abs(sr_masked - ref_masked))
        else:
            bias = torch.mean(torch.abs(sr_temperature - reference_temperature))
        
        return bias
    
    def radiance_consistency_loss(self, 
                                 sr_temperature: torch.Tensor,
                                 emissivity_map: torch.Tensor,
                                 reference_radiance: torch.Tensor) -> torch.Tensor:
        """Compute radiance consistency loss."""
        # Convert SR temperature to radiance
        sr_radiance = self.atmospheric_corrector.temperature_to_radiance(
            sr_temperature, emissivity_map
        )
        
        # L1 loss in radiance domain
        radiance_loss = torch.mean(torch.abs(sr_radiance - reference_radiance))
        
        return radiance_loss
    
    def energy_balance_loss(self, 
                           sr_temperature: torch.Tensor,
                           homogeneous_mask: torch.Tensor) -> torch.Tensor:
        """Energy balance constraint for homogeneous regions."""
        if homogeneous_mask is None:
            return torch.tensor(0.0, device=sr_temperature.device)
        
        # Compute spatial gradients in homogeneous regions
        grad_x = torch.abs(sr_temperature[:, :, :, 1:] - sr_temperature[:, :, :, :-1])
        grad_y = torch.abs(sr_temperature[:, :, 1:, :] - sr_temperature[:, :, :-1, :])
        
        # Apply mask (exclude boundaries)
        mask_x = homogeneous_mask[:, :, :, 1:] * homogeneous_mask[:, :, :, :-1]
        mask_y = homogeneous_mask[:, :, 1:, :] * homogeneous_mask[:, :, :-1, :]
        
        # Penalize large gradients in homogeneous regions
        energy_loss = (torch.sum(grad_x * mask_x) + torch.sum(grad_y * mask_y)) / (
            torch.sum(mask_x) + torch.sum(mask_y) + 1e-8
        )
        
        return energy_loss
    
    def forward(self, 
                sr_temperature: torch.Tensor,
                reference_temperature: torch.Tensor,
                emissivity_map: torch.Tensor,
                reference_radiance: torch.Tensor = None,
                roi_mask: torch.Tensor = None,
                homogeneous_mask: torch.Tensor = None,
                weights: Dict[str, float] = None) -> Dict[str, torch.Tensor]:
        """Compute all physics consistency losses."""
        if weights is None:
            weights = {'bias': 1.0, 'radiance': 0.5, 'energy': 0.1}
        
        losses = {}
        
        # Temperature bias loss
        losses['bias'] = self.temperature_bias_loss(
            sr_temperature, reference_temperature, roi_mask
        )
        
        # Radiance consistency loss
        if reference_radiance is not None:
            losses['radiance'] = self.radiance_consistency_loss(
                sr_temperature, emissivity_map, reference_radiance
            )
        else:
            losses['radiance'] = torch.tensor(0.0, device=sr_temperature.device)
        
        # Energy balance loss
        losses['energy'] = self.energy_balance_loss(sr_temperature, homogeneous_mask)
        
        # Weighted total
        total_loss = sum(weights[k] * losses[k] for k in losses.keys())
        losses['total'] = total_loss
        
        return losses