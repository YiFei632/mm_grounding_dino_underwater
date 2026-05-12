"""FFT Band0 Low-Frequency Filter for Sonar Features.                                                                                                                                                
                                                                                                                                                                                                       
Extracts low-frequency semantic information from sonar features using FFT.                                                                                                                           
"""                                                                                                                                                                                                  
                                                                                                                                                                                                    
import torch                                                                                                                                                                                         
import torch.nn as nn                                                                                                                                                                                
import torch.fft as fft                                                                                                                                                                              
from typing import Optional                                                                                                                                                                          
                                                                                                                                                                                                    
                                                                                                                                                                                                    
class FFTBand0Filter(nn.Module):                                                                                                                                                                     
    """FFT-based low-frequency (Band0) filter.                                                                                                                                                       
                                                                                                                                                                                                    
    This module applies FFT to extract only the low-frequency components                                                                                                                             
    (Band0) from input features, which contain semantic information while                                                                                                                            
    filtering out high-frequency details and noise.                                                                                                                                                  
                                                                                                                                                                                                    
    Args:                                                                                                                                                                                            
        radius_ratio (float): Ratio of Band0 outer radius to max radius.                                                                                                                             
            Smaller values preserve only very low frequencies.                                                                                                                                       
            Recommended: 0.1-0.3 for semantic preservation.                                                                                                                                          
            Default: 0.2                                                                                                                                                                             
        enabled (bool): Whether to apply filtering. If False, acts as identity.                                                                                                                      
            Default: True                                                                                                                                                                            
    """                                                                                                                                                                                              
                                                                                                                                                                                                    
    def __init__(                                                                                                                                                                                    
        self,                                                                                                                                                                                        
        radius_ratio: float = 0.2,                                                                                                                                                                   
        enabled: bool = True,                                                                                                                                                                        
    ):                                                                                                                                                                                               
        super().__init__()                                                                                                                                                                           
        self.radius_ratio = float(radius_ratio)                                                                                                                                                      
        self.enabled = bool(enabled)                                                                                                                                                                 
                                                                                                                                                                                                    
        # Cache for radius grid (computed once per spatial size)                                                                                                                                     
        self._cached_mask: Optional[torch.Tensor] = None                                                                                                                                             
        self._cached_size: Optional[tuple] = None                                                                                                                                                    
                                                                                                                                                                                                    
    def _compute_band0_mask(                                                                                                                                                                         
        self,                                                                                                                                                                                        
        H: int,                                                                                                                                                                                      
        W: int,                                                                                                                                                                                      
        device: torch.device,                                                                                                                                                                        
        dtype: torch.dtype,                                                                                                                                                                          
    ) -> torch.Tensor:                                                                                                                                                                               
        """Compute Band0 frequency mask.                                                                                                                                                             
                                                                                                                                                                                                    
        Args:                                                                                                                                                                                        
            H, W: Spatial dimensions                                                                                                                                                                 
            device, dtype: Tensor properties                                                                                                                                                         
                                                                                                                                                                                                    
        Returns:                                                                                                                                                                                     
            mask: (H, W) binary mask, 1.0 for Band0 frequencies                                                                                                                                      
        """                                                                                                                                                                                          
        # Check cache                                                                                                                                                                                
        if (self._cached_mask is not None and                                                                                                                                                        
            self._cached_size == (H, W) and                                                                                                                                                          
            self._cached_mask.device == device and                                                                                                                                                   
            self._cached_mask.dtype == dtype):                                                                                                                                                       
            return self._cached_mask                                                                                                                                                                 
                                                                                                                                                                                                    
        # Compute radius grid                                                                                                                                                                        
        max_radius = min(H // 2, W // 2)                                                                                                                                                             
        band0_radius = max_radius * self.radius_ratio                                                                                                                                                
                                                                                                                                                                                                    
        cy, cx = H // 2, W // 2                                                                                                                                                                      
        y, x = torch.meshgrid(                                                                                                                                                                       
            torch.arange(H, device=device, dtype=dtype) - cy,                                                                                                                                        
            torch.arange(W, device=device, dtype=dtype) - cx,                                                                                                                                        
            indexing='ij'                                                                                                                                                                            
        )                                                                                                                                                                                            
        radius = torch.sqrt(y**2 + x**2)                                                                                                                                                             
                                                                                                                                                                                                    
        # Band0 mask: radius <= band0_radius                                                                                                                                                         
        mask = (radius <= band0_radius).to(dtype=dtype)                                                                                                                                              
                                                                                                                                                                                                    
        # Cache for reuse                                                                                                                                                                            
        self._cached_mask = mask                                                                                                                                                                     
        self._cached_size = (H, W)                                                                                                                                                                   
                                                                                                                                                                                                    
        return mask                                                                                                                                                                                  
                                                                                                                                                                                                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:                                                                                                                                              
        """Apply FFT Band0 filtering.                                                                                                                                                                
                                                                                                                                                                                                    
        Args:                                                                                                                                                                                        
            x: Input features, shape (B, C, H, W)                                                                                                                                                    
                                                                                                                                                                                                    
        Returns:                                                                                                                                                                                     
            x_band0: Low-frequency filtered features, shape (B, C, H, W)                                                                                                                             
        """                                                                                                                                                                                          
        if not self.enabled:                                                                                                                                                                         
            return x                                                                                                                                                                                 
                                                                                                                                                                                                    
        B, C, H, W = x.shape                                                                                                                                                                         
        device = x.device                                                                                                                                                                            
        dtype = x.dtype                                                                                                                                                                              
                                                                                                                                                                                                    
        # Use float32 for FFT computation                                                                                                                                                            
        working_dtype = torch.float32                                                                                                                                                                
        x_fp32 = x.to(dtype=working_dtype) if dtype != working_dtype else x                                                                                                                          
                                                                                                                                                                                                    
        # 1. FFT transform                                                                                                                                                                           
        x_fft = fft.fftshift(fft.fft2(x_fp32, dim=(-2, -1)), dim=(-2, -1))                                                                                                                           
                                                                                                                                                                                                    
        # 2. Generate Band0 mask                                                                                                                                                                     
        mask = self._compute_band0_mask(H, W, device, working_dtype)                                                                                                                                 
        mask = mask.view(1, 1, H, W)  # (1, 1, H, W) for broadcasting                                                                                                                                
                                                                                                                                                                                                    
        # 3. Apply mask (keep only Band0 frequencies)                                                                                                                                                
        x_filtered = x_fft * mask                                                                                                                                                                    
                                                                                                                                                                                                    
        # 4. Inverse FFT                                                                                                                                                                             
        x_band0 = fft.ifft2(                                                                                                                                                                         
            fft.ifftshift(x_filtered, dim=(-2, -1)),                                                                                                                                                 
            dim=(-2, -1)                                                                                                                                                                             
        ).real                                                                                                                                                                                       
                                                                                                                                                                                                    
        # Convert back to original dtype                                                                                                                                                             
        if x_band0.dtype != dtype:                                                                                                                                                                   
            x_band0 = x_band0.to(dtype=dtype)                                                                                                                                                        
                                                                                                                                                                                                    
        return x_band0