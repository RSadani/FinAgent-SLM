import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import os
import time
from loguru import logger
import pandas as pd

class Quantizer:
    """Model Quantization Utilities for FinAgent-SLM."""
    
    @staticmethod
    def dynamic_quantize_int8(model: nn.Module) -> nn.Module:
        """Applies INT8 dynamic quantization to Linear layers."""
        logger.info("Applying INT8 dynamic quantization...")
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        return quantized_model

    @staticmethod
    def static_quantize(model: nn.Module, calibration_loader: Any) -> nn.Module:
        """Applies static quantization using calibration data."""
        logger.info("Applying static quantization with calibration...")
        model.eval()
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        torch.ao.quantization.prepare(model, inplace=True)
        
        # Calibrate
        with torch.no_grad():
            for batch in calibration_loader:
                model(batch['features'])
                
        torch.ao.quantization.convert(model, inplace=True)
        return model

    @staticmethod
    def simulate_gptq(model: nn.Module, bits: int = 4) -> nn.Module:
        """Simulates GPTQ-style weight-only quantization."""
        logger.info(f"Simulating GPTQ {bits}-bit quantization...")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                scale = weight.abs().max() / ((2 ** (bits - 1)) - 1)
                quantized = torch.round(weight / scale)
                dequantized = quantized * scale
                module.weight.data = dequantized
        return model

    @staticmethod
    def simulate_awq(model: nn.Module, bits: int = 4) -> nn.Module:
        """Simulates Activation-aware Weight Quantization (AWQ)."""
        logger.info(f"Simulating AWQ {bits}-bit quantization...")
        # Placeholder for actual AWQ implementation: scale weights based on activation magnitude
        return Quantizer.simulate_gptq(model, bits)

    @staticmethod
    def export_to_onnx(model: nn.Module, dummy_input: torch.Tensor, save_path: str):
        """Exports the model to ONNX format."""
        logger.info(f"Exporting model to ONNX: {save_path}")
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['fraud_logits', 'fraud_type_logits', 'risk_score'],
            dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'}}
        )


class QuantizationMatrix:
    """Benchmarks different quantization strategies."""
    
    def __init__(self, model: nn.Module, dummy_input: torch.Tensor):
        self.base_model = model
        self.dummy_input = dummy_input
        self.results = []

    def _get_model_size_mb(self, model: nn.Module) -> float:
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def _measure_latency(self, model: nn.Module, runs: int = 100) -> float:
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(runs):
                model(self.dummy_input)
        return ((time.time() - start_time) / runs) * 1000  # ms

    def benchmark_all(self) -> pd.DataFrame:
        logger.info("Starting quantization benchmarking suite...")
        
        # Base Model (FP32)
        base_size = self._get_model_size_mb(self.base_model)
        base_latency = self._measure_latency(self.base_model)
        self.results.append({
            'Method': 'FP32 (Base)',
            'Size (MB)': base_size,
            'Latency (ms)': base_latency,
            'Size Ratio': 1.0
        })
        
        import copy
        
        # Dynamic INT8
        int8_model = Quantizer.dynamic_quantize_int8(copy.deepcopy(self.base_model))
        int8_size = self._get_model_size_mb(int8_model)
        int8_latency = self._measure_latency(int8_model)
        self.results.append({
            'Method': 'Dynamic INT8',
            'Size (MB)': int8_size,
            'Latency (ms)': int8_latency,
            'Size Ratio': int8_size / base_size
        })
        
        # GPTQ 4-bit Simulation
        gptq_model = Quantizer.simulate_gptq(copy.deepcopy(self.base_model), bits=4)
        gptq_latency = self._measure_latency(gptq_model)
        self.results.append({
            'Method': 'GPTQ 4-bit (Sim)',
            'Size (MB)': base_size / 8,  # Theoretical max compression from fp32
            'Latency (ms)': gptq_latency,
            'Size Ratio': (base_size / 8) / base_size
        })

        # AWQ 4-bit Simulation
        awq_model = Quantizer.simulate_awq(copy.deepcopy(self.base_model), bits=4)
        awq_latency = self._measure_latency(awq_model)
        self.results.append({
            'Method': 'AWQ 4-bit (Sim)',
            'Size (MB)': base_size / 8, 
            'Latency (ms)': awq_latency,
            'Size Ratio': (base_size / 8) / base_size
        })

        df = pd.DataFrame(self.results)
        logger.info("\n" + str(df))
        return df
