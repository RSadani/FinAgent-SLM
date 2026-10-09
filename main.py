#!/usr/bin/env python3
"""
FinAgent-SLM: Autonomous Agentic AI for Real-Time Tax Fraud Detection
and Audit Trail Generation

Main entry point for the FinAgent-SLM system.
Supports multiple execution modes: train, evaluate, serve, demo, and UI.

Ministry of Finance & Banking PSUs
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FinAgent-SLM: Tax Fraud Detection & Audit Trail System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode demo          Run with synthetic data demo
  python main.py --mode train         Train the fraud detection model
  python main.py --mode evaluate      Evaluate model performance
  python main.py --mode ui            Launch Streamlit dashboard
  python main.py --mode agent         Run multi-agent orchestration
  python main.py --mode quantize      Run quantization benchmarks
  python main.py --mode full          Run complete pipeline
        """
    )
    parser.add_argument(
        "--mode", type=str, default="demo",
        choices=["demo", "train", "evaluate", "ui", "agent", "quantize", "full"],
        help="Execution mode (default: demo)"
    )
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to dataset CSV file (uses synthetic data if not provided)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to saved model checkpoint")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Compute device (default: auto)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory for results (default: outputs)")
    parser.add_argument("--dataset", type=str, default="paysim",
                        choices=["paysim", "tax"],
                        help="Dataset to use (default: paysim)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def setup_device(device_arg: str) -> str:
    """Determine the best available compute device."""
    import torch
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✓ Using GPU: {gpu_name}")
            print(f"  ✓ CUDA Version: {torch.version.cuda}")
        else:
            device = "cpu"
            print(f"  ✓ Using CPU (CUDA not available)")
    else:
        device = device_arg
        print(f"  ✓ Using device: {device}")
    return device


def print_banner():
    """Print the application banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███████╗██╗███╗   ██╗ █████╗  ██████╗ ███████╗███╗   ██╗   ║
║   ██╔════╝██║████╗  ██║██╔══██╗██╔════╝ ██╔════╝████╗  ██║   ║
║   █████╗  ██║██╔██╗ ██║███████║██║  ███╗█████╗  ██╔██╗ ██║   ║
║   ██╔══╝  ██║██║╚██╗██║██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ║
║   ██║     ██║██║ ╚████║██║  ██║╚██████╔╝███████╗██║ ╚████║   ║
║   ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ║
║                                                               ║
║      FinAgent-SLM v1.0.0 | Tax Fraud Detection System        ║
║      Ministry of Finance & Banking PSUs                       ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def run_demo(args):
    """Run a complete demonstration with synthetic data."""
    import torch
    import numpy as np
    from data.dataset_loader import MultiDatasetLoader
    from models.fraud_detector import FraudDetectionSLM
    from agents.orchestrator import FinAgentOrchestrator

    print("\n" + "=" * 60)
    print("  DEMO MODE: Full Pipeline Demonstration")
    print("=" * 60)
    device = setup_device(args.device)

    # Step 1: Load data via MultiDatasetLoader
    print("\n[1/5] Generating synthetic datasets...")
    loader = MultiDatasetLoader(batch_size=args.batch_size)
    data_loaders = loader.prepare_data(dataset_type="paysim")
    print(f"  ✓ PaySim data loaded with {len(data_loaders['train'])} training batches")

    # Step 2: Initialize model – infer input_dim from the first batch
    print("\n[2/5] Initializing FraudDetectionSLM model...")
    sample_batch = next(iter(data_loaders['train']))
    features_sample, _ = sample_batch
    input_dim = features_sample.shape[-1]
    model = FraudDetectionSLM(input_dim=input_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model parameters: {total_params:,}")
    print(f"  ✓ Model size: {total_params * 4 / 1e6:.1f} MB (FP32)")

    # Step 3: Quick training (3 epochs demo)
    print("\n[3/5] Quick training (3 epochs)...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(3):
        epoch_loss = 0.0
        n_batches = 0
        for batch_features, batch_labels in data_loaders['train']:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs['fraud_logits'], batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/3 - Loss: {avg_loss:.4f}")

    # Step 4: Evaluate
    print("\n[4/5] Evaluating model...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_features, batch_labels in data_loaders['test']:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            preds = torch.argmax(outputs['fraud_logits'], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    print(f"  ✓ Test Accuracy: {acc:.4f}")
    print(f"  ✓ Test F1 Score: {f1:.4f}")

    # Step 5: Multi-agent analysis
    print("\n[5/5] Running multi-agent fraud analysis...")
    orchestrator = FinAgentOrchestrator(model=model, device=device)
    # Create sample transactions for agent processing
    sample_transactions = []
    for i in range(5):
        sample_transactions.append({
            "id": f"TXN-{10000+i}",
            "type": np.random.choice(["TRANSFER", "CASH_OUT", "PAYMENT"]),
            "amount": float(np.random.lognormal(8, 2)),
            "oldbalanceOrg": float(np.random.lognormal(9, 2)),
            "newbalanceOrig": float(np.random.lognormal(7, 2)),
            "oldbalanceDest": float(np.random.lognormal(9, 1.5)),
            "newbalanceDest": float(np.random.lognormal(9, 1.5)),
        })
    results = orchestrator.process_batch(sample_transactions)
    flagged = sum(1 for r in results if r.get('is_suspicious', False))
    audit_trail = orchestrator.get_audit_trail()

    # Summary
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    print(f"\n  📊 Test Accuracy: {acc:.4f}")
    print(f"  📊 Test F1 Score: {f1:.4f}")
    print(f"  🔍 Transactions analyzed by agents: {len(results)}")
    print(f"  🚨 Flagged suspicious: {flagged}")
    print(f"  📋 Audit entries: {len(audit_trail)}")
    print(f"  ⚡ Model parameters: {total_params:,}")
    print(f"\n  Launch UI dashboard: python main.py --mode ui")
    print()


def run_training(args):
    """Run full model training pipeline."""
    import torch
    from data.dataset_loader import MultiDatasetLoader
    from models.fraud_detector import FraudDetectionSLM

    print("\n" + "=" * 60)
    print("  TRAINING MODE: Full Model Training")
    print("=" * 60)
    device = setup_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\n  Preparing {args.dataset} dataset...")
    loader = MultiDatasetLoader(batch_size=args.batch_size)
    data_loaders = loader.prepare_data(dataset_type=args.dataset)

    # Infer input dim
    sample_features, _ = next(iter(data_loaders['train']))
    input_dim = sample_features.shape[-1]
    model = FraudDetectionSLM(input_dim=input_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model: FraudDetectionSLM ({total_params:,} params, input_dim={input_dim})")

    # Train with simple loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 50.0]).to(device)  # heavy weight for fraud class
    )
    history = {'train_loss': [], 'val_accuracy': [], 'val_f1': []}

    print(f"\n  Training for {args.epochs} epochs on {device}...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss, n = 0.0, 0
        for features, labels in data_loaders['train']:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs['fraud_logits'], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        scheduler.step()
        avg_loss = epoch_loss / max(n, 1)
        history['train_loss'].append(avg_loss)

        # Validate
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for features, labels in data_loaders['val']:
                features = features.to(device)
                out = model(features)
                preds.extend(torch.argmax(out['fraud_logits'], dim=-1).cpu().numpy())
                truths.extend(labels.numpy())
        from sklearn.metrics import accuracy_score, f1_score
        val_acc = accuracy_score(truths, preds)
        val_f1 = f1_score(truths, preds, zero_division=0)
        history['val_accuracy'].append(val_acc)
        history['val_f1'].append(val_f1)
        print(f"  Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    # Save model
    model_path = output_dir / "fraud_detector_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {'input_dim': input_dim, 'd_model': 256, 'n_heads': 8, 'n_layers': 6},
        'training_history': history,
        'timestamp': datetime.now().isoformat()
    }, model_path)
    print(f"\n  ✓ Model saved to: {model_path}")

    # Evaluate on test set
    print("\n  Running test evaluation...")
    model.eval()
    preds, truths, probs_list = [], [], []
    with torch.no_grad():
        for features, labels in data_loaders['test']:
            features = features.to(device)
            out = model(features)
            p = torch.softmax(out['fraud_logits'], dim=-1)
            preds.extend(torch.argmax(p, dim=-1).cpu().numpy())
            probs_list.extend(p[:, 1].cpu().numpy())
            truths.extend(labels.numpy())
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    metrics = {
        'accuracy': accuracy_score(truths, preds),
        'precision': precision_score(truths, preds, zero_division=0),
        'recall': recall_score(truths, preds, zero_division=0),
        'f1': f1_score(truths, preds, zero_division=0),
    }
    try:
        metrics['auc_roc'] = roc_auc_score(truths, probs_list)
    except ValueError:
        metrics['auc_roc'] = 0.0

    print(f"\n  {'Metric':<20} {'Value':>10}")
    print(f"  {'─' * 32}")
    for key, value in metrics.items():
        print(f"  {key:<20} {value:>10.4f}")

    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "metrics": {k: float(v) for k, v in metrics.items()},
            "history": {k: [float(x) for x in v] for k, v in history.items()},
            "config": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr}
        }, f, indent=2)
    print(f"  ✓ Results saved to: {results_path}")


def run_evaluation(args):
    """Evaluate a trained model."""
    import torch
    from data.dataset_loader import MultiDatasetLoader
    from models.fraud_detector import FraudDetectionSLM

    print("\n" + "=" * 60)
    print("  EVALUATION MODE")
    print("=" * 60)
    device = setup_device(args.device)

    loader = MultiDatasetLoader(batch_size=args.batch_size)
    data_loaders = loader.prepare_data(dataset_type=args.dataset)
    sample_features, _ = next(iter(data_loaders['test']))
    input_dim = sample_features.shape[-1]
    model = FraudDetectionSLM(input_dim=input_dim).to(device)

    if args.model_path and Path(args.model_path).exists():
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded model from: {args.model_path}")
    else:
        print("  ⚠ No model checkpoint found, using random weights for demo")

    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for features, labels in data_loaders['test']:
            features = features.to(device)
            out = model(features)
            preds.extend(torch.argmax(out['fraud_logits'], dim=-1).cpu().numpy())
            truths.extend(labels.numpy())

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
    print(f"\n  Test Accuracy:  {accuracy_score(truths, preds):.4f}")
    print(f"  Test Precision: {precision_score(truths, preds, zero_division=0):.4f}")
    print(f"  Test Recall:    {recall_score(truths, preds, zero_division=0):.4f}")
    print(f"  Test F1:        {f1_score(truths, preds, zero_division=0):.4f}")
    print(f"\n{classification_report(truths, preds, target_names=['Legit', 'Fraud'], zero_division=0)}")


def run_agent(args):
    """Run multi-agent orchestration."""
    import numpy as np
    from agents.orchestrator import FinAgentOrchestrator

    print("\n" + "=" * 60)
    print("  MULTI-AGENT MODE: Fraud Analysis Pipeline")
    print("=" * 60)
    device = setup_device(args.device)

    orchestrator = FinAgentOrchestrator(model=None, device=device)

    # Generate sample transactions
    transactions = []
    for i in range(20):
        transactions.append({
            "id": f"TXN-{20000+i}",
            "type": np.random.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN"]),
            "amount": float(np.random.lognormal(8, 2)),
            "oldbalanceOrg": float(np.random.lognormal(9, 2)),
            "newbalanceOrig": float(np.random.lognormal(7, 2)),
            "oldbalanceDest": float(np.random.lognormal(9, 1.5)),
            "newbalanceDest": float(np.random.lognormal(9, 1.5)),
            "reported_income": float(np.random.lognormal(11, 0.8)),
        })

    print(f"\n  Processing {len(transactions)} transactions through multi-agent pipeline...")
    results = orchestrator.process_batch(transactions)

    suspicious = [r for r in results if r.get('is_suspicious', False)]
    print(f"\n  Results Summary:")
    print(f"  ✓ Total processed: {len(results)}")
    print(f"  ✓ Flagged suspicious: {len(suspicious)}")
    print(f"  ✓ Flag rate: {len(suspicious)/len(results):.1%}")

    audit = orchestrator.get_audit_trail()
    print(f"\n  Audit Trail ({len(audit)} entries):")
    for entry in audit[:5]:
        ts = entry.get('timestamp', 'N/A')
        if hasattr(ts, 'strftime'):
            ts = ts.strftime('%H:%M:%S')
        print(f"    [{ts}] {entry.get('action', 'N/A')} - Risk: {entry.get('risk_level', 'N/A')}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "audit_trail.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"\n  ✓ Audit trail saved to: {audit_path}")


def run_quantization(args):
    """Run quantization benchmarks."""
    import torch
    from models.fraud_detector import FraudDetectionSLM
    from models.quantizer import QuantizationMatrix

    print("\n" + "=" * 60)
    print("  QUANTIZATION MODE: Model Compression Benchmarks")
    print("=" * 60)

    model = FraudDetectionSLM(input_dim=15)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: FraudDetectionSLM ({total_params:,} params)")

    # Create dummy input for benchmarking
    dummy_input = torch.randn(1, 15)
    quant_matrix = QuantizationMatrix(model, dummy_input)
    results_df = quant_matrix.benchmark_all()

    print(f"\n  Quantization Benchmark Results:")
    print(f"  {'Method':<20} {'Size (MB)':<12} {'Latency (ms)':<14} {'Size Ratio':<12}")
    print(f"  {'─' * 58}")
    for _, row in results_df.iterrows():
        print(f"  {row['Method']:<20} {row['Size (MB)']:<12.2f} "
              f"{row['Latency (ms)']:<14.2f} {row['Size Ratio']:<12.2f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quant_path = output_dir / "quantization_matrix.json"
    results_df.to_json(quant_path, orient="records", indent=2)
    print(f"\n  ✓ Quantization matrix saved to: {quant_path}")


def run_ui(args):
    """Launch the Streamlit dashboard."""
    import subprocess
    print("\n  Launching FinAgent-SLM Dashboard...")
    print("  URL: http://localhost:8501")
    print("  Press Ctrl+C to stop\n")
    ui_path = PROJECT_ROOT / "ui" / "app.py"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(ui_path),
        "--server.port", "8501",
        "--server.headless", "true",
        "--theme.base", "dark"
    ])


def run_full_pipeline(args):
    """Run the complete pipeline: train → evaluate → quantize → agent."""
    print("\n" + "=" * 60)
    print("  FULL PIPELINE MODE")
    print("=" * 60)
    for i, (name, func) in enumerate([
        ("Training", run_training),
        ("Evaluation", run_evaluation),
        ("Quantization", run_quantization),
        ("Multi-Agent Analysis", run_agent),
    ], 1):
        print(f"\n{'━' * 60}")
        print(f"  Step {i}/4: {name}")
        print(f"{'━' * 60}")
        try:
            func(args)
        except Exception as e:
            print(f"  ⚠ Error in {name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    print(f"\n{'═' * 60}")
    print("  FULL PIPELINE COMPLETE")
    print(f"{'═' * 60}")
    print(f"\n  Launch dashboard: python main.py --mode ui")


def main():
    """Main entry point."""
    print_banner()
    args = parse_args()

    print(f"  Mode: {args.mode}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]}")

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
    except ImportError:
        print("  ⚠ PyTorch not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    mode_map = {
        "demo": run_demo, "train": run_training, "evaluate": run_evaluation,
        "ui": run_ui, "agent": run_agent, "quantize": run_quantization,
        "full": run_full_pipeline,
    }

    start_time = time.time()
    try:
        mode_map[args.mode](args)
    except KeyboardInterrupt:
        print("\n\n  ⚠ Interrupted by user")
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        elapsed = time.time() - start_time
        print(f"\n  ⏱ Elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
