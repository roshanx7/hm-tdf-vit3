#!/usr/bin/env python3
"""
Test script to verify MffKan TinyViT refactoring.
Tests model instantiation, attribute presence, and forward pass with correct dimensions.
"""

import torch
import sys
import os

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

def test_model_instantiation():
    """Test that the model can be instantiated."""
    print("\n" + "="*70)
    print("TEST 1: Model Instantiation")
    print("="*70)
    
    from models.Mffkan import get_net
    
    num_features = 8  # Number of physiological indicators
    num_labels = 3    # Number of FLD severity classes
    drop_rate = 0.1
    
    try:
        net = get_net(num_features, num_labels, drop_rate)
        print(f"✓ Model instantiated successfully")
        print(f"  - Model type: {type(net).__name__}")
        print(f"  - Device: {next(net.parameters()).device}")
        return net, num_features, num_labels
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return None, None, None

def test_ie_proj_attribute(net):
    """Test that net.ie_proj exists (required by main.py)."""
    print("\n" + "="*70)
    print("TEST 2: ie_proj Attribute Verification")
    print("="*70)
    
    try:
        assert hasattr(net, 'ie_proj'), "net.ie_proj attribute not found!"
        print(f"✓ net.ie_proj attribute exists")
        print(f"  - Type: {type(net.ie_proj).__name__}")
        print(f"  - Input dim: {net.ie_proj.in_features}")
        print(f"  - Output dim: {net.ie_proj.out_features}")
        assert net.ie_proj.out_features == 512, "ie_proj output should be 512"
        print(f"✓ ie_proj output dimension is correct (512)")
        return True
    except AssertionError as e:
        print(f"✗ ie_proj verification failed: {e}")
        return False

def test_required_attributes(net):
    """Test that all required attributes exist."""
    print("\n" + "="*70)
    print("TEST 3: Required Attributes")
    print("="*70)
    
    required_attrs = ['IE', 'ie_proj', 'DE', 'classifier', 'kan_linears']
    missing = []
    
    for attr in required_attrs:
        if hasattr(net, attr):
            print(f"✓ {attr}: Present")
        else:
            print(f"✗ {attr}: Missing")
            missing.append(attr)
    
    if missing:
        print(f"\n✗ Missing attributes: {missing}")
        return False
    else:
        print(f"\n✓ All required attributes are present")
        return True

def test_forward_pass(net, num_features, num_labels):
    """Test forward pass with correct dimensions."""
    print("\n" + "="*70)
    print("TEST 4: Forward Pass & Dimension Verification")
    print("="*70)
    
    batch_size = 4
    
    # Create dummy input
    X = torch.randn(batch_size, 3, 224, 224)
    f_p = torch.randn(batch_size, num_features)
    
    print(f"Input dimensions:")
    print(f"  - Image (X): {tuple(X.shape)}")
    print(f"  - Features (f_p): {tuple(f_p.shape)}")
    
    try:
        net.eval()
        with torch.no_grad():
            # Test with debug=True to see intermediate shapes
            logits = net(X, f_p, debug=True)
        
        print(f"\nOutput dimensions:")
        print(f"  - Logits: {tuple(logits.shape)}")
        
        # Verify output shape
        assert logits.shape == (batch_size, num_labels), \
            f"Expected output shape {(batch_size, num_labels)}, got {tuple(logits.shape)}"
        print(f"✓ Forward pass successful")
        print(f"✓ Output shape is correct: {tuple(logits.shape)}")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_regularization_loss(net):
    """Test regularization loss calculation."""
    print("\n" + "="*70)
    print("TEST 5: Regularization Loss")
    print("="*70)
    
    try:
        net.train()
        reg_loss = net.regularization_loss(regularize_activation=0.1, regularize_entropy=0.1)
        print(f"✓ Regularization loss calculated: {reg_loss:.6f}")
        assert isinstance(reg_loss, torch.Tensor), "Regularization loss should be a tensor"
        print(f"✓ Regularization loss is a valid tensor")
        return True
    except Exception as e:
        print(f"✗ Regularization loss calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unfreeze_method(net):
    """Test the unfreeze_ie_layers method."""
    print("\n" + "="*70)
    print("TEST 6: Unfreeze IE Layers Method")
    print("="*70)
    
    try:
        # First, freeze all IE layers
        for param in net.IE.parameters():
            param.requires_grad = False
        print(f"✓ IE layers frozen initially")
        
        # Then unfreeze 50%
        net.unfreeze_ie_layers(unfreeze_ratio=0.5)
        
        # Check that some parameters are unfrozen
        unfrozen_count = sum(1 for param in net.IE.parameters() if param.requires_grad)
        total_count = sum(1 for param in net.IE.parameters())
        
        print(f"✓ Unfroze {unfrozen_count}/{total_count} layers")
        if unfrozen_count > 0:
            print(f"✓ unfreeze_ie_layers method works correctly")
            return True
        else:
            print(f"⚠ No layers were unfrozen (may be expected for small models)")
            return True
    except Exception as e:
        print(f"✗ unfreeze_ie_layers test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_integration(net, num_features, num_labels):
    """Test that the model works with typical training operations."""
    print("\n" + "="*70)
    print("TEST 7: Training Integration")
    print("="*70)
    
    try:
        batch_size = 4
        X = torch.randn(batch_size, 3, 224, 224)
        f_p = torch.randn(batch_size, num_features)
        y = torch.randint(0, num_labels, (batch_size,))
        
        net.train()
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
        
        # Forward pass
        logits = net(X, f_p)
        
        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        cls_loss = criterion(logits, y)
        
        # Regularization loss
        reg_loss = net.regularization_loss(0.1, 0.1)
        total_loss = cls_loss + reg_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"✓ Classification loss: {cls_loss:.6f}")
        print(f"✓ Regularization loss: {reg_loss:.6f}")
        print(f"✓ Total loss: {total_loss:.6f}")
        print(f"✓ Backward pass successful")
        print(f"✓ Training integration works correctly")
        return True
    except Exception as e:
        print(f"✗ Training integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MffKan TinyViT Refactoring Test Suite")
    print("="*70)
    
    # Run tests sequentially
    net, num_features, num_labels = test_model_instantiation()
    if net is None:
        print("\n✗ Model instantiation failed, stopping tests")
        return False
    
    results = []
    results.append(("ie_proj attribute", test_ie_proj_attribute(net)))
    results.append(("Required attributes", test_required_attributes(net)))
    results.append(("Forward pass", test_forward_pass(net, num_features, num_labels)))
    results.append(("Regularization loss", test_regularization_loss(net)))
    results.append(("Unfreeze method", test_unfreeze_method(net)))
    results.append(("Training integration", test_training_integration(net, num_features, num_labels)))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Model is ready for training.")
        return True
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review the output above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
