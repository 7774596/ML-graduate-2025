import numpy as np
import time
from model import Model


def load_data():
    """加载训练数据"""
    print("Loading training data...")
    data = np.load("data/train.npz")
    X_train = data["X_train"]
    y_train = data["y_train"]
    print(f"Training data loaded: X shape={X_train.shape}, y shape={y_train.shape}")
    return X_train, y_train


def evaluate_model(model, X, y):
    """评估模型性能"""
    print("\nEvaluating model...")
    start_time = time.time()
    y_pred = model.predict(X)
    prediction_time = time.time() - start_time
    
    accuracy = np.mean(y_pred == y)
    
    # 计算混淆矩阵
    n_classes = len(np.unique(y))
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y, y_pred):
        confusion_matrix[true_label, pred_label] += 1
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Prediction time: {prediction_time:.2f}s")
    print(f"Average time per sample: {prediction_time/len(X)*1000:.2f}ms")
    print("\nConfusion Matrix:")
    print(confusion_matrix)
    
    # 分析错误分类
    print("\nPer-class accuracy:")
    for i in range(n_classes):
        class_acc = confusion_matrix[i, i] / confusion_matrix[i, :].sum()
        print(f"  Class {i}: {class_acc:.4f} ({confusion_matrix[i, i]}/{confusion_matrix[i, :].sum()})")
    
    return accuracy, prediction_time, confusion_matrix


def train_and_save_model(kernel='rbf', C=1.0, gamma=0.01, degree=3, coef0=1, 
                         max_iter=100, model_path='svm_model.npz'):
    """
    训练并保存模型
    
    参数:
        kernel: 核函数类型 ('rbf', 'poly', 'linear')
        C: 正则化参数
        gamma: RBF核的gamma参数
        degree: 多项式核的degree参数
        coef0: 多项式核的coef0参数
        max_iter: 最大迭代次数
        model_path: 模型保存路径
    """
    # 加载数据
    X_train, y_train = load_data()
    
    # 创建并训练模型
    print(f"\n{'='*60}")
    print(f"Training SVM with kernel={kernel}, C={C}")
    if kernel == 'rbf':
        print(f"Gamma={gamma}")
    elif kernel == 'poly':
        print(f"Degree={degree}, Coef0={coef0}")
    print(f"{'='*60}\n")
    
    model = Model(
        C=C,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        max_iter=max_iter,
        tol=1e-3,
        verbose=True
    )
    
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"\nTraining completed in {train_time:.2f}s")
    
    # 评估模型
    accuracy, pred_time, conf_matrix = evaluate_model(model, X_train, y_train)
    
    # 保存模型
    model.save(model_path)
    
    return model, accuracy, pred_time


def grid_search_rbf():
    """
    对RBF核进行网格搜索
    """
    print("\n" + "="*60)
    print("Grid Search for RBF Kernel")
    print("="*60 + "\n")
    
    X_train, y_train = load_data()
    
    # 划分训练集和验证集
    n_samples = len(X_train)
    n_val = int(0.2 * n_samples)
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    X_train_sub = X_train[train_indices]
    y_train_sub = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    
    print(f"Training set: {len(X_train_sub)} samples")
    print(f"Validation set: {len(X_val)} samples\n")
    
    # 网格搜索参数
    C_values = [0.5, 1.0, 2.0, 5.0]
    gamma_values = [0.001, 0.005, 0.01, 0.05]
    
    best_accuracy = 0
    best_params = {}
    results = []
    
    for C in C_values:
        for gamma in gamma_values:
            print(f"\nTesting C={C}, gamma={gamma}")
            
            model = Model(
                C=C,
                kernel='rbf',
                gamma=gamma,
                max_iter=50,  # 减少迭代次数以加速搜索
                tol=1e-3,
                verbose=False
            )
            
            start_time = time.time()
            model.fit(X_train_sub, y_train_sub)
            train_time = time.time() - start_time
            
            # 在验证集上评估
            y_pred = model.predict(X_val)
            accuracy = np.mean(y_pred == y_val)
            
            print(f"  Train time: {train_time:.2f}s, Val accuracy: {accuracy:.4f}")
            
            results.append({
                'C': C,
                'gamma': gamma,
                'accuracy': accuracy,
                'train_time': train_time
            })
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'C': C, 'gamma': gamma}
    
    # 打印结果
    print("\n" + "="*60)
    print("Grid Search Results")
    print("="*60)
    print(f"{'C':<10}{'Gamma':<10}{'Accuracy':<15}{'Train Time':<15}")
    print("-"*60)
    for result in results:
        print(f"{result['C']:<10}{result['gamma']:<10}{result['accuracy']:<15.4f}{result['train_time']:<15.2f}")
    
    print("\n" + "="*60)
    print(f"Best parameters: C={best_params['C']}, gamma={best_params['gamma']}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print("="*60 + "\n")
    
    return best_params


def grid_search_poly():
    """
    对多项式核进行网格搜索
    """
    print("\n" + "="*60)
    print("Grid Search for Polynomial Kernel")
    print("="*60 + "\n")
    
    X_train, y_train = load_data()
    
    # 划分训练集和验证集
    n_samples = len(X_train)
    n_val = int(0.2 * n_samples)
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    X_train_sub = X_train[train_indices]
    y_train_sub = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    
    print(f"Training set: {len(X_train_sub)} samples")
    print(f"Validation set: {len(X_val)} samples\n")
    
    # 网格搜索参数
    C_values = [0.5, 1.0, 2.0]
    degree_values = [2, 3, 4]
    coef0_values = [0, 1]
    
    best_accuracy = 0
    best_params = {}
    results = []
    
    for C in C_values:
        for degree in degree_values:
            for coef0 in coef0_values:
                print(f"\nTesting C={C}, degree={degree}, coef0={coef0}")
                
                model = Model(
                    C=C,
                    kernel='poly',
                    degree=degree,
                    coef0=coef0,
                    max_iter=50,
                    tol=1e-3,
                    verbose=False
                )
                
                start_time = time.time()
                model.fit(X_train_sub, y_train_sub)
                train_time = time.time() - start_time
                
                # 在验证集上评估
                y_pred = model.predict(X_val)
                accuracy = np.mean(y_pred == y_val)
                
                print(f"  Train time: {train_time:.2f}s, Val accuracy: {accuracy:.4f}")
                
                results.append({
                    'C': C,
                    'degree': degree,
                    'coef0': coef0,
                    'accuracy': accuracy,
                    'train_time': train_time
                })
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'C': C, 'degree': degree, 'coef0': coef0}
    
    # 打印结果
    print("\n" + "="*60)
    print("Grid Search Results")
    print("="*60)
    print(f"{'C':<10}{'Degree':<10}{'Coef0':<10}{'Accuracy':<15}{'Train Time':<15}")
    print("-"*60)
    for result in results:
        print(f"{result['C']:<10}{result['degree']:<10}{result['coef0']:<10}"
              f"{result['accuracy']:<15.4f}{result['train_time']:<15.2f}")
    
    print("\n" + "="*60)
    print(f"Best parameters: C={best_params['C']}, degree={best_params['degree']}, coef0={best_params['coef0']}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print("="*60 + "\n")
    
    return best_params


def main():
    """主函数"""
    print("="*60)
    print("SVM Training Script")
    print("="*60)
    
    # 选项1: 使用预设参数直接训练
    print("\n[Option 1] Train with preset parameters")
    print("1. Train RBF kernel SVM")
    print("2. Train Polynomial kernel SVM")
    print("\n[Option 2] Grid search for best parameters")
    print("3. Grid search for RBF kernel")
    print("4. Grid search for Polynomial kernel")
    print("\n[Option 3] Train with custom parameters")
    print("5. Custom training")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        # RBF核 - 预设参数
        train_and_save_model(
            kernel='rbf',
            C=0.7,
            gamma=0.0013,
            max_iter=100,
            model_path='svm_rbf_model.npz'
        )
        
    elif choice == '2':
        # 多项式核 - 预设参数
        train_and_save_model(
            kernel='poly',
            C=1.0,
            degree=3,
            coef0=1,
            max_iter=100,
            model_path='svm_poly_model.npz'
        )
        
    elif choice == '3':
        # RBF核网格搜索
        best_params = grid_search_rbf()
        
        # 使用最佳参数在全部训练集上训练
        print("\nTraining final model with best parameters on full training set...")
        train_and_save_model(
            kernel='rbf',
            C=best_params['C'],
            gamma=best_params['gamma'],
            max_iter=150,
            model_path='svm_rbf_best_model.npz'
        )
        
    elif choice == '4':
        # 多项式核网格搜索
        best_params = grid_search_poly()
        
        # 使用最佳参数在全部训练集上训练
        print("\nTraining final model with best parameters on full training set...")
        train_and_save_model(
            kernel='poly',
            C=best_params['C'],
            degree=best_params['degree'],
            coef0=best_params['coef0'],
            max_iter=150,
            model_path='svm_poly_best_model.npz'
        )
        
    elif choice == '5':
        # 自定义参数
        print("\nCustom training")
        kernel = input("Enter kernel type (rbf/poly/linear): ").strip()
        C = float(input("Enter C value: "))
        
        if kernel == 'rbf':
            gamma = float(input("Enter gamma value: "))
            max_iter = int(input("Enter max iterations: "))
            model_path = input("Enter model save path (e.g., my_model.npz): ").strip()
            
            train_and_save_model(
                kernel='rbf',
                C=C,
                gamma=gamma,
                max_iter=max_iter,
                model_path=model_path
            )
            
        elif kernel == 'poly':
            degree = int(input("Enter degree: "))
            coef0 = float(input("Enter coef0: "))
            max_iter = int(input("Enter max iterations: "))
            model_path = input("Enter model save path (e.g., my_model.npz): ").strip()
            
            train_and_save_model(
                kernel='poly',
                C=C,
                degree=degree,
                coef0=coef0,
                max_iter=max_iter,
                model_path=model_path
            )
            
        elif kernel == 'linear':
            max_iter = int(input("Enter max iterations: "))
            model_path = input("Enter model save path (e.g., my_model.npz): ").strip()
            
            train_and_save_model(
                kernel='linear',
                C=C,
                max_iter=max_iter,
                model_path=model_path
            )
        else:
            print("Invalid kernel type!")
            return
    else:
        print("Invalid choice!")
        return
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
