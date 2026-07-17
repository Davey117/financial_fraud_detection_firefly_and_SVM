import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

class BinaryFireflyOptimizer:
    def __init__(self, N=10, max_iter=15, alpha=0.2, gamma=1.0, random_state=42):
        self.N = N
        self.max_iter = max_iter
        self.alpha = alpha
        self.gamma = gamma
        self.random_state = random_state
        np.random.seed(random_state)
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def _hamming_distance(self, x_i, x_j):
        """Calculate Hamming distance between two binary vectors."""
        return np.sum(x_i != x_j)
        
    def _evaluate_fitness(self, X_train, X_val, y_train, y_val, feature_mask):
        """
        Fitness is MCC of SVM-RBF on validation set.
        If all features are 0, return -1.0 (worst fitness).
        """
        if np.sum(feature_mask) == 0:
            return -1.0
            
        # Select features
        X_train_subset = X_train[:, feature_mask == 1]
        X_val_subset = X_val[:, feature_mask == 1]
        
        # Train SVM
        clf = SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=self.random_state)
        clf.fit(X_train_subset, y_train)
        
        # Predict and evaluate
        preds = clf.predict(X_val_subset)
        return matthews_corrcoef(y_val, preds)
        
    def optimize(self, X, y, progress_callback=None):
        """
        Run the BFA wrapper optimization.
        X, y should be the training subset used for fitness evaluation.
        """
        num_features = X.shape[1]
        
        # Split the given data into train and validation for fitness evaluation
        X_fit_train, X_fit_val, y_fit_train, y_fit_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        # Initialize fireflies randomly (binary)
        # N x num_features
        fireflies = np.random.randint(0, 2, size=(self.N, num_features))
        
        # Array to store fitness of each firefly
        fitness = np.zeros(self.N)
        
        # History tracking
        convergence_history = []
        global_best_position = None
        global_best_fitness = -1.0
        
        for t in range(self.max_iter):
            # Evaluate fitness
            for i in range(self.N):
                fitness[i] = self._evaluate_fitness(
                    X_fit_train, X_fit_val, y_fit_train, y_fit_val, fireflies[i]
                )
                
                # Update global best
                if fitness[i] > global_best_fitness:
                    global_best_fitness = fitness[i]
                    global_best_position = fireflies[i].copy()
            
            # Record history
            num_selected = int(np.sum(global_best_position))
            convergence_history.append({
                'Iteration': t + 1,
                'Best MCC': global_best_fitness,
                'Number of Features Selected': num_selected
            })
            
            # Firefly movement
            new_fireflies = np.copy(fireflies)
            
            for i in range(self.N):
                for j in range(self.N):
                    # If j is brighter than i, move i towards j
                    if fitness[j] > fitness[i]:
                        r = self._hamming_distance(fireflies[i], fireflies[j])
                        beta = np.exp(-self.gamma * (r ** 2))
                        
                        # Continuous movement
                        # v_i = beta * (x_j - x_i) + alpha * (rand - 0.5)
                        rand_term = self.alpha * (np.random.rand(num_features) - 0.5)
                        movement = beta * (fireflies[j] - fireflies[i]) + rand_term
                        
                        # Sigmoid transfer and binary conversion
                        prob = self._sigmoid(movement)
                        rand_draw = np.random.rand(num_features)
                        
                        # Update position
                        new_fireflies[i] = np.where(rand_draw < prob, 1, 0)
                        
            fireflies = new_fireflies
            
            # Callback for UI
            if progress_callback:
                progress_callback(t + 1, self.max_iter, global_best_fitness, num_selected)
                
        return global_best_position, convergence_history
